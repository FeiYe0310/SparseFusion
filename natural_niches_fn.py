import jax

# Enable 64-bit precision for JAX. This is crucial for handling large numbers,
# such as the total parameter count of a 7B model, which exceeds the int32 limit.
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from collections import defaultdict
from tqdm import tqdm
import torch
import numpy as np
from transformers import AutoTokenizer
from datasets import load_from_disk
from typing import Callable
import os
import math
import pickle
from contextlib import nullcontext

try:
    from jax.sharding import Mesh, NamedSharding, PartitionSpec
    from jax import make_array_from_callback
except ImportError:  # pragma: no cover - back-compat with older JAX
    Mesh = None
    NamedSharding = None
    PartitionSpec = None
    make_array_from_callback = None

try:
    default_device_ctx = jax.default_device
except AttributeError:  # pragma: no cover - older JAX
    from contextlib import contextmanager

    def default_device_ctx(_device):
        @contextmanager
        def _noop():
            yield

        return _noop()

# --- Imports for Multi-GPU ---
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler


from helper_fn import (
    crossover,
    crossover_without_splitpoint,
    mutate,
    get_pre_trained_models_and_skeleton,
    jax_flattened_to_pytorch_model,
)
from config import GSM8K_DIR, RESULTS_DIR


def _init_distributed_if_needed() -> tuple[int, int]:
    """Initialise torch.distributed with sensible defaults and backend selection."""

    if not dist.is_available():
        raise RuntimeError(
            "torch.distributed is not available in this build of PyTorch"
        )

    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()

    # Provide defaults so that single-process runs do not crash when torchrun env vars are missing.
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("LOCAL_RANK", os.environ.get("RANK", "0"))

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    backend = os.environ.get("TORCH_BACKEND")
    if not backend:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )

    return rank, world_size


def extract_answer(text: str) -> str:
    """
    从GSM8K生成的文本中提取数字答案
    GSM8K的标准格式：答案在####后面
    """
    import re
    
    # 尝试找到####后的答案
    if '####' in text:
        answer = text.split('####')[-1].strip()
    else:
        # 如果没有####，尝试提取最后一个数字
        answer = text.strip()
    
    # 提取数字（可能带逗号、小数点）
    numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', answer.replace(',', ''))
    if numbers:
        return numbers[-1]  # 返回最后一个数字
    return ""


def create_evaluation_fn_for_llm(
    model_skeleton: torch.nn.Module,
    param_shapes: list,
    tokenized_dataset,
    tokenizer: AutoTokenizer,
    batch_size: int = 4,
    distributed: bool = False,
    world_size: int = 1,
    rank: int = 0,
    use_real_eval: bool = False,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Creates an evaluation function for GSM8K.
    
    评估模式（use_real_eval控制）：
    - False: Token-level准确率（快速，teacher forcing）
    - True: Generation + Exact Match（慢但准确，真实评估）
    """
    base_model = (
        model_skeleton.module if hasattr(model_skeleton, "module") else model_skeleton
    )
    device = next(base_model.parameters()).device

    def collate_fn(batch):
        """
        自定义collate函数：
        - input_ids, attention_mask -> stack成tensor
        - answer_text -> 保持为字符串列表
        """
        import torch
        input_ids = torch.stack([torch.tensor(item["input_ids"]) for item in batch])
        attention_mask = torch.stack([torch.tensor(item["attention_mask"]) for item in batch])
        answer_texts = [item["answer_text"] for item in batch]
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "answer_text": answer_texts
        }
    
    def evaluation_fn(flat_params: jnp.ndarray) -> jnp.ndarray:
        # Get device from model
        device = next(base_model.parameters()).device
        
        # Restore parameters into the raw model (not the DDP wrapper)
        restored_model = jax_flattened_to_pytorch_model(
            flat_params, base_model, param_shapes
        )
        restored_model.eval()

        # Sampler ensures each GPU gets a different slice of data when distributed
        if distributed:
            sampler = DistributedSampler(
                tokenized_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
            )
            data_loader = DataLoader(
                tokenized_dataset, 
                batch_size=batch_size, 
                sampler=sampler,
                collate_fn=collate_fn if use_real_eval else None
            )
        else:
            data_loader = DataLoader(
                tokenized_dataset, 
                batch_size=batch_size, 
                shuffle=False,
                collate_fn=collate_fn if use_real_eval else None
            )

        local_scores = []
        with torch.no_grad():
            if use_real_eval:
                # ===== 真实GSM8K评估：Generation + Exact Match =====
                for batch in data_loader:
                    input_ids = batch["input_ids"].to(device)
                    answer_texts = batch["answer_text"]
                    
                    # 生成答案
                    generated_ids = restored_model.generate(
                        input_ids,
                        max_new_tokens=256,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                    
                    # 解码生成的文本
                    generated_texts = tokenizer.batch_decode(
                        generated_ids[:, input_ids.shape[1]:],
                        skip_special_tokens=True
                    )
                    
                    # 对比预测答案和ground truth
                    batch_scores = []
                    for gen_text, gt_text in zip(generated_texts, answer_texts):
                        pred_answer = extract_answer(gen_text)
                        gt_answer = extract_answer(gt_text)
                        is_correct = (pred_answer == gt_answer) and (pred_answer != "")
                        batch_scores.append(1.0 if is_correct else 0.0)
                    
                    local_scores.extend(batch_scores)
            else:
                # ===== Token准确率评估：快速但不准确 =====
                for batch in data_loader:
                    input_ids = batch["input_ids"].to(device)
                    labels = batch["labels"].to(device)

                    outputs = restored_model(input_ids=input_ids, labels=labels)
                    logits = outputs.logits

                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    predictions = torch.argmax(shift_logits, dim=-1)
                    mask = shift_labels != -100
                    correct_tokens = (predictions == shift_labels) & mask

                    total_correct = correct_tokens.sum(dim=1)
                    total_valid = mask.sum(dim=1)

                    accuracy_per_sequence = (
                        total_correct.float() / total_valid.float().clamp(min=1)
                    )
                    local_scores.append(accuracy_per_sequence.cpu())

        if not local_scores:
            return jnp.zeros(len(tokenized_dataset))

        # 转为tensor
        if use_real_eval:
            local_results_tensor = torch.tensor(local_scores, dtype=torch.float32)
        else:
            local_results_tensor = torch.cat(local_scores)

        if distributed:
            # Move to GPU for NCCL backend
            local_results_tensor = local_results_tensor.to(device)
            gathered_tensors = [
                torch.empty_like(local_results_tensor) for _ in range(world_size)
            ]
            dist.all_gather(gathered_tensors, local_results_tensor)
            full_results_tensor = torch.cat(gathered_tensors)[: len(tokenized_dataset)]
            # Move back to CPU for numpy conversion
            full_results_tensor = full_results_tensor.cpu()
        else:
            full_results_tensor = local_results_tensor[: len(tokenized_dataset)]

        return jnp.array(full_results_tensor.numpy())

    return evaluation_fn


def sample_parents(
    archive: jnp.ndarray,
    scores: jnp.ndarray,
    rand_key: jnp.ndarray,
    alpha: float,
    use_matchmaker: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    k1, k2 = jax.random.split(rand_key)
    z = scores.sum(axis=0)
    z = jnp.where(z, z, 1) ** alpha
    fitness_matrix = scores / z[None, :]
    fitness = jnp.sum(fitness_matrix, axis=1)
    probs = fitness / jnp.sum(fitness)
    # first parent
    if use_matchmaker:
        parent_1_idx = jax.random.choice(k1, probs.size, shape=(1,), p=probs)[0]
        # second parent
        match_score = jnp.maximum(
            0, fitness_matrix - fitness_matrix[parent_1_idx, :]
        ).sum(axis=1)
        probs = match_score / jnp.sum(match_score)
        parent_2_idx = jax.random.choice(k2, probs.size, shape=(1,), p=probs)[0]
    else:
        parent_2_idx, parent_1_idx = jax.random.choice(
            k1, probs.size, shape=(2,), p=probs
        )
    return archive[parent_1_idx], archive[parent_2_idx]


def update_archive(
    score: jnp.ndarray,
    param: jnp.ndarray,
    archive: jnp.ndarray,
    scores: jnp.ndarray,
    alpha: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:

    ext_scores = jnp.concatenate(
        [scores, score[None, ...]], axis=0
    )  # (pop_size + 1, num_datapoints)

    z = jnp.sum(ext_scores, axis=0) ** alpha  # (num_datapoints,)
    # avoid div by zero
    z = jnp.where(z, z, 1)

    ext_scores /= z[None, :]
    fitness = jnp.sum(ext_scores, axis=1)  # (pop_size + 1,)

    # get worst performing
    worst_ix = jnp.asarray(jnp.argmin(fitness), dtype=jnp.int32)
    scores_len = jnp.asarray(scores.shape[0], dtype=jnp.int32)
    update_mask = worst_ix < scores_len

    row_selector = (jnp.arange(archive.shape[0], dtype=jnp.int32) == worst_ix) & update_mask
    row_selector = row_selector[:, None]

    archive = jnp.where(row_selector, param[None, :], archive)
    scores = jnp.where(row_selector[: scores.shape[0], :], score[None, :], scores)

    return archive, scores


def run_natural_niches(
    runs: int,
    pop_size: int,
    total_forward_passes: int,
    store_train_results: bool,
    no_matchmaker: bool,
    no_crossover: bool,
    no_splitpoint: bool,
    alpha: float = 1.0,
    use_pre_trained: bool = False,
    model1_path: str = "",
    model2_path: str = "",
    distributed: bool = False,
    archive_backend: str = "gpu",
    use_real_gsm8k_eval: bool = False,
) -> list:
    archive_backend = archive_backend.lower()
    if archive_backend not in {"gpu", "cpu"}:
        raise ValueError("archive_backend must be 'gpu' or 'cpu'")

    use_matchmaker, use_crossover, use_splitpoint = (
        not no_matchmaker,
        not no_crossover,
        not no_splitpoint,
    )

    # --- Multi-GPU Distributed Setup ---
    # torchrun will set these environment variables; fall back to sane defaults otherwise.
    if distributed:
        rank, world_size = _init_distributed_if_needed()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
    else:
        rank, world_size = 0, 1
        local_rank = 0

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    is_main_process = rank == 0
    dist_enabled = distributed and world_size > 1

    # --- LLM & Data Loading ---
    if is_main_process:
        print("Loading tokenizer and models...")

    (
        model_1,
        model_2,
        param_shapes,
        tokenizer,
        model_skeleton,
    ) = get_pre_trained_models_and_skeleton(
        model1_path,
        model2_path,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_params_llm = model_1.shape[0]

    if is_main_process:
        print("Loading and preprocessing GSM8K dataset from local directory...")
        if dist_enabled:
            # Let main process prepare the dataset. It will be cached for others.
            # Path is now imported from the global config file.
            load_from_disk(GSM8K_DIR)

    if dist_enabled:
        dist.barrier()

    # Now all processes can load from the cache without disk contention.
    # Path is now imported from the global config file.
    dataset = load_from_disk(GSM8K_DIR)

    # 根据评估模式设置padding方式
    if use_real_gsm8k_eval:
        tokenizer.padding_side = 'left'  # decoder-only模型generation需要left padding
    
    def preprocess_function(examples):
        """数据预处理（根据评估模式不同而不同）"""
        if use_real_gsm8k_eval:
            # 真实评估：只tokenize问题，保存答案文本
            model_inputs = tokenizer(
                examples["question"], 
                max_length=256, 
                padding="max_length", 
                truncation=True
            )
            model_inputs["answer_text"] = examples["answer"]
        else:
            # Token准确率：tokenize问题+答案，用于teacher forcing
            inputs = [q + " " + a for q, a in zip(examples["question"], examples["answer"])]
            model_inputs = tokenizer(
                inputs,
                max_length=256,
                padding="max_length",
                truncation=True
            )
            model_inputs["labels"] = model_inputs["input_ids"].copy()
        
        return model_inputs

    # 快速实验：训练集200样本，测试集50样本
    tokenized_train_dataset = (
        dataset["train"]
        .select(range(200))
        .map(preprocess_function, batched=True, remove_columns=["question", "answer"])
    )
    tokenized_test_dataset = (
        dataset["test"]
        .select(range(50))
        .map(preprocess_function, batched=True, remove_columns=["question", "answer"])
    )
    
    # Token准确率模式需要set_format，真实评估模式用custom collate_fn
    if not use_real_gsm8k_eval:
        tokenized_train_dataset.set_format(type="torch")
        tokenized_test_dataset.set_format(type="torch")

    # --- Evaluation Setup ---
    if is_main_process:
        print("Setting up evaluation environment...")
    model_skeleton.to(device)

    if dist_enabled:
        # --- DDP Initialization Fix for Gloo Backend ---
        # The 'gloo' backend does not support bfloat16 for broadcasting during DDP initialization.
        # We temporarily cast the model to float32, initialize DDP, and then cast it back.
        model_skeleton.to(torch.float32)

        # Wrap the model for distributed evaluation.
        ddp_kwargs = {}
        if device.type == "cuda":
            ddp_kwargs.update(device_ids=[device.index], output_device=device.index)
        model_skeleton = DDP(model_skeleton, **ddp_kwargs)

        # Cast back to bfloat16 for efficient computation.
        model_skeleton.to(torch.bfloat16)
    else:
        model_skeleton.to(torch.bfloat16)

    # 清理generation_config中的无效参数（仅真实评估模式需要）
    if use_real_gsm8k_eval and hasattr(model_skeleton, 'generation_config'):
        if hasattr(model_skeleton.generation_config, 'temperature'):
            model_skeleton.generation_config.temperature = None
        if hasattr(model_skeleton.generation_config, 'top_p'):
            model_skeleton.generation_config.top_p = None
        if hasattr(model_skeleton.generation_config, 'top_k'):
            model_skeleton.generation_config.top_k = None

    model_skeleton.eval()

    train_eval_fn = create_evaluation_fn_for_llm(
        model_skeleton,
        param_shapes,
        tokenized_train_dataset,
        tokenizer,
        distributed=dist_enabled,
        world_size=world_size,
        rank=rank,
        use_real_eval=use_real_gsm8k_eval,
    )
    test_eval_fn = create_evaluation_fn_for_llm(
        model_skeleton,
        param_shapes,
        tokenized_test_dataset,
        tokenizer,
        distributed=dist_enabled,
        world_size=world_size,
        rank=rank,
        use_real_eval=use_real_gsm8k_eval,
    )

    archive_sharding = None
    scores_sharding = None
    mesh_context = nullcontext()
    cpu_archive_device = None

    if archive_backend == "gpu" and Mesh and NamedSharding and PartitionSpec:
        available_jax_devices = jax.devices()
        if len(available_jax_devices) > 1 and pop_size > 1:
            shard_axis_size = min(len(available_jax_devices), pop_size)
            if pop_size % shard_axis_size != 0:
                shard_axis_size = math.gcd(pop_size, shard_axis_size)

            if shard_axis_size and shard_axis_size > 1:
                shard_mesh = Mesh(
                    np.array(available_jax_devices[:shard_axis_size]),
                    ("archive_shard",),
                )
                archive_sharding = NamedSharding(
                    shard_mesh, PartitionSpec("archive_shard", None)
                )
                scores_sharding = NamedSharding(
                    shard_mesh, PartitionSpec("archive_shard", None)
                )
                mesh_context = shard_mesh
                if is_main_process:
                    print(
                        f"Sharding archive across {shard_axis_size} JAX devices."
                    )
    elif archive_backend == "cpu":
        cpu_devices = jax.devices("cpu")
        if not cpu_devices:
            raise RuntimeError("No CPU devices available for archive_backend='cpu'")
        cpu_archive_device = cpu_devices[0]

    with mesh_context:
        if is_main_process:
            print("Setup complete. Starting evolution.")

        if archive_sharding is not None:
            update_archive_fn = jax.jit(
                update_archive,
                in_shardings=(
                    None,
                    None,
                    archive_sharding,
                    scores_sharding,
                    None,
                ),
                out_shardings=(archive_sharding, scores_sharding),
            )
        elif archive_backend == "cpu":
            update_archive_fn = jax.jit(update_archive, backend="cpu")
        else:
            update_archive_fn = jax.jit(update_archive)

        results = []
        # The main evolution loop runs on all processes, but JAX ensures results are identical
        # since the random keys and inputs are synchronized.
        # --- Major Refactor for Memory Optimization ---
        # Only the main process will hold the memory-intensive JAX archive and perform
        # the evolutionary operations. Other processes act as pure evaluators.

        if is_main_process:
            # --- Archive Initialization (Main Process Only) ---
            print(f"--- Initializing Archive on Main Process (Rank {rank}) ---")
            archive_shape = (pop_size, num_params_llm)
            scores_shape = (pop_size, len(tokenized_train_dataset))
            if archive_sharding is not None:
                archive = _sharded_zeros(archive_shape, jnp.bfloat16, archive_sharding)
                scores = _sharded_zeros(scores_shape, jnp.float32, scores_sharding)
            elif archive_backend == "cpu":
                with default_device_ctx(cpu_archive_device):
                    archive = jnp.zeros(archive_shape, dtype=jnp.bfloat16)
                    scores = jnp.zeros(scores_shape, dtype=jnp.float32)
            else:
                archive = jnp.zeros(archive_shape, dtype=jnp.bfloat16)
                scores = jnp.zeros(scores_shape, dtype=jnp.float32)

            # Evaluate initial models to populate the archive
            for model in (model_1, model_2):
                # All processes participate in evaluation
                score = train_eval_fn(model)
                # But only main process updates its archive
                archive, scores = update_archive_fn(
                    score, model, archive, scores, alpha
                )
        else:
            archive = None
            scores = None
            if dist_enabled:
                # Other processes participate in the initial evaluation but don't hold an archive
                for model in (model_1, model_2):
                    train_eval_fn(model)

        # Barrier to ensure rank 0 has finished initializing the archive
        if dist_enabled:
            dist.barrier()

        for run in range(runs):
            if is_main_process:
                print(f"--- Starting Run {run+1}/{runs} ---")

            # This part of result handling only needs to be on the main process
            if is_main_process:
                results.append(defaultdict(list))

            seed = 42 + run
            # All processes must have the same random key sequence
            key = jax.random.PRNGKey(seed)

            progress_bar = tqdm(
                range(total_forward_passes),
                desc="Forward passes",
                disable=not is_main_process,
            )
            for i in progress_bar:
                # All processes advance their random state synchronously
                k1, k2, k3, key = jax.random.split(key, 4)

                # --- Child Generation (Main Process Only) ---
                if is_main_process:
                    # 1. Main process does all the JAX work
                    parents_bf16 = sample_parents(
                        archive, scores, k1, alpha, use_matchmaker
                    )
                    parents_f32 = (
                        parents_bf16[0].astype(jnp.float32),
                        parents_bf16[1].astype(jnp.float32),
                    )

                    if use_crossover:
                        if use_splitpoint:
                            child_f32 = crossover(parents_f32, k2)
                        else:
                            child_f32 = crossover_without_splitpoint(parents_f32, k2)
                    else:
                        child_f32 = parents_f32[0]

                    child_f32 = mutate(child_f32, k3)
                    child_bf16_main = child_f32.astype(jnp.bfloat16)
                else:
                    child_bf16_main = None

                if dist_enabled:
                    if is_main_process:
                        # 2. Convert JAX array to a float32 torch tensor for broadcasting
                        child_tensor = torch.from_numpy(
                            np.array(child_bf16_main.astype(jnp.float32))
                        )
                    else:
                        # Other processes prepare an empty float32 tensor to receive the data
                        child_tensor = torch.empty(num_params_llm, dtype=torch.float32)

                    # --- Broadcast Child to All Processes ---
                    dist.broadcast(child_tensor, src=0)

                    # All processes convert the received float32 tensor back to a bfloat16 JAX array
                    child_bf16 = jnp.array(child_tensor.numpy()).astype(jnp.bfloat16)
                else:
                    # Single-process execution uses the locally generated child
                    child_bf16 = child_bf16_main

                # --- Evaluation (All Processes) ---
                score = train_eval_fn(child_bf16)

                # --- Archive Update (Main Process Only) ---
                if is_main_process:
                    archive, scores = update_archive_fn(
                        score, child_bf16, archive, scores, alpha
                    )

                # Barrier to ensure archive update is complete before the next iteration
                if dist_enabled:
                    dist.barrier()

                # --- Periodic Full Archive Evaluation ---
                # Every 10 forward passes, evaluate all models currently in the archive.
                if (i + 1) % 10 == 0:
                    if is_main_process:
                        print(
                            f"\n--- [Step {i+1}/{total_forward_passes}] Evaluating full archive ---"
                        )

                    # This loop runs on all processes to keep them in sync for broadcasts.
                    for j in range(pop_size):
                        if dist_enabled:
                            # 1. Main process gets the individual parameters for broadcast.
                            if is_main_process:
                                individual_params = archive[j]
                                params_tensor = torch.from_numpy(
                                    np.array(individual_params.astype(jnp.float32))
                                )
                            else:
                                # Workers prepare an empty tensor.
                                params_tensor = torch.empty(
                                    num_params_llm, dtype=torch.float32
                                )

                            # 2. Broadcast the parameters from main process to all workers.
                            dist.broadcast(params_tensor, src=0)

                            # 3. All processes convert the tensor back to a JAX array.
                            params_bf16 = jnp.array(params_tensor.numpy()).astype(
                                jnp.bfloat16
                            )
                        else:
                            params_bf16 = archive[j]

                        # 4. All processes evaluate the model on the test set.
                        test_scores_vector = test_eval_fn(params_bf16)

                        # 5. The main process calculates and prints the result.
                        if is_main_process:
                            acc = jnp.mean(test_scores_vector)
                            # We can also log this to the results dictionary if needed
                            # For now, just printing for clear visibility.
                            print(
                                f"  > Archive Individual {j+1}/{pop_size} | Test Accuracy: {acc:.4f}"
                            )

                # --- Periodic Checkpoint Save (Every 50 steps to prevent data loss) ---
                if (i + 1) % 50 == 0 and is_main_process:
                    from datetime import datetime
                    checkpoint_dir = os.path.join(RESULTS_DIR, "checkpoints")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    
                    checkpoint_path = os.path.join(
                        checkpoint_dir, 
                        f"checkpoint_baseline_run{run+1}_step{i+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                    )
                    
                    checkpoint_data = {
                        "iteration": i + 1,
                        "run": run + 1,
                        "archive": np.array(archive),
                        "scores": np.array(scores),
                        "results": results,
                    }
                    
                    with open(checkpoint_path, "wb") as f:
                        pickle.dump(checkpoint_data, f)
                    
                    print(f"💾 Checkpoint saved: {checkpoint_path}")

                if dist_enabled:
                    dist.barrier()

        # Final barrier to ensure all processes finish before main process returns
        if dist_enabled:
            dist.barrier()

        # --- Save the final best model (main process only) ---
        if is_main_process:
            if runs > 0:
                # Find the best parameters from the final archive state of the last run
                train_fitness = scores.mean(axis=1)
                best_individual_idx = jnp.argmax(train_fitness)
                best_params = archive[best_individual_idx]

                # Directory path is now imported from the global config file.
                os.makedirs(RESULTS_DIR, exist_ok=True)

                # Save the best parameters to a file
                from datetime import datetime

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(
                    RESULTS_DIR, f"best_model_run_{run+1}_{timestamp}.npz"
                )

                print(f"\n🏆 Saving the best model from the last run to: {save_path}")
                jnp.savez(save_path, params=best_params)
                print("✅ Model saved successfully.")

        return results


def _sharded_zeros(shape: tuple[int, ...], dtype, sharding):
    if sharding is None or make_array_from_callback is None:
        return jnp.zeros(shape, dtype=dtype)

    def _cb(index):
        slice_dims = []
        for dim_idx, dim_size in zip(index, shape):
            if isinstance(dim_idx, slice):
                start = 0 if dim_idx.start is None else dim_idx.start
                stop = dim_size if dim_idx.stop is None else dim_idx.stop
                slice_dims.append(stop - start)
            else:
                slice_dims.append(1)
        return jnp.zeros(tuple(slice_dims), dtype=dtype)

    return make_array_from_callback(shape, sharding, _cb)

                    )

                # Barrier to ensure archive update is complete before the next iteration
                if dist_enabled:
                    dist.barrier()

                # --- Periodic Full Archive Evaluation ---
                # Every 10 forward passes, evaluate all models currently in the archive.
                if (i + 1) % 10 == 0:
                    if is_main_process:
                        print(
                            f"\n--- [Step {i+1}/{total_forward_passes}] Evaluating full archive ---"
                        )

                    # This loop runs on all processes to keep them in sync for broadcasts.
                    for j in range(pop_size):
                        if dist_enabled:
                            # 1. Main process gets the individual parameters for broadcast.
                            if is_main_process:
                                individual_params = archive[j]
                                params_tensor = torch.from_numpy(
                                    np.array(individual_params.astype(jnp.float32))
                                )
                            else:
                                # Workers prepare an empty tensor.
                                params_tensor = torch.empty(
                                    num_params_llm, dtype=torch.float32
                                )

                            # 2. Broadcast the parameters from main process to all workers.
                            dist.broadcast(params_tensor, src=0)

                            # 3. All processes convert the tensor back to a JAX array.
                            params_bf16 = jnp.array(params_tensor.numpy()).astype(
                                jnp.bfloat16
                            )
                        else:
                            params_bf16 = archive[j]

                        # 4. All processes evaluate the model on the test set.
                        test_scores_vector = test_eval_fn(params_bf16)

                        # 5. The main process calculates and prints the result.
                        if is_main_process:
                            acc = jnp.mean(test_scores_vector)
                            # We can also log this to the results dictionary if needed
                            # For now, just printing for clear visibility.
                            print(
                                f"  > Archive Individual {j+1}/{pop_size} | Test Accuracy: {acc:.4f}"
                            )

                # --- Periodic Checkpoint Save (Every 50 steps to prevent data loss) ---
                if (i + 1) % 50 == 0 and is_main_process:
                    from datetime import datetime
                    checkpoint_dir = os.path.join(RESULTS_DIR, "checkpoints")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    
                    checkpoint_path = os.path.join(
                        checkpoint_dir, 
                        f"checkpoint_baseline_run{run+1}_step{i+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                    )
                    
                    checkpoint_data = {
                        "iteration": i + 1,
                        "run": run + 1,
                        "archive": np.array(archive),
                        "scores": np.array(scores),
                        "results": results,
                    }
                    
                    with open(checkpoint_path, "wb") as f:
                        pickle.dump(checkpoint_data, f)
                    
                    print(f"💾 Checkpoint saved: {checkpoint_path}")

                if dist_enabled:
                    dist.barrier()

        # Final barrier to ensure all processes finish before main process returns
        if dist_enabled:
            dist.barrier()

        # --- Save the final best model (main process only) ---
        if is_main_process:
            if runs > 0:
                # Find the best parameters from the final archive state of the last run
                train_fitness = scores.mean(axis=1)
                best_individual_idx = jnp.argmax(train_fitness)
                best_params = archive[best_individual_idx]

                # Directory path is now imported from the global config file.
                os.makedirs(RESULTS_DIR, exist_ok=True)

                # Save the best parameters to a file
                from datetime import datetime

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(
                    RESULTS_DIR, f"best_model_run_{run+1}_{timestamp}.npz"
                )

                print(f"\n🏆 Saving the best model from the last run to: {save_path}")
                jnp.savez(save_path, params=best_params)
                print("✅ Model saved successfully.")

        return results


def _sharded_zeros(shape: tuple[int, ...], dtype, sharding):
    if sharding is None or make_array_from_callback is None:
        return jnp.zeros(shape, dtype=dtype)

    def _cb(index):
        slice_dims = []
        for dim_idx, dim_size in zip(index, shape):
            if isinstance(dim_idx, slice):
                start = 0 if dim_idx.start is None else dim_idx.start
                stop = dim_size if dim_idx.stop is None else dim_idx.stop
                slice_dims.append(stop - start)
            else:
                slice_dims.append(1)
        return jnp.zeros(tuple(slice_dims), dtype=dtype)

    return make_array_from_callback(shape, sharding, _cb)
