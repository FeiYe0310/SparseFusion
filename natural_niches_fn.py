import jax

# Enable 64-bit precision for JAX. This is crucial for handling large numbers,
# such as the total parameter count of a 7B model, which exceeds the int32 limit.
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from collections import defaultdict
from tqdm import tqdm
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from typing import Callable
import os

# --- Imports for Multi-GPU ---
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler


from helper_fn import (
    crossover,
    crossover_without_splitpoint,
    mutate,
    get_pre_trained_models,
    jax_flattened_to_pytorch_model,
)
from config import GSM8K_DIR, RESULTS_DIR


def create_evaluation_fn_for_llm(
    model_skeleton: torch.nn.Module,
    param_shapes: list,
    tokenized_dataset,
    tokenizer: AutoTokenizer,
    batch_size: int = 4,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Creates an evaluation function for a given LLM.
    Handles unflattening and batched, distributed evaluation.
    """
    device = model_skeleton.device

    def evaluation_fn(flat_params: jnp.ndarray) -> jnp.ndarray:
        # Restore parameters into the raw model (not the DDP wrapper)
        # The DDP wrapper will automatically sync the updated weights.
        restored_model = jax_flattened_to_pytorch_model(
            flat_params, model_skeleton.module, param_shapes
        )
        restored_model.eval()

        # Sampler ensures each GPU gets a different slice of data
        sampler = DistributedSampler(
            tokenized_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=False,  # Keep order for consistent scoring
        )
        data_loader = DataLoader(
            tokenized_dataset, batch_size=batch_size, sampler=sampler
        )

        local_scores = []
        with torch.no_grad():
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

        # Each GPU now has a part of the results. We need to gather them all.
        all_gpu_scores = [
            torch.empty_like(local_scores[0]) for _ in range(dist.get_world_size())
        ]

        # This part is tricky. Let's gather all results as objects.
        gathered_objects = [None] * dist.get_world_size()

        # We need to send a single tensor from each process.
        local_results_tensor = torch.cat(local_scores)

        # Prepare a list to hold tensors from all processes
        gathered_tensors = [
            torch.empty_like(local_results_tensor) for _ in range(dist.get_world_size())
        ]

        dist.all_gather(gathered_tensors, local_results_tensor)

        # Concatenate and trim padding added by the sampler
        full_results_tensor = torch.cat(gathered_tensors)[: len(tokenized_dataset)]

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


@jax.jit
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
    worst_ix = jnp.argmin(fitness)
    update_mask = worst_ix < scores.shape[0]

    scores = scores.at[worst_ix].set(
        jax.lax.select(update_mask, score, scores[worst_ix])
    )
    archive = archive.at[worst_ix].set(
        jax.lax.select(update_mask, param, archive[worst_ix])
    )

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
) -> list:
    use_matchmaker, use_crossover, use_splitpoint = (
        not no_matchmaker,
        not no_crossover,
        not no_splitpoint,
    )

    # --- Multi-GPU Distributed Setup ---
    # torchrun will set these environment variables
    # Switching to 'gloo' backend as a more stable alternative to 'nccl'
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    # Let torchrun handle device assignment via LOCAL_RANK environment variable
    device = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(device)
    is_main_process = rank == 0

    # --- LLM & Data Loading ---
    if is_main_process:
        print("Loading tokenizer and models...")
    # Only the main process loads, flattens, and then broadcasts the initial models
    if is_main_process:
        model_1, model_2, param_shapes, tokenizer = get_pre_trained_models(
            model1_path,
            model2_path,
            return_tokenizer=True,
        )
        # JAX arrays can be broadcast as a list of objects
        initial_models_obj = [model_1, model_2, param_shapes, tokenizer]
        dist.broadcast_object_list(initial_models_obj, src=0)
    else:
        # Other processes receive the broadcasted data
        initial_models_obj = [None, None, None, None]
        dist.broadcast_object_list(initial_models_obj, src=0)
        model_1, model_2, param_shapes, tokenizer = initial_models_obj

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_params_llm = model_1.shape[0]

    if is_main_process:
        print("Loading and preprocessing GSM8K dataset from local directory...")
        # Let main process prepare the dataset. It will be cached for others.
        # Path is now imported from the global config file.
        load_dataset(GSM8K_DIR)

    # Barrier to ensure all processes wait for rank 0 to finish caching.
    dist.barrier()

    # Now all processes can load from the cache without disk contention.
    # Path is now imported from the global config file.
    dataset = load_dataset(GSM8K_DIR)

    def preprocess_function(examples):
        inputs = [q + " " + a for q, a in zip(examples["question"], examples["answer"])]
        model_inputs = tokenizer(
            inputs, max_length=256, padding="max_length", truncation=True
        )
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        return model_inputs

    tokenized_train_dataset = dataset["train"].map(
        preprocess_function, batched=True, remove_columns=["question", "answer"]
    )
    tokenized_test_dataset = (
        dataset["test"]
        .select(range(50))
        .map(preprocess_function, batched=True, remove_columns=["question", "answer"])
    )
    tokenized_train_dataset.set_format(type="torch")
    tokenized_test_dataset.set_format(type="torch")

    # --- Evaluation Setup ---
    if is_main_process:
        print("Setting up evaluation environment...")
        # Let main process prepare the model. It will be cached for others.
        AutoModelForCausalLM.from_pretrained(model1_path, torch_dtype=torch.bfloat16)

    # Barrier to ensure all processes wait for rank 0 to finish caching.
    dist.barrier()

    # Now all processes can load from the cache without disk contention.
    model_skeleton = AutoModelForCausalLM.from_pretrained(
        model1_path, torch_dtype=torch.bfloat16
    )
    model_skeleton.to(device)

    # --- DDP Initialization Fix for Gloo Backend ---
    # The 'gloo' backend does not support bfloat16 for broadcasting during DDP initialization.
    # We temporarily cast the model to float32, initialize DDP, and then cast it back.
    model_skeleton.to(torch.float32)

    # Wrap the model for distributed evaluation.
    model_skeleton = DDP(model_skeleton)

    # Cast back to bfloat16 for efficient computation.
    model_skeleton.to(torch.bfloat16)

    model_skeleton.eval()

    train_eval_fn = create_evaluation_fn_for_llm(
        model_skeleton, param_shapes, tokenized_train_dataset, tokenizer
    )
    test_eval_fn = create_evaluation_fn_for_llm(
        model_skeleton, param_shapes, tokenized_test_dataset, tokenizer
    )
    if is_main_process:
        print("Setup complete. Starting evolution.")

    results = []
    # The main evolution loop runs on all processes, but JAX ensures results are identical
    # since the random keys and inputs are synchronized.
    # --- Major Refactor for Memory Optimization ---
    # Only the main process will hold the memory-intensive JAX archive and perform
    # the evolutionary operations. Other processes act as pure evaluators.

    if is_main_process:
        # --- Archive Initialization (Main Process Only) ---
        print(f"--- Initializing Archive on Main Process (Rank {rank}) ---")
        archive = jnp.zeros([pop_size, num_params_llm], dtype=jnp.bfloat16)
        scores = jnp.zeros([pop_size, len(tokenized_train_dataset)], dtype=jnp.float32)

        # Evaluate initial models to populate the archive
        for model in (model_1, model_2):
            # All processes participate in evaluation
            score = train_eval_fn(model)
            # But only main process updates its archive
            archive, scores = update_archive(score, model, archive, scores, alpha)
    else:
        # Other processes participate in the initial evaluation but don't hold an archive
        for model in (model_1, model_2):
            train_eval_fn(model)
        archive = None
        scores = None

    # Barrier to ensure rank 0 has finished initializing the archive
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
                child_bf16 = child_f32.astype(jnp.bfloat16)

                # 2. Convert JAX array to a float32 torch tensor for broadcasting
                child_tensor = torch.from_numpy(
                    np.array(child_bf16.astype(jnp.float32))
                )
            else:
                # Other processes prepare an empty float32 tensor to receive the data
                child_tensor = torch.empty(num_params_llm, dtype=torch.float32)

            # --- Broadcast Child to All Processes ---
            dist.broadcast(child_tensor, src=0)

            # All processes convert the received float32 tensor back to a bfloat16 JAX array
            child_bf16 = jnp.array(child_tensor.numpy()).astype(jnp.bfloat16)

            # --- Evaluation (All Processes) ---
            score = train_eval_fn(child_bf16)

            # --- Archive Update (Main Process Only) ---
            if is_main_process:
                archive, scores = update_archive(
                    score, child_bf16, archive, scores, alpha
                )

            # Barrier to ensure archive update is complete before the next iteration
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
                    # 1. Main process gets the individual parameters for broadcast.
                    if is_main_process:
                        individual_params = archive[j]
                        params_tensor = torch.from_numpy(
                            np.array(individual_params.astype(jnp.float32))
                        )
                    else:
                        # Workers prepare an empty tensor.
                        params_tensor = torch.empty(num_params_llm, dtype=torch.float32)

                    # 2. Broadcast the parameters from main process to all workers.
                    dist.broadcast(params_tensor, src=0)

                    # 3. All processes convert the tensor back to a JAX array.
                    params_bf16 = jnp.array(params_tensor.numpy()).astype(jnp.bfloat16)

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

    # Final barrier to ensure all processes finish before main process returns
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
