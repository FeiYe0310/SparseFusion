import math
import os
from collections import defaultdict
from contextlib import nullcontext
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.distributed as dist
from datasets import load_from_disk
from torch.utils import dlpack as torch_dlpack
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)

from jax import dlpack as jax_dlpack
from jax.experimental.pjit import pjit
from jax.sharding import NamedSharding, PartitionSpec

from helper_fn import (
    crossover,
    crossover_without_splitpoint,
    mutate,
    get_pre_trained_models_and_skeleton,
    jax_flattened_to_pytorch_model,
)
from config import GSM8K_DIR, RESULTS_DIR
from natural_niches_fn import _init_distributed_if_needed, create_evaluation_fn_for_llm
from sharded_archive import (
    ShardedArchiveConfig,
    ShardedArchiveState,
    initialize_state,
    sample_parents as sharded_sample_parents,
    update_state as sharded_update_state,
    with_mesh,
)
from lib.async_shard import AsyncShardCoordinator


def run_natural_niches_sharded(
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
    async_num_nodes: Optional[int] = None,
    async_sync_interval: int = 10,
) -> list:
    use_matchmaker, use_crossover, use_splitpoint = (
        not no_matchmaker,
        not no_crossover,
        not no_splitpoint,
    )

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

    if is_main_process:
        print("Loading tokenizer and models on main process (CPU)...")
        # Load models on CPU on the main process to avoid OOM on GPUs.
        (
            model_1,
            model_2,
            param_shapes,
            tokenizer,
            model_skeleton,
        ) = get_pre_trained_models_and_skeleton(
            model1_path,
            model2_path,
            device="cpu",  # Explicitly load on CPU
        )
    else:
        # Other processes will receive the data via broadcast.
        # Initialize placeholders.
        model_1, model_2, param_shapes, tokenizer, model_skeleton = (
            None,
            None,
            None,
            None,
            None,
        )

    if dist_enabled and dist.is_initialized():
        # --- Broadcast tokenizer and metadata from main process ---
        objects_to_broadcast = [param_shapes, tokenizer, model_skeleton]
        if is_main_process:
            # The objects themselves are sent by the main process
            dist.broadcast_object_list(objects_to_broadcast, src=0)
        else:
            # Other processes receive the objects
            dist.broadcast_object_list(objects_to_broadcast, src=0)
            # Unpack the received objects
            param_shapes, tokenizer, model_skeleton = objects_to_broadcast

        # --- Broadcast the JAX model parameters ---
        # Determine num_params on the main process and broadcast it
        if is_main_process:
            num_params_llm = model_1.shape[0]
            num_params_tensor = torch.tensor(
                [num_params_llm], dtype=torch.long, device=device
            )
        else:
            num_params_tensor = torch.tensor([0], dtype=torch.long, device=device)

        dist.broadcast(num_params_tensor, src=0)
        num_params_llm = num_params_tensor.item()

        # Prepare tensors for broadcast
        if is_main_process:
            # Convert JAX arrays to PyTorch tensors for broadcasting
            model_1_tensor = torch_dlpack.from_dlpack(
                jax_dlpack.to_dlpack(model_1)
            ).to(device)
            model_2_tensor = torch_dlpack.from_dlpack(
                jax_dlpack.to_dlpack(model_2)
            ).to(device)
        else:
            # Create empty tensors on other processes to receive the data
            model_1_tensor = torch.empty(
                num_params_llm, dtype=torch.bfloat16, device=device
            )
            model_2_tensor = torch.empty(
                num_params_llm, dtype=torch.bfloat16, device=device
            )

        # Broadcast both model tensors
        dist.broadcast(model_1_tensor, src=0)
        dist.broadcast(model_2_tensor, src=0)
        dist.barrier()  # Synchronize all processes

        # Convert back to JAX arrays on each process (now on their respective GPUs)
        model_1 = jax_dlpack.from_dlpack(torch_dlpack.to_dlpack(model_1_tensor))
        model_2 = jax_dlpack.from_dlpack(torch_dlpack.to_dlpack(model_2_tensor))
    else:
        # For single-process execution, no broadcasting is needed.
        num_params_llm = model_1.shape[0]

    if is_main_process:
        print("Loading tokenizer and models... complete.")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_main_process:
        print("Loading and preprocessing GSM8K dataset from local directory...")
        if dist_enabled:
            load_from_disk(GSM8K_DIR)

    if dist_enabled and dist.is_initialized():
        dist.barrier()

    if async_num_nodes is None:
        async_num_nodes = world_size if dist_enabled else 1
    async_num_nodes = max(async_num_nodes, 1)
    async_enabled = async_num_nodes > 1
    async_coordinator: Optional[AsyncShardCoordinator] = None
    if async_enabled and is_main_process:
        async_coordinator = AsyncShardCoordinator(
            num_params=num_params_llm,
            num_nodes=async_num_nodes,
            sync_interval=async_sync_interval,
        )
        async_coordinator.bootstrap(model_1)

    dataset = load_from_disk(GSM8K_DIR)

    def _compute_max_length(ds) -> int:
        max_len = 0
        for split_name in ("train", "test"):
            split = ds[split_name]
            for example in split:
                text = example["question"] + " " + example["answer"]
                encoding = tokenizer(
                    text,
                    add_special_tokens=True,
                    padding=False,
                    truncation=False,
                )
                input_length = len(encoding["input_ids"])
                if input_length > max_len:
                    max_len = input_length
        return max_len

    max_sequence_length = _compute_max_length(dataset)
    if is_main_process:
        print(f"Max tokenized length for GSM8K: {max_sequence_length}")

    def preprocess_function(examples):
        inputs = [q + " " + a for q, a in zip(examples["question"], examples["answer"])]
        question_encodings = tokenizer(
            examples["question"],
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )
        model_inputs = tokenizer(
            inputs,
            max_length=max_sequence_length,
            padding="max_length",
            truncation=True,
            return_special_tokens_mask=True,
        )

        labels = []
        for idx, input_ids in enumerate(model_inputs["input_ids"]):
            label_row = input_ids.copy()
            special_mask = model_inputs["special_tokens_mask"][idx]
            question_tokens = question_encodings["input_ids"][idx]
            question_token_count = len(question_tokens)
            seen_non_special = 0
            for token_pos, is_special in enumerate(special_mask):
                if is_special:
                    continue
                if seen_non_special < question_token_count:
                    label_row[token_pos] = -100
                seen_non_special += 1
            labels.append(label_row)

        model_inputs.pop("special_tokens_mask", None)
        model_inputs["labels"] = labels
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

    if is_main_process:
        print("Setting up evaluation environment...")
    model_skeleton.to(device)
    model_skeleton.to(torch.bfloat16)
    model_skeleton.eval()

    train_eval_fn = create_evaluation_fn_for_llm(
        model_skeleton,
        param_shapes,
        tokenized_train_dataset,
        tokenizer,
        distributed=dist_enabled,
        world_size=world_size,
        rank=rank,
    )
    test_eval_fn = create_evaluation_fn_for_llm(
        model_skeleton,
        param_shapes,
        tokenized_test_dataset,
        tokenizer,
        distributed=dist_enabled,
        world_size=world_size,
        rank=rank,
    )

    if is_main_process:
        print("Setup complete. Starting evolution.")

    config = ShardedArchiveConfig(
        pop_size=pop_size,
        num_params=num_params_llm,
        num_datapoints=len(tokenized_train_dataset),
    )

    results = []

    axis_name = config.axis_name
    state: Optional[ShardedArchiveState] = None
    generate_child_cpu = None
    sample_parent_indices_cpu = None
    update_state_pjit = None
    replicated_sharding = None

    cpu_devices = jax.devices("cpu")
    if not cpu_devices:
        raise RuntimeError("No CPU device available for host-resident archive.")
    cpu_device = cpu_devices[0]
    state = None
    update_state_pjit = None
    replicated_sharding = None

    mesh_ctx = nullcontext()
    mesh_devices = None
    mesh = None
    mesh_axis_size = 1
    if is_main_process:
        available_devices = jax.devices()
        if not available_devices:
            raise RuntimeError("No JAX devices available for sharded execution.")

        mesh_axis_size = min(len(available_devices), pop_size)
        if mesh_axis_size == 0:
            mesh_axis_size = 1

        if pop_size % mesh_axis_size != 0:
            mesh_axis_size = math.gcd(pop_size, mesh_axis_size)

        if mesh_axis_size == 0:
            mesh_axis_size = 1

        if pop_size % mesh_axis_size != 0:
            raise ValueError(
                "Population size must be divisible by the mesh axis size for sharded execution."
            )

        mesh_devices = np.array(available_devices[:mesh_axis_size])
        mesh_ctx = with_mesh(mesh_devices, axis_name=axis_name)

    with mesh_ctx as mesh:
        mesh_axis_size = mesh.devices.size if mesh is not None else 1
        if is_main_process:
            replicated_sharding = NamedSharding(mesh, PartitionSpec()) if mesh else None

            state = initialize_state(config, device=cpu_device)

            def _fetch_archive_row(idx: int) -> jax.Array:
                return jax.device_put(state.archive[idx], cpu_device)

            def _sample_parent_indices(
                scores,
                key_sample,
                alpha_value,
                matchmaker_flag,
            ):
                scores = jnp.asarray(scores, dtype=jnp.float32)
                alpha_array = jnp.asarray(alpha_value, dtype=jnp.float32)
                k_first, k_second = jax.random.split(key_sample)

                z = scores.sum(axis=0)
                z = jnp.where(z, z, 1)
                z = z ** alpha_array
                fitness_matrix = scores / z[None, :]
                fitness = jnp.sum(fitness_matrix, axis=1)
                total_fitness = jnp.sum(fitness)
                total_fitness = jnp.where(total_fitness == 0, 1.0, total_fitness)
                probs = fitness / total_fitness
                probs = jnp.where(jnp.isnan(probs), 0.0, probs)

                parent_1_idx = jax.random.choice(
                    k_first, probs.shape[0], shape=(), p=probs
                )

                if matchmaker_flag:
                    match_score = jnp.maximum(
                        0, fitness_matrix - fitness_matrix[parent_1_idx, :]
                    ).sum(axis=1)
                    match_total = jnp.sum(match_score)
                    match_total = jnp.where(match_total == 0, 1.0, match_total)
                    match_probs = match_score / match_total
                    match_probs = jnp.where(jnp.isnan(match_probs), 0.0, match_probs)
                    parent_2_idx = jax.random.choice(
                        k_second, match_probs.shape[0], shape=(), p=match_probs
                    )
                else:
                    pair = jax.random.choice(
                        k_second, probs.shape[0], shape=(2,), p=probs
                    )
                    parent_2_idx = pair[0]
                    parent_1_idx = pair[1]

                return (
                    jnp.asarray(parent_1_idx, dtype=jnp.int32),
                    jnp.asarray(parent_2_idx, dtype=jnp.int32),
                )

            sample_parent_indices_cpu = jax.jit(
                _sample_parent_indices,
                backend="cpu",
                static_argnums=(3,),
            )

            def _generate_child_from_parents(
                parent_1,
                parent_2,
                key_crossover,
                key_mutate,
                crossover_flag,
                splitpoint_flag,
            ):
                parent_1 = jnp.asarray(parent_1, dtype=jnp.bfloat16)
                parent_2 = jnp.asarray(parent_2, dtype=jnp.bfloat16)
                parents = (parent_1, parent_2)
                child = parents[0]
                if crossover_flag:
                    if splitpoint_flag:
                        child = crossover(parents, key_crossover)
                    else:
                        child = crossover_without_splitpoint(
                            parents, key_crossover
                        )

                child = mutate(child, key_mutate)
                return child.astype(jnp.bfloat16)

            generate_child_cpu = jax.jit(
                _generate_child_from_parents,
                backend="cpu",
                static_argnums=(4, 5),
            )

            alpha_scalar = float(alpha)
            for seed_idx, model in enumerate((model_1, model_2)):
                score_vector = train_eval_fn(model)
                score_vector = jnp.asarray(score_vector, dtype=jnp.float32)
                state = sharded_update_state(
                    state,
                    score_vector,
                    model,
                    alpha_scalar,
                )
                if async_coordinator is not None:
                    owner_idx = seed_idx % async_coordinator.num_nodes
                    async_coordinator.commit(model, owner_idx)
                    if async_coordinator.synced():
                        _apply_async_sync_to_archive(state, async_coordinator)
        else:
            if dist_enabled and dist.is_initialized():
                for model in (model_1, model_2):
                    train_eval_fn(model)

        if dist_enabled and dist.is_initialized():
            dist.barrier()

        alpha_scalar = float(alpha)

        for run in range(runs):
            if is_main_process:
                print(f"--- Starting Run {run+1}/{runs} ---")
                results.append(defaultdict(list))

            seed = 42 + run
            key = jax.random.PRNGKey(seed)

            progress_bar = tqdm(
                range(total_forward_passes),
                desc="Forward passes",
                disable=not is_main_process,
            )
            for i in progress_bar:
                k1, k2, k3, key = jax.random.split(key, 4)
                shard_owner = None
                if async_coordinator is not None:
                    shard_owner = i % async_coordinator.num_nodes

                if is_main_process:
                    scores_cpu = np.asarray(state.scores, dtype=np.float32)
                    parent_indices = sample_parent_indices_cpu(
                        scores_cpu,
                        k1,
                        alpha_scalar,
                        use_matchmaker,
                    )
                    parent_1_idx = int(parent_indices[0])
                    parent_2_idx = int(parent_indices[1])

                    parent_1_cpu = np.asarray(
                        _fetch_archive_row(parent_1_idx), dtype=np.dtype("bfloat16")
                    )
                    parent_2_cpu = np.asarray(
                        _fetch_archive_row(parent_2_idx), dtype=np.dtype("bfloat16")
                    )

                    child_cpu = generate_child_cpu(
                        parent_1_cpu,
                        parent_2_cpu,
                        k2,
                        k3,
                        use_crossover,
                        use_splitpoint,
                    )
                    child_cpu = jnp.asarray(child_cpu, dtype=jnp.bfloat16)
                    if async_coordinator is not None and shard_owner is not None:
                        child_cpu = async_coordinator.prepare_candidate(
                            child_cpu, shard_owner
                        )
                    child_bf16_main = jax.device_put(child_cpu)
                else:
                    child_bf16_main = None

                if dist_enabled and dist.is_initialized():
                    if is_main_process:
                        child_tensor = torch_dlpack.from_dlpack(
                            jax_dlpack.to_dlpack(child_bf16_main)
                        ).to(device)
                    else:
                        child_tensor = torch.empty(
                            num_params_llm,
                            dtype=torch.bfloat16,
                            device=device,
                        )

                    dist.broadcast(child_tensor, src=0)
                    child_bf16 = jax_dlpack.from_dlpack(
                        torch_dlpack.to_dlpack(child_tensor)
                    )
                    if is_main_process:
                        child_bf16_main = child_bf16
                else:
                    child_bf16 = child_bf16_main

                score = train_eval_fn(child_bf16)

                if is_main_process:
                    score_array = jnp.asarray(score, dtype=jnp.float32)
                    score_cpu = jax.device_put(score_array, cpu_device)
                    child_cpu_host = jax.device_put(child_bf16_main, cpu_device)
                    state = sharded_update_state(
                        state,
                        score_cpu,
                        child_cpu_host,
                        alpha_scalar,
                    )
                    if async_coordinator is not None and shard_owner is not None:
                        async_coordinator.commit(child_cpu_host, shard_owner)
                        if async_coordinator.synced():
                            _apply_async_sync_to_archive(state, async_coordinator)

                if dist_enabled and dist.is_initialized():
                    dist.barrier()

                if (i + 1) % 10 == 0:
                    if is_main_process:
                        print(
                            f"\n--- [Step {i+1}/{total_forward_passes}] Evaluating full archive ---"
                        )

                    for j in range(pop_size):
                        if dist_enabled and dist.is_initialized():
                            if is_main_process:
                                params_bf16 = _fetch_archive_row(j)
                                params_tensor = torch_dlpack.from_dlpack(
                                    jax_dlpack.to_dlpack(params_bf16)
                                ).to(device)
                            else:
                                params_tensor = torch.empty(
                                    num_params_llm,
                                    dtype=torch.bfloat16,
                                    device=device,
                                )

                            dist.broadcast(params_tensor, src=0)
                            params_bf16 = jax_dlpack.from_dlpack(
                                torch_dlpack.to_dlpack(params_tensor)
                            )
                        else:
                            params_bf16 = _fetch_archive_row(j)

                        test_scores_vector = test_eval_fn(params_bf16)

                        if is_main_process:
                            acc = jnp.mean(test_scores_vector)
                            print(
                                f"  > Archive Individual {j+1}/{pop_size} | Test Accuracy: {acc:.4f}"
                            )

        if dist_enabled and dist.is_initialized():
            dist.barrier()

        if is_main_process and runs > 0 and state is not None:
            train_fitness = state.scores.mean(axis=1)
            best_individual_idx = int(jnp.argmax(train_fitness))
            best_params = _fetch_archive_row(best_individual_idx)
            best_params_host = jax.device_get(best_params)

            os.makedirs(RESULTS_DIR, exist_ok=True)
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(
                RESULTS_DIR, f"best_model_run_{runs}_{timestamp}.npz"
            )

            print(f"\n🏆 Saving the best model from the last run to: {save_path}")
            jnp.savez(save_path, params=best_params_host)
            print("✅ Model saved successfully.")

    return results


def _apply_async_sync_to_archive(
    state: ShardedArchiveState, coordinator: AsyncShardCoordinator
) -> None:
    """Overlay synchronised shard slices onto the archive buffer."""
    if state.archive is None:
        return

    global_params = coordinator.global_params()
    if global_params is None:
        return

    archive = state.archive
    dtype = archive.dtype
    synced_nodes = coordinator.synced_nodes()

    for idx in synced_nodes:
        start, end = coordinator.shard_slices[idx]
        shard_vals = jnp.asarray(global_params[start:end], dtype=dtype)
        shard_vals = jnp.broadcast_to(shard_vals, (archive.shape[0], end - start))
        archive = archive.at[:, start:end].set(shard_vals)

    state.archive = archive


__all__ = ["run_natural_niches_sharded"]
