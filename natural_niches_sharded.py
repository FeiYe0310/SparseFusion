import jax
import jax.numpy as jnp
from collections import defaultdict
from tqdm import tqdm
import torch
import torch.distributed as dist
import numpy as np
from datasets import load_from_disk
from typing import Callable
import os

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
)


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
            load_from_disk(GSM8K_DIR)

    if dist_enabled and dist.is_initialized():
        dist.barrier()

    dataset = load_from_disk(GSM8K_DIR)

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

    state = initialize_state(config)

    if is_main_process:
        for model in (model_1, model_2):
            score = train_eval_fn(model)
            state = sharded_update_state(state, score, model, alpha)

    if dist_enabled and dist.is_initialized():
        dist.barrier()

    results = []

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

            if is_main_process:
                parents_bf16 = sharded_sample_parents(
                    state, k1, alpha, use_matchmaker
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

            if dist_enabled and dist.is_initialized():
                if is_main_process:
                    child_tensor = torch.from_numpy(
                        np.array(child_bf16_main.astype(jnp.float32))
                    )
                else:
                    child_tensor = torch.empty(num_params_llm, dtype=torch.float32)

                dist.broadcast(child_tensor, src=0)
                child_bf16 = jnp.array(child_tensor.numpy()).astype(jnp.bfloat16)
            else:
                child_bf16 = child_bf16_main

            score = train_eval_fn(child_bf16)

            if is_main_process:
                state = sharded_update_state(state, score, child_bf16, alpha)

            if dist_enabled and dist.is_initialized():
                dist.barrier()

            if (i + 1) % 10 == 0:
                if is_main_process:
                    print(
                        f"\n--- [Step {i+1}/{total_forward_passes}] Evaluating full archive ---"
                    )

                for j in range(pop_size):
                    if is_main_process:
                        individual_params = state.archive[j]
                        params_tensor = torch.from_numpy(
                            np.array(individual_params.astype(jnp.float32))
                        )
                    else:
                        params_tensor = torch.empty(num_params_llm, dtype=torch.float32)

                    if dist_enabled and dist.is_initialized():
                        dist.broadcast(params_tensor, src=0)
                        params_bf16 = jnp.array(params_tensor.numpy()).astype(
                            jnp.bfloat16
                        )
                    else:
                        params_bf16 = state.archive[j]

                    test_scores_vector = test_eval_fn(params_bf16)

                    if is_main_process:
                        acc = jnp.mean(test_scores_vector)
                        print(
                            f"  > Archive Individual {j+1}/{pop_size} | Test Accuracy: {acc:.4f}"
                        )

        if dist_enabled and dist.is_initialized():
            dist.barrier()

    if is_main_process:
        if runs > 0:
            train_fitness = state.scores.mean(axis=1)
            best_individual_idx = jnp.argmax(train_fitness)
            best_params = state.archive[best_individual_idx]

            os.makedirs(RESULTS_DIR, exist_ok=True)
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(
                RESULTS_DIR, f"best_model_run_{runs}_{timestamp}.npz"
            )

            print(f"\n🏆 Saving the best model from the last run to: {save_path}")
            jnp.savez(save_path, params=best_params)
            print("✅ Model saved successfully.")

    return results


__all__ = ["run_natural_niches_sharded"]
