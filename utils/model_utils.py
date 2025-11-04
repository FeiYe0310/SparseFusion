"""
Model Utilities for SparseFusion

This module contains model conversion and distributed training utilities:
- PyTorch ↔ JAX parameter conversion
- Distributed training initialization
- Evolutionary operations (crossover, mutation, slerp)
- Model loading and tokenizer merging
- Sharded array operations
- Async shard coordination

All helper functions from helper_fn.py have been integrated here.
"""

import os
import jax
import jax.numpy as jnp
import torch
import torch.distributed as dist
import numpy as np
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer
from datetime import timedelta
from typing import Tuple, Optional, List, Any, Union, Literal
from tqdm.auto import tqdm

# JAX sharding imports (optional, for compatibility)
try:
    from jax.sharding import Mesh, NamedSharding, PartitionSpec
    from jax import make_array_from_callback
except ImportError:
    Mesh = None
    NamedSharding = None
    PartitionSpec = None
    make_array_from_callback = None


def init_distributed_if_needed() -> Tuple[int, int]:
    """
    初始化 torch.distributed（如果需要）
    
    自动选择后端（NCCL for GPU, Gloo for CPU），设置超长超时时间
    
    Returns:
        (rank, world_size) 元组
    """
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

    # Set a very long timeout (7 days) to prevent NCCL timeout errors
    # Default is 10 minutes (600s), which is too short for long computations
    timeout = timedelta(days=7)

    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        timeout=timeout,
    )

    return rank, world_size


def apply_async_sync_to_matrix(
    archive: jnp.ndarray, coordinator
) -> jnp.ndarray:
    """
    将异步分片协调器的同步参数广播到种群archive
    
    Args:
        archive: 种群archive矩阵 (pop_size, num_params)
        coordinator: AsyncShardCoordinator实例
    
    Returns:
        更新后的archive
    """
    if archive is None:
        return archive

    global_params = coordinator.global_params()
    if global_params is None:
        return archive

    dtype = archive.dtype
    synced_nodes = coordinator.synced_nodes()
    updated = archive
    for idx in synced_nodes:
        start, end = coordinator.shard_slices[idx]
        shard_vals = jnp.asarray(global_params[start:end], dtype=dtype)
        shard_vals = jnp.broadcast_to(shard_vals, (archive.shape[0], end - start))
        updated = updated.at[:, start:end].set(shard_vals)
    return updated


def sharded_zeros(shape: Tuple[int, ...], dtype, sharding=None):
    """
    创建分片的零数组（用于JAX sharding）
    
    Args:
        shape: 数组形状
        dtype: 数据类型
        sharding: JAX sharding对象（可选）
    
    Returns:
        分片的零数组
    """
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


# ==============================================================================
# PyTorch ↔ JAX Parameter Conversion
# ==============================================================================

def pytorch_to_jax_flattened(
    model: torch.nn.Module,
) -> Tuple[jnp.ndarray, List[Tuple[Any, Any]], int]:
    """
    Flattens a PyTorch model's parameters into a single JAX array
    and returns the flattened parameters, shapes, and total parameter count.
    The final JAX array is cast to bfloat16 to save memory.
    """
    params_np = []
    param_shapes = []
    total_params = 0
    param_list = list(model.parameters())
    for p in tqdm(param_list, desc="Flattening PyTorch parameters", disable=False):
        # Convert to float32 for numpy compatibility, then to numpy array on CPU
        param_np = p.detach().cpu().to(torch.float32).numpy()
        params_np.append(param_np.flatten())
        total_params += p.numel()
        # Store the original torch dtype, not the numpy dtype
        param_shapes.append((p.shape, p.dtype))

    # Perform the concatenation in NumPy (CPU RAM) to avoid GPU OOM
    flat_params_np = np.concatenate(params_np)
    # Move the final, single flat array to a JAX array on the default device
    flat_params_f32 = jnp.asarray(flat_params_np)

    # Cast to bfloat16 to save significant memory in the JAX archive
    flat_params_bf16 = flat_params_f32.astype(jnp.bfloat16)

    return flat_params_bf16, param_shapes, total_params


def jax_flattened_to_pytorch_model(
    flat_params: jnp.ndarray,
    model_skeleton: torch.nn.Module,
    param_shapes: List[Tuple[Any, Any]],
) -> torch.nn.Module:
    """
    Loads flattened JAX parameters back into a PyTorch model skeleton
    by directly updating the model's parameter data, processing chunk by chunk
    to avoid creating a large intermediate float32 copy in RAM.
    """
    current_pos = 0

    with torch.no_grad():
        params_to_update = list(model_skeleton.parameters())
        if len(params_to_update) != len(param_shapes):
            raise ValueError(
                "Model structure mismatch: The number of parameters in the skeleton model and the shape specification do not match."
            )

        for i, (shape, dtype) in enumerate(param_shapes):
            # Calculate the number of elements for the current parameter
            param_numel = np.prod(shape).item()

            # Slice the corresponding chunk from the flattened bfloat16 JAX array
            chunk_bf16 = jax.lax.dynamic_slice_in_dim(
                flat_params, current_pos, param_numel
            )

            # Convert only the small chunk to float32 for numpy conversion
            chunk_f32_np = np.array(chunk_bf16.astype(jnp.float32))

            # Ensure the sliced chunk has the correct number of elements
            if params_to_update[i].numel() != param_numel:
                raise ValueError(
                    f"Shape mismatch at parameter {i}. Expected {params_to_update[i].numel()} elements, but got {param_numel} from shapes list."
                )

            # Reshape and convert back to a float32 PyTorch tensor first
            tensor_chunk = torch.from_numpy(chunk_f32_np.reshape(shape))

            # Now, cast to the original target dtype (e.g., bfloat16)
            tensor_chunk = tensor_chunk.to(dtype)

            # Update the parameter data in place
            params_to_update[i].data.copy_(tensor_chunk)

            # Move to the next position
            current_pos += param_numel

    return model_skeleton


# ==============================================================================
# Evolutionary Operations (SLERP, Crossover, Mutation)
# ==============================================================================

def slerp(val: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Spherical linear interpolation between two vectors"""
    # Normalize the inputs.
    norm_x = x / jnp.linalg.norm(x)
    norm_y = y / jnp.linalg.norm(y)

    # Cosine of the angle.
    dot = jnp.dot(norm_x, norm_y)
    omega = jnp.arccos(jnp.clip(dot, -1, 1))
    sin_omega = jnp.sin(omega)

    # Calculate scales for input vectors.
    scale_x = jnp.sin((1.0 - val) * omega) / sin_omega
    scale_y = jnp.sin(val * omega) / sin_omega

    # Linear interpolation weights.
    lin_scale_x = 1.0 - val
    lin_scale_y = val

    return jnp.where(
        sin_omega > 1e-6, scale_x * x + scale_y * y, lin_scale_x * x + lin_scale_y * y
    )


@jax.jit
def crossover_without_splitpoint(
    parents: tuple[jnp.ndarray, jnp.ndarray], rand_key: jnp.ndarray
) -> jnp.ndarray:
    """Crossover without splitpoint (uniform SLERP)"""
    w = jax.random.uniform(rand_key)
    return slerp(w, parents[0], parents[1])


def slerp_w_splitpoint(
    val: jnp.ndarray, split_point: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray
) -> jnp.ndarray:
    """SLERP with a splitpoint that flips the interpolation weight"""
    val = jnp.ones_like(x) * val
    mask = jnp.arange(len(val)) < split_point
    val = jnp.where(mask, val, 1 - val)

    # Normalize the inputs.
    norm_x = x / jnp.linalg.norm(x)
    norm_y = y / jnp.linalg.norm(y)

    # Cosine of the angle.
    dot = jnp.dot(norm_x, norm_y)
    omega = jnp.arccos(jnp.clip(dot, -1, 1))
    sin_omega = jnp.sin(omega)

    # Calculate scales for input vectors.
    scale_x = jnp.sin((1.0 - val) * omega) / sin_omega
    scale_y = jnp.sin(val * omega) / sin_omega

    # Linear interpolation weights.
    lin_scale_x = 1.0 - val
    lin_scale_y = val

    return jnp.where(
        sin_omega > 1e-6, scale_x * x + scale_y * y, lin_scale_x * x + lin_scale_y * y
    )


@jax.jit
def crossover(
    parents: tuple[jnp.ndarray, jnp.ndarray], rand_key: jnp.ndarray
) -> jnp.ndarray:
    """Crossover with random splitpoint"""
    k1, k2 = jax.random.split(rand_key)
    # Use int64 for the split point to handle models with more than 2^31 parameters.
    split_point = jax.random.randint(
        k1, shape=(), minval=0, maxval=parents[0].shape[0], dtype=jnp.int64
    )
    w = jax.random.uniform(k2)
    return slerp_w_splitpoint(w, split_point, parents[0], parents[1])


@jax.jit
def mutate(params: jnp.ndarray, rand_key: jnp.ndarray, std: float = 0.01):
    """Add Gaussian noise to parameters"""
    dtype = params.dtype
    std_scalar = jnp.asarray(std, dtype=dtype)
    noise = jax.random.normal(rand_key, shape=params.shape, dtype=dtype) * std_scalar
    return params + noise


# ==============================================================================
# Model Loading and Tokenizer Merging
# ==============================================================================

def _extend_embeddings_with_zeros(model: torch.nn.Module, new_vocab_size: int) -> None:
    """Extend model embeddings to new vocabulary size with zeros"""
    embedding = model.get_input_embeddings()
    if embedding is None:
        return

    old_vocab_size, _ = embedding.weight.shape
    if new_vocab_size <= old_vocab_size:
        return

    old_weight = embedding.weight.detach().clone()
    model.resize_token_embeddings(new_vocab_size)
    with torch.no_grad():
        updated_embedding = model.get_input_embeddings()
        updated_embedding.weight.zero_()
        updated_embedding.weight[:old_vocab_size] = old_weight


def _remap_embeddings_to_vocab(
    model: torch.nn.Module, original_vocab: dict[str, int], merged_vocab: dict[str, int]
) -> None:
    """Remap model embeddings to merged vocabulary"""
    embedding = model.get_input_embeddings()
    if embedding is None:
        return

    old_weight = embedding.weight.detach().clone()
    new_vocab_size = len(merged_vocab)
    model.resize_token_embeddings(new_vocab_size)

    with torch.no_grad():
        updated_embedding = model.get_input_embeddings()
        updated_embedding.weight.zero_()
        for token, original_idx in tqdm(
            sorted(original_vocab.items(), key=lambda item: item[1]),
            desc="Remapping embeddings",
            total=len(original_vocab),
        ):
            new_idx = merged_vocab.get(token)
            if new_idx is not None:
                updated_embedding.weight[new_idx] = old_weight[original_idx]


def merge_tokenizers_and_align_models(
    base_model: torch.nn.Module,
    base_tokenizer,
    other_model: torch.nn.Module,
    other_tokenizer,
):
    """Extend the base tokenizer with tokens from the other and align both models."""
    base_vocab = base_tokenizer.get_vocab()
    other_vocab = other_tokenizer.get_vocab()

    if base_vocab == other_vocab:
        return base_tokenizer, base_model, other_model

    tokens_to_add = [
        token
        for token, _ in sorted(other_vocab.items(), key=lambda item: item[1])
        if token not in base_vocab
    ]

    if tokens_to_add:
        base_tokenizer.add_tokens(tokens_to_add)
        base_vocab = base_tokenizer.get_vocab()

    merged_vocab = base_vocab
    merged_vocab_size = len(merged_vocab)

    _extend_embeddings_with_zeros(base_model, merged_vocab_size)
    _remap_embeddings_to_vocab(other_model, other_vocab, merged_vocab)

    for model in (base_model, other_model):
        if hasattr(model.config, "vocab_size"):
            model.config.vocab_size = merged_vocab_size

    for model in (base_model, other_model):
        tie_weights = getattr(model, "tie_weights", None)
        if callable(tie_weights):
            try:
                tie_weights()
            except Exception:
                pass

    return base_tokenizer, base_model, other_model


def _load_model(
    model_path: str, dtype: torch.dtype, device: torch.device
) -> torch.nn.Module:
    """Loads a model onto a specified device."""
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, trust_remote_code=True
        ).to(device)
    except Exception:
        return AutoModel.from_pretrained(
            model_path, torch_dtype=dtype, trust_remote_code=True
        ).to(device)


def get_pre_trained_models_and_skeleton(
    model1_path: str,
    model2_path: str,
    *,
    base_tokenizer: Literal["model1", "model2"] = "model1",
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, List[Tuple[Any, Any]], Any, torch.nn.Module]:
    """
    Loads two pre-trained models, merges their tokenizers (extending the selected base
    with the other's vocabulary), aligns the embedding weights to the merged vocabulary,
    and returns their flattened parameter vectors plus shape metadata. Additionally
    returns a PyTorch model skeleton whose architecture matches the merged tokenizer.
    """
    # For 7B models, it's recommended to use bfloat16 to save memory
    dtype = torch.bfloat16

    if device is None:
        # Default to CPU if no device is specified to prevent accidental GPU OOM
        target_device = torch.device("cpu")
        print("Warning: No device specified for model loading. Defaulting to CPU.")
    else:
        target_device = torch.device(device)

    print(
        f"Loading model 1 from: {model1_path} with dtype={dtype} onto device: {target_device}"
    )
    model1 = _load_model(model1_path, dtype, target_device)

    print(
        f"Loading model 2 from: {model2_path} with dtype={dtype} onto device: {target_device}"
    )
    model2 = _load_model(model2_path, dtype, target_device)

    # Ensure models are in evaluation mode
    model1.eval()
    model2.eval()

    tokenizer1 = AutoTokenizer.from_pretrained(
        model1_path, trust_remote_code=True, use_fast=False
    )
    tokenizer2 = AutoTokenizer.from_pretrained(
        model2_path, trust_remote_code=True, use_fast=False
    )

    # Ensure left padding for decoder-only models
    try:
        tokenizer1.padding_side = "left"
        tokenizer2.padding_side = "left"
    except Exception:
        pass

    # 可通过环境变量禁用分词器合并（默认禁用，减少乱码风险）
    disable_merge = os.environ.get("DISABLE_TOKENIZER_MERGE", "1") == "1"
    base_choice = base_tokenizer.lower()
    if base_choice not in {"model1", "model2"}:
        raise ValueError("base_tokenizer must be 'model1' or 'model2'")

    if disable_merge:
        # 使用基准模型的分词器，不做合并/重映射
        tokenizer_shared = tokenizer1 if base_choice == "model1" else tokenizer2
    else:
        # 合并词表并对齐嵌入（如需混用不同变体时开启）
        if base_choice == "model1":
            merged_tokenizer, model1, model2 = merge_tokenizers_and_align_models(
                model1, tokenizer1, model2, tokenizer2
            )
        else:
            merged_tokenizer, model2, model1 = merge_tokenizers_and_align_models(
                model2, tokenizer2, model1, tokenizer1
            )
        tokenizer_shared = merged_tokenizer

    print("Flattening parameters of both models...")
    flat_params1, param_shapes1, total_params1 = pytorch_to_jax_flattened(model1)
    flat_params2, param_shapes2, total_params2 = pytorch_to_jax_flattened(model2)

    if total_params1 != total_params2 or len(param_shapes1) != len(param_shapes2):
        raise ValueError(
            "Model structures differ (param count/shapes). 请使用相同架构与词表的模型，或将 DISABLE_TOKENIZER_MERGE=0 以启用合并对齐。"
        )

    # --- Memory Optimization ---
    del model2
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Freed secondary model from memory; retaining skeleton on CPU.")

    model_skeleton = model1.to(torch.device("cpu"))
    model_skeleton.eval()

    print("Model loading and flattening complete.")

    return (
        flat_params1,
        flat_params2,
        param_shapes1,
        tokenizer_shared,
        model_skeleton,
    )


def get_pre_trained_models(
    model1_path: str,
    model2_path: str,
    *,
    base_tokenizer: Literal["model1", "model2"] = "model1",
    return_tokenizer: bool = False,
    device: Optional[Union[str, torch.device]] = None,
) -> Union[
    Tuple[jnp.ndarray, jnp.ndarray, List[Tuple[Any, Any]]],
    Tuple[jnp.ndarray, jnp.ndarray, List[Tuple[Any, Any]], Any],
]:
    """Backward-compatible wrapper that omits the model skeleton from the return."""

    results = get_pre_trained_models_and_skeleton(
        model1_path,
        model2_path,
        base_tokenizer=base_tokenizer,
        device=device,
    )

    if return_tokenizer:
        flat_params1, flat_params2, param_shapes1, tokenizer_shared, _ = results
        return flat_params1, flat_params2, param_shapes1, tokenizer_shared

    flat_params1, flat_params2, param_shapes1, _, _ = results
    return flat_params1, flat_params2, param_shapes1
