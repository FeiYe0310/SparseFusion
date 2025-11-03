"""
Model Utilities for SparseFusion

This module contains model conversion and distributed training utilities:
- PyTorch ↔ JAX parameter conversion
- Distributed training initialization
- Sharded array operations
- Async shard coordination
"""

import os
import jax
import jax.numpy as jnp
import torch
import torch.distributed as dist
from datetime import timedelta
from typing import Tuple, Optional

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


# Note: pytorch_to_jax_flattened and jax_flattened_to_pytorch_model
# are imported from helper_fn.py in the main code, so we keep them there
# to avoid circular dependencies. This module focuses on distributed
# and sharding utilities only.

