#!/usr/bin/env bash
set -euo pipefail

# Single-node launcher for Sparsity-Aware Natural Niches
# Supports both single-GPU and multi-GPU execution with JAX sharding or PyTorch DDP

# Offline / cache configuration (override via environment when needed)
unset HF_ENDPOINT
unset HF_HUB_BASE_URL
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
umask 002

# NCCL safety defaults (harmless for single-node; tweak as desired)
unset NCCL_ASYNC_ERROR_HANDLING
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=WARN
export TORCH_NCCL_TRACE_BUFFER_SIZE=$((8*1024*1024))
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

if [[ -n "${NCCL_SOCKET_IFNAME:-}" ]]; then
  export NCCL_SOCKET_IFNAME
else
  unset NCCL_SOCKET_IFNAME
fi

export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export NCCL_NSOCKS_PERTHREAD=2
export NCCL_SOCKET_NTHREADS=2

# Workspace root and caches
ROOTPATH=${ROOTPATH:-/mnt/shared-storage-user/yefei}
cd "$ROOTPATH/SparseFusion"

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=$ROOTPATH/cache
export WANDB_CACHE_DIR=$ROOTPATH/cache
export TORCH_EXTENSION_DIR=$ROOTPATH/cache

export PATH="$ROOTPATH/miniconda3/bin:$PATH"
export PATH="$ROOTPATH/miniconda3/envs/sparsefusion/bin:$PATH"

# Configuration
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
USE_SINGLE_PROCESS_SHARDING="${USE_SINGLE_PROCESS_SHARDING:-0}"
ARCHIVE_BACKEND="${ARCHIVE_BACKEND:-gpu}"

# Allow callers to append extra CLI args for main_sparsity_aware.py
MAIN_ARGS=("$@")

# Main execution logic
if (( GPUS_PER_NODE > 1 )); then
  if [[ "$USE_SINGLE_PROCESS_SHARDING" != "0" ]]; then
    # ===========================================================================
    # Mode 1: Single-process with JAX sharding across multiple GPUs
    # ===========================================================================
    echo "[SparseFusion] Launching single-process run with ${GPUS_PER_NODE} visible GPUs" >&2
    echo "[SparseFusion] JAX will automatically shard archive across GPUs" >&2
    if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
      echo "[SparseFusion] CUDA_VISIBLE_DEVICES not set; all GPUs on the node will be visible to the process." >&2
    else
      echo "[SparseFusion] Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
    fi
    
    # JAX sharding mode: evaluation on GPU, archive sharded across GPUs
    exec python main_sparsity_aware.py \
      --archive_backend "${ARCHIVE_BACKEND}" \
      "${MAIN_ARGS[@]}"
  else
    # ===========================================================================
    # Mode 2: Multi-process with PyTorch DDP for evaluation
    # ===========================================================================
    if ! command -v torchrun >/dev/null 2>&1; then
      echo "[SparseFusion] torchrun is required for multi-GPU execution but was not found in PATH." >&2
      exit 1
    fi

    echo "[SparseFusion] Launching torchrun with ${GPUS_PER_NODE} processes on a single node" >&2
    echo "[SparseFusion] PyTorch DDP will parallelize evaluation across GPUs" >&2
    
    # PyTorch DDP mode: JAX on CPU, PyTorch evaluation distributed
    export JAX_PLATFORM_NAME=cpu
    exec torchrun \
      --standalone \
      --nproc_per_node="${GPUS_PER_NODE}" \
      main_sparsity_aware.py \
      --distributed \
      --archive_backend "${ARCHIVE_BACKEND}" \
      "${MAIN_ARGS[@]}"
  fi
else
  # ===========================================================================
  # Mode 3: Single-process, single-GPU
  # ===========================================================================
  echo "[SparseFusion] Launching single-process python run on 1 GPU" >&2
  exec python main_sparsity_aware.py \
    --archive_backend "${ARCHIVE_BACKEND}" \
    "${MAIN_ARGS[@]}"
fi

