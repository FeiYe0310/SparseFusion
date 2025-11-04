#!/usr/bin/env bash
set -euo pipefail

# Single-node launcher for BFCL Multi-Task Natural Niches
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

# ============================================================================
# BFCL Experiment Parameters (modify here)
# ============================================================================
POP_SIZE="${POP_SIZE:-5}"
TOTAL_FORWARD_PASSES="${TOTAL_FORWARD_PASSES:-3000}"
RUNS="${RUNS:-1}"
OMEGA="${OMEGA:-0.7}"
BETA="${BETA:-0.3}"
PRUNING_SPARSITY="${PRUNING_SPARSITY:-0.3}"
EVAL_SUBSET_SIZE="${EVAL_SUBSET_SIZE:-30}"

# BFCL-specific parameters
USE_BFCL_EVAL="${USE_BFCL_EVAL:-true}"
BFCL_DATA_PATH="${BFCL_DATA_PATH:-bfcl/data/bfcl_test_200.json}"
GSM8K_WEIGHT="${GSM8K_WEIGHT:-0.5}"
BFCL_WEIGHT="${BFCL_WEIGHT:-0.5}"

# 🔄 Dynamic Sparsity parameters (NEW!)
USE_DYNAMIC_SPARSITY="${USE_DYNAMIC_SPARSITY:-false}"
SPARSITY_MIN="${SPARSITY_MIN:-0.1}"
SPARSITY_MAX="${SPARSITY_MAX:-0.6}"
SPARSITY_T0="${SPARSITY_T0:-100}"
SPARSITY_T_MULT="${SPARSITY_T_MULT:-2}"

# Model paths
MODEL1_PATH="${MODEL1_PATH:-models/Qwen2.5-0.5B-Instruct}"
MODEL2_PATH="${MODEL2_PATH:-models/Qwen2.5-0.5B-Instruct}"

# Output directory
OUTPUT_DIR="${OUTPUT_DIR:-results_bfcl_multitask}"

# ============================================================================
# Construct main arguments
# ============================================================================
MAIN_ARGS=(
  --model1_path "$MODEL1_PATH"
  --model2_path "$MODEL2_PATH"
  --pop_size "$POP_SIZE"
  --total_forward_passes "$TOTAL_FORWARD_PASSES"
  --runs "$RUNS"
  --omega "$OMEGA"
  --beta "$BETA"
  --pruning_sparsity "$PRUNING_SPARSITY"
  --eval_subset_size "$EVAL_SUBSET_SIZE"
  --output_dir "$OUTPUT_DIR"
  --log_sparsity_stats
)

# Add BFCL parameters if enabled
if [[ "$USE_BFCL_EVAL" == "true" ]]; then
  MAIN_ARGS+=(
    --use_bfcl_eval
    --bfcl_data_path "$BFCL_DATA_PATH"
    --gsm8k_weight "$GSM8K_WEIGHT"
    --bfcl_weight "$BFCL_WEIGHT"
  )
fi

# Add Dynamic Sparsity parameters if enabled
if [[ "$USE_DYNAMIC_SPARSITY" == "true" ]]; then
  MAIN_ARGS+=(
    --use_dynamic_sparsity
    --sparsity_min "$SPARSITY_MIN"
    --sparsity_max "$SPARSITY_MAX"
    --sparsity_t0 "$SPARSITY_T0"
    --sparsity_t_mult "$SPARSITY_T_MULT"
  )
fi

# Allow callers to append extra CLI args
if [[ $# -gt 0 ]]; then
  MAIN_ARGS+=("$@")
fi

# Main execution logic
if (( GPUS_PER_NODE > 1 )); then
  if [[ "$USE_SINGLE_PROCESS_SHARDING" != "0" ]]; then
    # ===========================================================================
    # Mode 1: Single-process with JAX sharding across multiple GPUs
    # ===========================================================================
    echo "[BFCL] Launching single-process run with ${GPUS_PER_NODE} visible GPUs" >&2
    echo "[BFCL] JAX will automatically shard archive across GPUs" >&2
    if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
      echo "[BFCL] CUDA_VISIBLE_DEVICES not set; all GPUs on the node will be visible to the process." >&2
    else
      echo "[BFCL] Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
    fi
    
    # JAX sharding mode: evaluation on GPU, archive sharded across GPUs
    exec python natural_niches_sparsity_aware_fn.py \
      --archive_backend "${ARCHIVE_BACKEND}" \
      "${MAIN_ARGS[@]}"
  else
    # ===========================================================================
    # Mode 2: Multi-process with PyTorch DDP for evaluation
    # ===========================================================================
    if ! command -v torchrun >/dev/null 2>&1; then
      echo "[BFCL] torchrun is required for multi-GPU execution but was not found in PATH." >&2
      exit 1
    fi

    echo "[BFCL] Launching torchrun with ${GPUS_PER_NODE} processes on a single node" >&2
    echo "[BFCL] PyTorch DDP will parallelize evaluation across GPUs" >&2
    
    # PyTorch DDP mode: JAX on CPU, PyTorch evaluation distributed
    export JAX_PLATFORM_NAME=cpu
    exec torchrun \
      --standalone \
      --nproc_per_node="${GPUS_PER_NODE}" \
      natural_niches_sparsity_aware_fn.py \
      --distributed \
      --archive_backend "${ARCHIVE_BACKEND}" \
      "${MAIN_ARGS[@]}"
  fi
else
  # ===========================================================================
  # Mode 3: Single-process, single-GPU
  # ===========================================================================
  echo "[BFCL] Launching single-process python run on 1 GPU" >&2
  exec python natural_niches_sparsity_aware_fn.py \
    --archive_backend "${ARCHIVE_BACKEND}" \
    "${MAIN_ARGS[@]}"
fi

