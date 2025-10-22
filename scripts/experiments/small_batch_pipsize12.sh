#!/usr/bin/env bash
set -euo pipefail

# Small Batch Test Script with PIPSize=12
# 快速测试脚本,使用较小的批次进行验证

# Offline / cache configuration
unset HF_ENDPOINT
unset HF_HUB_BASE_URL
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
umask 002

# NCCL safety defaults
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
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
USE_SINGLE_PROCESS_SHARDING="${USE_SINGLE_PROCESS_SHARDING:-0}"
ARCHIVE_BACKEND="${ARCHIVE_BACKEND:-gpu}"

# ============================================================================
# Small Batch Experiment Parameters
# ============================================================================
POP_SIZE="${POP_SIZE:-8}"
NUM_ITERATIONS="${NUM_ITERATIONS:-30}"
TOTAL_FORWARD_PASSES="${TOTAL_FORWARD_PASSES:-$((POP_SIZE * NUM_ITERATIONS))}"
RUNS="${RUNS:-1}"
OMEGA="${OMEGA:-0.7}"
BETA="${BETA:-0.3}"
PRUNING_SPARSITY="${PRUNING_SPARSITY:-0.0}"
EVAL_SUBSET_SIZE="${EVAL_SUBSET_SIZE:-12}"  # PIPSize=12

# Multi-task evaluation parameters
USE_MBPP_EVAL="${USE_MBPP_EVAL:-true}"
USE_MULT4_EVAL="${USE_MULT4_EVAL:-true}"
USE_MULT5_EVAL="${USE_MULT5_EVAL:-false}"
USE_BOOL_EVAL="${USE_BOOL_EVAL:-true}"
USE_BFCL_EVAL="${USE_BFCL_EVAL:-false}"

# Task weights (excluding GSM8K)
GSM8K_WEIGHT="${GSM8K_WEIGHT:-0.00}"
BFCL_WEIGHT="${BFCL_WEIGHT:-0.00}"
MBPP_WEIGHT="${MBPP_WEIGHT:-0.40}"
MULT4_WEIGHT="${MULT4_WEIGHT:-0.30}"
MULT5_WEIGHT="${MULT5_WEIGHT:-0.00}"
BOOL_WEIGHT="${BOOL_WEIGHT:-0.30}"

# Data paths
MBPP_DATA_PATH="${MBPP_DATA_PATH:-datasets/mbpp/mbpp_test.jsonl}"
BFCL_DATA_PATH="${BFCL_DATA_PATH:-bfcl/data/bfcl_test_200.json}"

# Model paths
MODEL1_PATH="${MODEL1_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"
MODEL2_PATH="${MODEL2_PATH:-Qwen/Qwen2.5-Coder-0.5B-Instruct}"

# Output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_DIR:-results/small_batch_pipsize12_${TIMESTAMP}}"

# ============================================================================
# Print configuration
# ============================================================================
echo "🚀 开始小批次运行测试 (PIPSize=12)"
echo "================================================"
echo "模型1: Qwen/Qwen2.5-0.5B-Instruct"
echo "模型2: Qwen/Qwen2.5-Coder-0.5B-Instruct"
echo "Population: $POP_SIZE"
echo "Iterations: $NUM_ITERATIONS"
echo "PIPSize: $EVAL_SUBSET_SIZE"
echo "评估样本数/任务: $EVAL_SUBSET_SIZE"
echo "启用任务: MBPP (${MBPP_WEIGHT}), Mult4 (${MULT4_WEIGHT}), Boolean (${BOOL_WEIGHT})"
echo "输出目录: $OUTPUT_DIR"
echo "================================================"

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
  --gsm8k_weight "$GSM8K_WEIGHT"
)

# Add multi-task evaluation parameters
if [[ "$USE_MBPP_EVAL" == "true" ]]; then
  MAIN_ARGS+=(
    --use_mbpp_eval
    --mbpp_data_path "$MBPP_DATA_PATH"
    --mbpp_weight "$MBPP_WEIGHT"
  )
fi

if [[ "$USE_MULT4_EVAL" == "true" ]]; then
  MAIN_ARGS+=(
    --use_mult4_eval
    --mult4_weight "$MULT4_WEIGHT"
  )
fi

if [[ "$USE_MULT5_EVAL" == "true" ]]; then
  MAIN_ARGS+=(
    --use_mult5_eval
    --mult5_weight "$MULT5_WEIGHT"
  )
fi

if [[ "$USE_BOOL_EVAL" == "true" ]]; then
  MAIN_ARGS+=(
    --use_bool_eval
    --bool_weight "$BOOL_WEIGHT"
  )
fi

if [[ "$USE_BFCL_EVAL" == "true" ]]; then
  MAIN_ARGS+=(
    --use_bfcl_eval
    --bfcl_data_path "$BFCL_DATA_PATH"
    --bfcl_weight "$BFCL_WEIGHT"
  )
fi

# Allow callers to append extra CLI args
if [[ $# -gt 0 ]]; then
  MAIN_ARGS+=("$@")
fi

# ============================================================================
# Main execution logic
# ============================================================================
if (( GPUS_PER_NODE > 1 )); then
  if [[ "$USE_SINGLE_PROCESS_SHARDING" != "0" ]]; then
    # Single-process with JAX sharding
    echo "[Small Batch] Launching single-process run with ${GPUS_PER_NODE} visible GPUs" >&2
    echo "[Small Batch] JAX will automatically shard archive across GPUs" >&2
    
    exec python main_sparsity_aware.py \
      --archive_backend "${ARCHIVE_BACKEND}" \
      "${MAIN_ARGS[@]}"
  else
    # Multi-process with PyTorch DDP
    if ! command -v torchrun >/dev/null 2>&1; then
      echo "[Small Batch] torchrun is required for multi-GPU execution but was not found in PATH." >&2
      exit 1
    fi

    echo "[Small Batch] Launching torchrun with ${GPUS_PER_NODE} processes on a single node" >&2
    echo "[Small Batch] PyTorch DDP will parallelize evaluation across GPUs" >&2
    
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
  # Single-process, single-GPU
  echo "[Small Batch] Launching single-process python run on 1 GPU" >&2
  exec python main_sparsity_aware.py \
    --archive_backend "${ARCHIVE_BACKEND}" \
    "${MAIN_ARGS[@]}"
fi

