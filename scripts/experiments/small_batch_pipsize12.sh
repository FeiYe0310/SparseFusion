#!/usr/bin/env bash
set -euo pipefail

# Small batch test with pipsize=12 - Multi-GPU support
# 小批次运行测试 (pipsize=12) - 支持多卡

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
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
USE_SINGLE_PROCESS_SHARDING="${USE_SINGLE_PROCESS_SHARDING:-0}"
ARCHIVE_BACKEND="${ARCHIVE_BACKEND:-gpu}"

# ============================================================================
# Small Batch Experiment Parameters (pipsize=12)
# ============================================================================
POP_SIZE="${POP_SIZE:-8}"
TOTAL_FORWARD_PASSES="${TOTAL_FORWARD_PASSES:-600}"  # 50 iterations
RUNS="${RUNS:-1}"
OMEGA="${OMEGA:-1.0}"     # baseline: 只看fitness
BETA="${BETA:-0.0}"
TAU="${TAU:-1.0}"
PRUNING_SPARSITY="${PRUNING_SPARSITY:-0.0}"  # 不启用pruning
EVAL_SUBSET_SIZE="${EVAL_SUBSET_SIZE:-12}"   # 每个任务12个样本
PIPSIZE="${PIPSIZE:-12}"

# Multi-task weights
USE_MULTI_TASK="${USE_MULTI_TASK:-true}"
GSM8K_WEIGHT="${GSM8K_WEIGHT:-0.50}"

USE_BFCL_EVAL="${USE_BFCL_EVAL:-true}"
BFCL_WEIGHT="${BFCL_WEIGHT:-0.00}"
BFCL_DATA_PATH="${BFCL_DATA_PATH:-bfcl/data/bfcl_test_200.json}"

USE_MBPP_EVAL="${USE_MBPP_EVAL:-true}"
MBPP_WEIGHT="${MBPP_WEIGHT:-0.50}"

USE_MULT4_EVAL="${USE_MULT4_EVAL:-true}"
MULT4_WEIGHT="${MULT4_WEIGHT:-0.25}"

USE_MULT5_EVAL="${USE_MULT5_EVAL:-true}"
MULT5_WEIGHT="${MULT5_WEIGHT:-0.00}"

USE_BOOL_EVAL="${USE_BOOL_EVAL:-true}"
BOOL_WEIGHT="${BOOL_WEIGHT:-0.25}"

# Model paths
MODEL1_PATH="${MODEL1_PATH:-models/Qwen2.5-0.5B-Instruct}"
MODEL2_PATH="${MODEL2_PATH:-models/Qwen2.5-0.5B-Instruct}"

# Output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_DIR:-results/small_batch_pipsize12_${TIMESTAMP}}"
mkdir -p "$OUTPUT_DIR"

# ============================================================================
# Construct main arguments
# ============================================================================
MAIN_ARGS=(
  --model1_path "$MODEL1_PATH"
  --model2_path "$MODEL2_PATH"
  --pop_size "$POP_SIZE"
  --total_forward_passes "$TOTAL_FORWARD_PASSES"
  --runs "$RUNS"
  --pipsize "$PIPSIZE"
  --sparsity_aware
  --omega "$OMEGA"
  --beta "$BETA"
  --tau "$TAU"
  --pruning_sparsity "$PRUNING_SPARSITY"
  --eval_subset_size "$EVAL_SUBSET_SIZE"
  --save_dir "$OUTPUT_DIR"
)

# Add multi-task parameters
if [[ "$USE_MULTI_TASK" == "true" ]]; then
  MAIN_ARGS+=(
    --use_multi_task
    --gsm8k_weight "$GSM8K_WEIGHT"
  )
fi

# BFCL
if [[ "$USE_BFCL_EVAL" == "true" ]]; then
  MAIN_ARGS+=(
    --use_bfcl_eval
    --bfcl_data_path "$BFCL_DATA_PATH"
    --bfcl_weight "$BFCL_WEIGHT"
  )
fi

# MBPP
if [[ "$USE_MBPP_EVAL" == "true" ]]; then
  MAIN_ARGS+=(
    --use_mbpp_eval
    --mbpp_weight "$MBPP_WEIGHT"
  )
fi

# Mult4
if [[ "$USE_MULT4_EVAL" == "true" ]]; then
  MAIN_ARGS+=(
    --use_mult4_eval
    --mult4_weight "$MULT4_WEIGHT"
  )
fi

# Mult5
if [[ "$USE_MULT5_EVAL" == "true" ]]; then
  MAIN_ARGS+=(
    --use_mult5_eval
    --mult5_weight "$MULT5_WEIGHT"
  )
fi

# Boolean
if [[ "$USE_BOOL_EVAL" == "true" ]]; then
  MAIN_ARGS+=(
    --use_bool_eval
    --bool_weight "$BOOL_WEIGHT"
  )
fi

# Allow callers to append extra CLI args
if [[ $# -gt 0 ]]; then
  MAIN_ARGS+=("$@")
fi

# Print configuration
echo "🚀 小批次运行测试 (PIPSize=${PIPSIZE})"
echo "================================================"
echo "模型: ${MODEL1_PATH}"
echo "Population: ${POP_SIZE}"
echo "Total Forward Passes: ${TOTAL_FORWARD_PASSES}"
echo "PIPSize: ${PIPSIZE}"
echo "评估样本数/任务: ${EVAL_SUBSET_SIZE}"
echo "GPUs: ${GPUS_PER_NODE}"
echo "输出目录: ${OUTPUT_DIR}"
echo "================================================"
echo "任务权重:"
echo "  GSM8K: ${GSM8K_WEIGHT}"
echo "  BFCL: ${BFCL_WEIGHT}"
echo "  MBPP: ${MBPP_WEIGHT}"
echo "  Mult4: ${MULT4_WEIGHT}"
echo "  Mult5: ${MULT5_WEIGHT}"
echo "  Boolean: ${BOOL_WEIGHT}"
echo "================================================"

# Main execution logic
if (( GPUS_PER_NODE > 1 )); then
  if [[ "$USE_SINGLE_PROCESS_SHARDING" != "0" ]]; then
    # Single-process with JAX sharding
    echo "[INFO] Launching single-process run with ${GPUS_PER_NODE} visible GPUs" >&2
    echo "[INFO] JAX will automatically shard archive across GPUs" >&2
    
    exec python main_sparsity_aware.py \
      --archive_backend "${ARCHIVE_BACKEND}" \
      "${MAIN_ARGS[@]}" \
      2>&1 | tee "$OUTPUT_DIR/run.log"
  else
    # Multi-process with PyTorch DDP
    if ! command -v torchrun >/dev/null 2>&1; then
      echo "[ERROR] torchrun is required for multi-GPU execution but was not found in PATH." >&2
      exit 1
    fi

    echo "[INFO] Launching torchrun with ${GPUS_PER_NODE} processes on a single node" >&2
    echo "[INFO] PyTorch DDP will parallelize evaluation across GPUs" >&2
    
    export JAX_PLATFORM_NAME=cpu
    exec torchrun \
      --standalone \
      --nproc_per_node="${GPUS_PER_NODE}" \
      main_sparsity_aware.py \
      --distributed \
      --archive_backend "${ARCHIVE_BACKEND}" \
      "${MAIN_ARGS[@]}" \
      2>&1 | tee "$OUTPUT_DIR/run.log"
  fi
else
  # Single-process, single-GPU
  echo "[INFO] Launching single-process python run on 1 GPU" >&2
  exec python main_sparsity_aware.py \
    --archive_backend "${ARCHIVE_BACKEND}" \
    "${MAIN_ARGS[@]}" \
    2>&1 | tee "$OUTPUT_DIR/run.log"
fi
