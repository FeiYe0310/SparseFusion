#!/usr/bin/env bash
set -euo pipefail

# 🧪 BFCL Quick Test - Fast validation run
# Based on run_sparsity_single_node.sh

# ============================================================================
# Offline / cache configuration
# ============================================================================
unset HF_ENDPOINT
unset HF_HUB_BASE_URL
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
umask 002

# ============================================================================
# NCCL safety defaults
# ============================================================================
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

# ============================================================================
# Workspace root and caches
# ============================================================================
ROOTPATH=${ROOTPATH:-/mnt/shared-storage-user/yefei}
cd "$ROOTPATH/SparseFusion"

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=$ROOTPATH/cache
export WANDB_CACHE_DIR=$ROOTPATH/cache
export TORCH_EXTENSION_DIR=$ROOTPATH/cache

export PATH="$ROOTPATH/miniconda3/bin:$PATH"
export PATH="$ROOTPATH/miniconda3/envs/sparsefusion/bin:$PATH"

# ============================================================================
# GPU Configuration
# ============================================================================
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
USE_SINGLE_PROCESS_SHARDING="${USE_SINGLE_PROCESS_SHARDING:-0}"
ARCHIVE_BACKEND="${ARCHIVE_BACKEND:-gpu}"

# ============================================================================
# Quick Test Configuration (smaller scale)
# ============================================================================
POP_SIZE=2
TOTAL_PASSES=100    # Much shorter for quick validation
EVAL_SUBSET=30
OMEGA=0.7
BETA=0.3
PRUNING=0.3
GSM8K_WEIGHT=0.5
BFCL_WEIGHT=0.5

OUTPUT_DIR="results_bfcl_quick_test"

MODEL1_PATH="models/Qwen2.5-0.5B-Instruct"
MODEL2_PATH="models/Qwen2.5-0.5B-Instruct"
BFCL_DATA_PATH="data/bfcl/data/bfcl_test_200.json"

# ============================================================================
# Pre-flight checks
# ============================================================================
echo "========================================"
echo "🧪 BFCL Quick Test (Validation Run)"
echo "========================================"
echo ""
echo "📊 Configuration:"
echo "  - Population size: $POP_SIZE (small)"
echo "  - Total passes: $TOTAL_PASSES (short)"
echo "  - Eval subset: $EVAL_SUBSET per task"
echo "  - Omega: $OMEGA, Beta: $BETA"
echo "  - Pruning: $PRUNING"
echo "  - Output: $OUTPUT_DIR"
echo "  - GPUs: $GPUS_PER_NODE"
echo "  - Expected time: ~1-2 hours"
echo ""

# Check BFCL data
if [[ ! -f "$BFCL_DATA_PATH" ]]; then
  echo "❌ BFCL data not found: $BFCL_DATA_PATH" >&2
  exit 1
fi
echo "✅ BFCL data ready"
echo ""
echo "🚀 Starting quick test..."
echo "========================================"
echo ""

# ============================================================================
# Construct main arguments
# ============================================================================
MAIN_ARGS=(
  --model1_path "$MODEL1_PATH"
  --model2_path "$MODEL2_PATH"
  --pop_size "$POP_SIZE"
  --total_forward_passes "$TOTAL_PASSES"
  --runs 1
  --omega "$OMEGA"
  --beta "$BETA"
  --pruning_sparsity "$PRUNING"
  --eval_subset_size "$EVAL_SUBSET"
  --use_bfcl_eval
  --bfcl_data_path "$BFCL_DATA_PATH"
  --gsm8k_weight "$GSM8K_WEIGHT"
  --bfcl_weight "$BFCL_WEIGHT"
  --output_dir "$OUTPUT_DIR"
  --log_sparsity_stats
)

# ============================================================================
# Main execution logic
# ============================================================================
if (( GPUS_PER_NODE > 1 )); then
  if [[ "$USE_SINGLE_PROCESS_SHARDING" != "0" ]]; then
    echo "[QuickTest] Single-process with ${GPUS_PER_NODE} GPUs (JAX sharding)" >&2
    exec python main_natural_niches_sparsity_aware_fn.py \
      --archive_backend "${ARCHIVE_BACKEND}" \
      "${MAIN_ARGS[@]}"
  else
    echo "[QuickTest] Multi-process with torchrun (${GPUS_PER_NODE} GPUs)" >&2
    export JAX_PLATFORM_NAME=cpu
    exec torchrun \
      --standalone \
      --nproc_per_node="${GPUS_PER_NODE}" \
      main_natural_niches_sparsity_aware_fn.py \
      --distributed \
      --archive_backend "${ARCHIVE_BACKEND}" \
      "${MAIN_ARGS[@]}"
  fi
else
  echo "[QuickTest] Single-GPU mode" >&2
  exec python main_natural_niches_sparsity_aware_fn.py \
    --archive_backend "${ARCHIVE_BACKEND}" \
    "${MAIN_ARGS[@]}"
fi
