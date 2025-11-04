#!/usr/bin/env bash
set -euo pipefail

# 🎯 BFCL Multi-Task Experiment Launcher
# Single-node execution for GSM8K + BFCL multi-task evaluation
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
# NCCL safety defaults (for multi-GPU if needed)
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
# BFCL Experiment Configuration
# ============================================================================
POP_SIZE=5
TOTAL_PASSES=3000
EVAL_SUBSET=30      # GSM8K 30样本 + BFCL 30样本 = 60个任务
OMEGA=0.7           # Fitness权重
BETA=0.3            # Sparsity权重
PRUNING=0.3         # 30%稀疏度
GSM8K_WEIGHT=0.5    # GSM8K任务权重
BFCL_WEIGHT=0.5     # BFCL任务权重

OUTPUT_DIR="results_bfcl_multitask_pop${POP_SIZE}"

# Model paths (modify if needed)
MODEL1_PATH="models/Qwen2.5-0.5B-Instruct"
MODEL2_PATH="models/Qwen2.5-0.5B-Instruct"

# BFCL data path
BFCL_DATA_PATH="bfcl/data/bfcl_test_200.json"

# ============================================================================
# Pre-flight checks
# ============================================================================
echo "========================================"
echo "🎯 BFCL Multi-Task Experiment"
echo "========================================"
echo ""
echo "📊 Configuration:"
echo "  - Population size: $POP_SIZE"
echo "  - Total passes: $TOTAL_PASSES"
echo "  - Eval subset: $EVAL_SUBSET per task (GSM8K + BFCL)"
echo "  - Omega (fitness): $OMEGA"
echo "  - Beta (sparsity): $BETA"
echo "  - Pruning sparsity: $PRUNING"
echo "  - GSM8K weight: $GSM8K_WEIGHT"
echo "  - BFCL weight: $BFCL_WEIGHT"
echo "  - Output: $OUTPUT_DIR"
echo "  - GPUs per node: $GPUS_PER_NODE"
echo "  - Archive backend: $ARCHIVE_BACKEND"
echo ""

# Check BFCL data
if [[ ! -f "$BFCL_DATA_PATH" ]]; then
  echo "❌ BFCL data not found: $BFCL_DATA_PATH" >&2
  echo "Please run: python tools/convert_bfcl_data.py" >&2
  exit 1
fi
echo "✅ BFCL data ready: $BFCL_DATA_PATH"

# Check models
if [[ ! -d "$MODEL1_PATH" ]]; then
  echo "⚠️  Model1 not found: $MODEL1_PATH" >&2
  echo "Continuing anyway - will attempt to download from HuggingFace" >&2
fi

echo ""
echo "🚀 Starting BFCL experiment..."
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
# Main execution logic (same as run_sparsity_single_node.sh)
# ============================================================================
if (( GPUS_PER_NODE > 1 )); then
  if [[ "$USE_SINGLE_PROCESS_SHARDING" != "0" ]]; then
    # ===========================================================================
    # Mode 1: Single-process with JAX sharding across multiple GPUs
    # ===========================================================================
    echo "[BFCL] Launching single-process run with ${GPUS_PER_NODE} visible GPUs" >&2
    echo "[BFCL] JAX will automatically shard archive across GPUs" >&2
    if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
      echo "[BFCL] CUDA_VISIBLE_DEVICES not set; all GPUs on the node will be visible." >&2
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
