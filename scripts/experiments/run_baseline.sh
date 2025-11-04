#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Baseline Run: 原始进化算法（无sparsity-aware，无剪枝）
# 
# 用途：作为对照组，测试原始Natural Niches算法的性能
# 特点：
#   - omega=1.0, beta=0.0 (只考虑fitness)
#   - pruning_sparsity=0.0 (不剪枝)
#   - 其他参数与sparsity-aware版本保持一致
# ============================================================================

# Offline / cache configuration
unset HF_ENDPOINT
unset HF_HUB_BASE_URL
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
umask 002

# NCCL settings
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

# GPU configuration
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
ARCHIVE_BACKEND="${ARCHIVE_BACKEND:-gpu}"

# ============================================================================
# Baseline Parameters (无sparsity-aware)
# 默认值，可通过命令行参数覆盖
# ============================================================================
MODEL1="${MODEL1:-models/Qwen2.5-0.5B-Instruct}"
MODEL2="${MODEL2:-models/Qwen2.5-Coder-0.5B-Instruct}"
POP_SIZE="${POP_SIZE:-10}"
TOTAL_FORWARD_PASSES="${TOTAL_FORWARD_PASSES:-3000}"
OUTPUT_DIR="${OUTPUT_DIR:-results/baseline}"

# 关键：baseline设置（固定值，确保baseline一致性）
OMEGA=1.0           # 100% fitness权重
BETA=0.0            # 0% sparsity权重 (不考虑稀疏性)
PRUNING_SPARSITY=0.0  # 不剪枝

# 捕获所有命令行参数
MAIN_ARGS=("$@")

# ============================================================================
# Launch
# ============================================================================
echo "=========================================="
echo "  Running BASELINE (No Sparsity-Aware)"
echo "=========================================="
echo "Models: $MODEL1 + $MODEL2"
echo "Archive Size: $POP_SIZE"
echo "Forward Passes: $TOTAL_FORWARD_PASSES"
echo "Omega (fitness): $OMEGA"
echo "Beta (sparsity): $BETA"
echo "Pruning: $PRUNING_SPARSITY (disabled)"
echo "GPUs: $GPUS_PER_NODE"
echo "=========================================="
echo ""

if (( GPUS_PER_NODE > 1 )); then
    if ! command -v torchrun >/dev/null 2>&1; then
        echo "ERROR: torchrun is required for multi-GPU but not found in PATH." >&2
        exit 1
    fi
    
    echo "[Baseline] Launching torchrun with ${GPUS_PER_NODE} processes"
    exec torchrun \
        --standalone \
        --nproc_per_node="${GPUS_PER_NODE}" \
        main_natural_niches_sparsity_aware_fn.py \
        --model1_path "${MODEL1}" \
        --model2_path "${MODEL2}" \
        --pop_size ${POP_SIZE} \
        --total_forward_passes ${TOTAL_FORWARD_PASSES} \
        --omega ${OMEGA} \
        --beta ${BETA} \
        --pruning_sparsity ${PRUNING_SPARSITY} \
        --output_dir "${OUTPUT_DIR}" \
        --archive_backend "${ARCHIVE_BACKEND}" \
        --distributed \
        "${MAIN_ARGS[@]}"
else
    echo "[Baseline] Launching single-process python run"
    exec python main_natural_niches_sparsity_aware_fn.py \
        --model1_path "${MODEL1}" \
        --model2_path "${MODEL2}" \
        --pop_size ${POP_SIZE} \
        --total_forward_passes ${TOTAL_FORWARD_PASSES} \
        --omega ${OMEGA} \
        --beta ${BETA} \
        --pruning_sparsity ${PRUNING_SPARSITY} \
        --output_dir "${OUTPUT_DIR}" \
        --archive_backend "${ARCHIVE_BACKEND}" \
        "${MAIN_ARGS[@]}"
fi

