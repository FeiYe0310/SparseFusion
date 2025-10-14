#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# 快速Baseline实验（内存优化版本）
# 
# 优化策略：
#   1. 减小Archive大小 (pop_size=5)
#   2. 更激进的GPU内存管理
#   3. 减少forward passes用于快速测试
#   4. 使用单GPU避免分布式开销
# ============================================================================

# Offline / cache configuration
unset HF_ENDPOINT
unset HF_HUB_BASE_URL
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
umask 002

# Workspace root
ROOTPATH=${ROOTPATH:-/mnt/shared-storage-user/yefei}
cd "$ROOTPATH/SparseFusion"

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=$ROOTPATH/cache
export WANDB_CACHE_DIR=$ROOTPATH/cache
export TORCH_EXTENSION_DIR=$ROOTPATH/cache
export PATH="$ROOTPATH/miniconda3/bin:$PATH"
export PATH="$ROOTPATH/miniconda3/envs/sparsefusion/bin:$PATH"

# GPU & Memory settings (关键优化！)
export CUDA_VISIBLE_DEVICES=0  # 只用1个GPU
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128  # 减少内存碎片
export CUDA_LAUNCH_BLOCKING=0

# ============================================================================
# 参数配置
# ============================================================================
MODEL1="${MODEL1:-models/Qwen2.5-0.5B-Instruct}"
MODEL2="${MODEL2:-models/Qwen2.5-Coder-0.5B-Instruct}"
POP_SIZE="${POP_SIZE:-5}"  # 减小Archive大小！
TOTAL_FORWARD_PASSES="${TOTAL_FORWARD_PASSES:-200}"  # 快速测试
OUTPUT_DIR="${OUTPUT_DIR:-results/baseline_fast}"

# Baseline设置
OMEGA=1.0
BETA=0.0
PRUNING_SPARSITY=0.0

# ============================================================================
# Launch
# ============================================================================
echo "=========================================="
echo "  快速Baseline实验 (内存优化)"
echo "=========================================="
echo "Models: $MODEL1 + $MODEL2"
echo "Archive Size: $POP_SIZE (优化：减小以节省内存)"
echo "Forward Passes: $TOTAL_FORWARD_PASSES"
echo "GPU: Single GPU (避免分布式开销)"
echo "Omega: $OMEGA, Beta: $BETA"
echo "Pruning: DISABLED"
echo "=========================================="
echo ""

python main_sparsity_aware.py \
    --model1_path "${MODEL1}" \
    --model2_path "${MODEL2}" \
    --pop_size ${POP_SIZE} \
    --total_forward_passes ${TOTAL_FORWARD_PASSES} \
    --omega ${OMEGA} \
    --beta ${BETA} \
    --pruning_sparsity ${PRUNING_SPARSITY} \
    --output_dir "${OUTPUT_DIR}" \
    --archive_backend gpu \
    "$@"

