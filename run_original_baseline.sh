#!/bin/bash
# ============================================================================
# 原始Natural Niches Baseline实验（无sparsity-aware，更快！）
# 
# 优势：
#   1. 没有sparsity计算开销
#   2. 没有pruning开销
#   3. 代码更简单，更稳定
#   4. 训练集：200样本（快速）
#   5. Checkpoint：每50步保存
# ============================================================================

set -euo pipefail

# 工作目录
ROOTPATH=${ROOTPATH:-/mnt/shared-storage-user/yefei}
cd "$ROOTPATH/SparseFusion"

# 缓存设置
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=$ROOTPATH/cache
export WANDB_CACHE_DIR=$ROOTPATH/cache
export TORCH_EXTENSION_DIR=$ROOTPATH/cache
export PATH="$ROOTPATH/miniconda3/bin:$PATH"
export PATH="$ROOTPATH/miniconda3/envs/sparsefusion/bin:$PATH"

# GPU设置（单GPU，防OOM）
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=0

# ============================================================================
# 实验参数
# ============================================================================
MODEL1="${MODEL1:-models/Qwen2.5-0.5B-Instruct}"
MODEL2="${MODEL2:-models/Qwen2.5-Coder-0.5B-Instruct}"
POP_SIZE="${POP_SIZE:-5}"
TOTAL_PASSES="${TOTAL_PASSES:-500}"

echo "=========================================="
echo "  原始Natural Niches Baseline"
echo "=========================================="
echo "模型: $MODEL1"
echo "      $MODEL2"
echo "Archive大小: $POP_SIZE"
echo "总步数: $TOTAL_PASSES"
echo "训练集: 200 samples (超快！)"
echo "测试集: 50 samples"
echo "Checkpoint: 每50步保存"
echo "优势: 无sparsity开销，比sparsity-aware版本快2-3倍！"
echo "=========================================="
echo ""

# ============================================================================
# 启动实验
# ============================================================================
nohup python main.py \
  --model1_path "${MODEL1}" \
  --model2_path "${MODEL2}" \
  --pop_size ${POP_SIZE} \
  --total_forward_passes ${TOTAL_PASSES} \
  --archive_backend gpu \
  --runs 1 \
  --method natural_niches \
  > baseline_original_$(date +%Y%m%d_%H%M%S).log 2>&1 &

PID=$!

echo "✅ 原始Baseline实验已启动！"
echo ""
echo "进程ID: $PID"
echo "日志文件: baseline_original_*.log"
echo ""
echo "监控命令:"
echo "  tail -f baseline_original_*.log"
echo "  watch -n 10 'ls -lh results/checkpoints/ | tail -5'"
echo ""
echo "预计时间:"
echo "  50步（第一个checkpoint）: ~5-8分钟"
echo "  200步（可画图）: ~25分钟"
echo "  500步（完整）: ~1小时"
echo ""
echo "=========================================="

