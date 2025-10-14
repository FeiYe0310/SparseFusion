#!/bin/bash
# ============================================================================
# 真实GSM8K评估版本 - Generation + Exact Match
# 
# 特点：
#   1. 真实生成答案（model.generate()）
#   2. 答案提取 + 精确匹配
#   3. 200样本训练集，50样本测试集
#   4. 每50步保存checkpoint
#   5. 无sparsity，无pruning开销
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
export CUDA_VISIBLE_DEVICES=1  # 使用GPU 1，避免与旧实验冲突
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
echo "  真实GSM8K评估版本"
echo "=========================================="
echo "模型: $MODEL1"
echo "      $MODEL2"
echo "Archive大小: $POP_SIZE"
echo "总步数: $TOTAL_PASSES"
echo "训练集: 200 samples"
echo "测试集: 50 samples"
echo "评估方式: Generation + Exact Match ✅"
echo "GPU: 1 (避免冲突)"
echo "=========================================="
echo ""

# ============================================================================
# 启动实验
# ============================================================================
nohup python -u main.py \
  --model1_path "${MODEL1}" \
  --model2_path "${MODEL2}" \
  --pop_size ${POP_SIZE} \
  --total_forward_passes ${TOTAL_PASSES} \
  --archive_backend gpu \
  --runs 1 \
  --method natural_niches \
  --use_real_gsm8k_eval \
  > baseline_real_gsm8k_$(date +%Y%m%d_%H%M%S).log 2>&1 &

PID=$!

echo "✅ 真实GSM8K评估实验已启动！"
echo ""
echo "进程ID: $PID"
echo "日志文件: baseline_real_gsm8k_*.log"
echo "GPU: 1 (旧实验在GPU 0)"
echo ""
echo "监控命令:"
echo "  tail -f baseline_real_gsm8k_*.log"
echo "  watch -n 10 'ls -lh results/checkpoints/ | tail -10'"
echo ""
echo "预计速度: ~60-90秒/步（generation比token准确率慢5-8倍）"
echo "预计时间:"
echo "  50步: ~60分钟"
echo "  200步: ~4小时"
echo "  500步: ~10小时"
echo ""
echo "=========================================="




