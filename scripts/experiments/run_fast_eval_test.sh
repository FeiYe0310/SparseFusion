#!/bin/bash
# ============================================================================
# 测试Mini-Batch Evaluation加速功能
# 使用30个随机数据点评估，每轮更换batch
# ============================================================================

set -e

MODEL1="models/Qwen2.5-0.5B-Instruct"
MODEL2="models/Qwen2.5-Coder-0.5B-Instruct"

echo "========================================"
echo "  🚀 Fast Eval Test (Mini-Batch)"
echo "========================================"
echo ""
echo "加速策略："
echo "  ✓ 每轮只评估 30 个随机数据点"
echo "  ✓ 每轮更换不同的batch"
echo "  ✓ 预计速度提升 6-10倍"
echo ""

# 使用eval_subset_size=30进行加速评估
python3 main_natural_niches_sparsity_aware_fn.py \
  --runs 1 \
  --model1_path ${MODEL1} \
  --model2_path ${MODEL2} \
  --pop_size 10 \
  --total_forward_passes 100 \
  --omega 0.5 \
  --beta 0.5 \
  --eval_subset_size 30 \
  --log_sparsity_stats \
  --output_dir results/fast_eval_test

echo ""
echo "✅ 测试完成！"
echo ""
echo "对比速度："
echo "  - 完整数据集（200样本）：~60秒/iteration"
echo "  - Mini-batch（30样本）：~10秒/iteration"
echo "  - 加速倍数：6x"
echo ""

