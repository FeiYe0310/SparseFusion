#!/bin/bash
# ============================================================================
# 集群上更新代码并运行加速版本
# ============================================================================

set -e

echo "========================================"
echo "  🚀 更新代码并运行Mini-Batch加速版本"
echo "========================================"
echo ""

# 1. 更新代码
echo "1️⃣  从GitHub拉取最新代码..."
git pull origin main
echo ""

# 2. 检查环境
echo "2️⃣  检查环境..."
conda activate sparsefusion
echo "   ✓ Conda环境激活"
echo ""

# 3. 运行加速版本
echo "3️⃣  运行Mini-Batch加速实验..."
echo ""
echo "配置："
echo "  - eval_subset_size: 30（每轮随机采样30个数据点）"
echo "  - pop_size: 10"
echo "  - total_forward_passes: 500"
echo "  - 预计速度：~10秒/iteration（原来60秒）"
echo ""

# 单GPU版本
GPUS_PER_NODE=1 ./run_sparsity_single_node.sh \
  --runs 1 \
  --model1_path models/Qwen2.5-0.5B-Instruct \
  --model2_path models/Qwen2.5-Coder-0.5B-Instruct \
  --eval_subset_size 30 \
  --pop_size 10 \
  --total_forward_passes 500 \
  --omega 0.5 \
  --beta 0.5 \
  --log_sparsity_stats

echo ""
echo "✅ 实验完成！"
echo ""
echo "结果保存在："
echo "  - results/checkpoints/  (每50步保存)"
echo "  - results/*.pkl  (最终结果)"
echo ""
echo "画图命令："
echo "  python plot_training_curves.py --checkpoint_dir results/checkpoints"
echo ""


# 集群上更新代码并运行加速版本
# ============================================================================

set -e

echo "========================================"
echo "  🚀 更新代码并运行Mini-Batch加速版本"
echo "========================================"
echo ""

# 1. 更新代码
echo "1️⃣  从GitHub拉取最新代码..."
git pull origin main
echo ""

# 2. 检查环境
echo "2️⃣  检查环境..."
conda activate sparsefusion
echo "   ✓ Conda环境激活"
echo ""

# 3. 运行加速版本
echo "3️⃣  运行Mini-Batch加速实验..."
echo ""
echo "配置："
echo "  - eval_subset_size: 30（每轮随机采样30个数据点）"
echo "  - pop_size: 10"
echo "  - total_forward_passes: 500"
echo "  - 预计速度：~10秒/iteration（原来60秒）"
echo ""

# 单GPU版本
GPUS_PER_NODE=1 ./run_sparsity_single_node.sh \
  --runs 1 \
  --model1_path models/Qwen2.5-0.5B-Instruct \
  --model2_path models/Qwen2.5-Coder-0.5B-Instruct \
  --eval_subset_size 30 \
  --pop_size 10 \
  --total_forward_passes 500 \
  --omega 0.5 \
  --beta 0.5 \
  --log_sparsity_stats

echo ""
echo "✅ 实验完成！"
echo ""
echo "结果保存在："
echo "  - results/checkpoints/  (每50步保存)"
echo "  - results/*.pkl  (最终结果)"
echo ""
echo "画图命令："
echo "  python plot_training_curves.py --checkpoint_dir results/checkpoints"
echo ""



