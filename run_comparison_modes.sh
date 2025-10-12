#!/usr/bin/env bash
#
# 运行三种模式的对比实验
#
# 使用方法：
#   bash run_comparison_modes.sh
#

# 公共配置
MODEL1="models/Qwen2.5-0.5B-Instruct"
MODEL2="models/Qwen2.5-Coder-0.5B-Instruct"
POP_SIZE=4
FORWARD_PASSES=20

echo "================================================"
echo "SparseFusion 对比实验"
echo "================================================"
echo "模型1: $MODEL1"
echo "模型2: $MODEL2"
echo "种群大小: $POP_SIZE"
echo "迭代次数: $FORWARD_PASSES"
echo "================================================"
echo ""

# 模式1：原始Natural Niches（无pruning，无sparsity-aware）
echo ">>> 模式1：原始Natural Niches (omega=1.0, beta=0.0, pruning=0.0)"
torchrun --nproc_per_node=8 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=localhost \
  --master_port=29500 \
  main_sparsity_aware.py \
  --model1_path $MODEL1 \
  --model2_path $MODEL2 \
  --pop_size $POP_SIZE \
  --total_forward_passes $FORWARD_PASSES \
  --omega 1.0 \
  --beta 0.0 \
  --pruning_sparsity 0.0 \
  --distributed \
  --archive_backend gpu \
  --output_dir results/comparison_mode1

echo ""
echo "================================================"
echo ""

# 模式2：只有Sparsity-Aware（无pruning）
echo ">>> 模式2：Sparsity-Aware Only (omega=0.5, beta=0.5, pruning=0.0)"
torchrun --nproc_per_node=8 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=localhost \
  --master_port=29500 \
  main_sparsity_aware.py \
  --model1_path $MODEL1 \
  --model2_path $MODEL2 \
  --pop_size $POP_SIZE \
  --total_forward_passes $FORWARD_PASSES \
  --omega 0.5 \
  --beta 0.5 \
  --pruning_sparsity 0.0 \
  --distributed \
  --archive_backend gpu \
  --log_sparsity_stats \
  --output_dir results/comparison_mode2

echo ""
echo "================================================"
echo ""

# 模式3：完整功能（Pruning + Sparsity-Aware）
echo ">>> 模式3：Full Method (omega=0.5, beta=0.5, pruning=0.2)"
torchrun --nproc_per_node=8 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=localhost \
  --master_port=29500 \
  main_sparsity_aware.py \
  --model1_path $MODEL1 \
  --model2_path $MODEL2 \
  --pop_size $POP_SIZE \
  --total_forward_passes $FORWARD_PASSES \
  --omega 0.5 \
  --beta 0.5 \
  --pruning_sparsity 0.2 \
  --pruning_method wanda \
  --distributed \
  --archive_backend gpu \
  --log_sparsity_stats \
  --output_dir results/comparison_mode3

echo ""
echo "================================================"
echo "✅ 所有实验完成！"
echo ""
echo "结果保存在："
echo "  - results/comparison_mode1/ (原始Natural Niches)"
echo "  - results/comparison_mode2/ (Sparsity-Aware Only)"
echo "  - results/comparison_mode3/ (Full Method)"
echo ""
echo "分析结果："
echo "  python analyze_results.py results/comparison_mode1/*.pkl"
echo "  python analyze_results.py results/comparison_mode2/*.pkl"
echo "  python analyze_results.py results/comparison_mode3/*.pkl"
echo "================================================"

