#!/bin/bash
# ============================================================================
# Baseline实验（完全匹配sparsity-aware配置）
# Natural Niches without sparsity-aware selection
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL1="models/Qwen2.5-0.5B-Instruct"
MODEL2="models/Qwen2.5-Coder-0.5B-Instruct"

echo "========================================"
echo "📊 Baseline - Natural Niches (无Sparsity-Aware)"
echo "========================================"
echo "Population size: 4"
echo "Total forward passes: 3000"
echo "Runs: 1"
echo "🚀 Eval subset size: 30 (加速模式)"
echo ""
echo "Baseline Parameters:"
echo "  ω (omega): 1.0 - Pure fitness weight"
echo "  β (beta): 0.0 - No sparsity (disabled)"
echo "  τ (tau): 1.0 - Temperature"
echo "  α (alpha): 1.0 - Fitness normalization"
echo "  ε (epsilon): 1e-10 - Zero threshold"
echo ""
echo "Pruning Parameters:"
echo "  Pruning DISABLED (baseline无剪枝)"
echo ""
echo "Experimental Setup:"
echo "  Crossover: Enabled"
echo "  Splitpoint: Enabled"
echo "  Matchmaker: Enabled"
echo "  Archive backend: gpu"
echo "  Distributed: True"
echo "========================================"
echo ""

# 多GPU分布式运行（完全匹配sparsity-aware设置）
GPUS_PER_NODE=8 "$SCRIPT_DIR/run_sparsity_single_node.sh" \
  --runs 1 \
  --model1_path ${MODEL1} \
  --model2_path ${MODEL2} \
  --pop_size 4 \
  --total_forward_passes 3000 \
  --omega 1.0 \
  --beta 0.0 \
  --tau 1.0 \
  --alpha 1.0 \
  --epsilon 1e-10 \
  --pruning_sparsity 0.0 \
  --eval_subset_size 30 \
  --output_dir results/baseline

echo ""
echo "✅ Baseline实验完成！"
echo ""
echo "对比："
echo "  Baseline:     omega=1.0, beta=0.0 (纯fitness)"
echo "  Your method:  omega=0.5, beta=0.5 (fitness+sparsity)"
echo ""
echo "结果对比查看："
echo "  python tools/plot_training_curves.py --checkpoint_dir results/"
echo ""