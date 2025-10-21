#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# 🧪 MBPP Quick Test - 快速验证MBPP集成
# ============================================================================

echo "========================================"
echo "🧪 MBPP Quick Test - Fast Validation"
echo "========================================"

# === 模型配置 ===
MODEL1_PATH="${MODEL1_PATH:-models/Qwen2.5-0.5B-Instruct}"
MODEL2_PATH="${MODEL2_PATH:-models/Qwen2.5-0.5B-Instruct}"

# === 实验参数 ===
POP_SIZE=4
TOTAL_FORWARD_PASSES=50      # 快速测试：50步
RUNS=1
EVAL_SUBSET_SIZE=5          # 每个任务只评估5个样本

# === 稀疏度参数 ===
OMEGA=0.7
BETA=0.3
PRUNING_SPARSITY=0.2
PRUNING_METHOD="wanda"

# === 多任务权重 ===
GSM8K_WEIGHT=0.4
BFCL_WEIGHT=0.3
MBPP_WEIGHT=0.3

# === 数据路径 ===
BFCL_DATA_PATH="${BFCL_DATA_PATH:-bfcl/data/bfcl_test_200.json}"
MBPP_DATA_PATH="${MBPP_DATA_PATH:-mbpp/data/mbpp_test_sample.json}"

# === 输出目录 ===
OUTPUT_DIR="${OUTPUT_DIR:-results/mbpp_quick_test}"

echo ""
echo "配置参数："
echo "  模型: $MODEL1_PATH"
echo "  种群大小: $POP_SIZE"
echo "  迭代次数: $TOTAL_FORWARD_PASSES"
echo "  评估子集: $EVAL_SUBSET_SIZE 样本/任务"
echo ""
echo "多任务权重："
echo "  GSM8K: $GSM8K_WEIGHT"
echo "  BFCL: $BFCL_WEIGHT"
echo "  MBPP: $MBPP_WEIGHT"
echo ""

# 检查数据文件
if [[ ! -f "$BFCL_DATA_PATH" ]]; then
  echo "❌ BFCL数据不存在: $BFCL_DATA_PATH" >&2
  exit 1
fi

if [[ ! -f "$MBPP_DATA_PATH" ]]; then
  echo "❌ MBPP数据不存在: $MBPP_DATA_PATH" >&2
  echo "提示: 使用示例数据 mbpp/data/mbpp_test_sample.json" >&2
  exit 1
fi

echo "✅ 数据检查通过"
echo ""
echo "🚀 开始MBPP三任务快速测试..."
echo ""

# 运行实验
python3 main_sparsity_aware.py \
  --runs $RUNS \
  --model1_path "$MODEL1_PATH" \
  --model2_path "$MODEL2_PATH" \
  --pop_size $POP_SIZE \
  --total_forward_passes $TOTAL_FORWARD_PASSES \
  --omega $OMEGA \
  --beta $BETA \
  --pruning_sparsity $PRUNING_SPARSITY \
  --pruning_method $PRUNING_METHOD \
  --eval_subset_size $EVAL_SUBSET_SIZE \
  --use_bfcl_eval \
  --bfcl_data_path "$BFCL_DATA_PATH" \
  --gsm8k_weight $GSM8K_WEIGHT \
  --bfcl_weight $BFCL_WEIGHT \
  --use_mbpp_eval \
  --mbpp_data_path "$MBPP_DATA_PATH" \
  --mbpp_weight $MBPP_WEIGHT \
  --output_dir "$OUTPUT_DIR" \
  --log_sparsity_stats

echo ""
echo "========================================"
echo "✅ MBPP快速测试完成！"
echo "========================================"
echo ""
echo "结果保存在: $OUTPUT_DIR"
echo ""
echo "下一步："
echo "  1. 查看结果: python tools/analyze_results.py $OUTPUT_DIR/*.pkl --no-plot"
echo "  2. 完整实验: bash scripts/experiments/run_mbpp_full_exp.sh"
echo ""

