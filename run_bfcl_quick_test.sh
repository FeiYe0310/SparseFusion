#!/bin/bash
# BFCL快速验证实验
# 任务10: 快速验证实验 - 运行小规模实验（pop_size=2, 100步）测试BFCL评估功能

echo "========================================"
echo "🚀 BFCL Quick Test Experiment"
echo "========================================"

# 设置代理
export https_proxy=https://yefei.p:AolVNTblt0qW28ZwuxuiwM0zKdisK5oSwW1sa3N8UHb1DclJo6kl67yzxe5g@volc-proxy.pjlab.org.cn:13128
export http_proxy=https://yefei.p:AolVNTblt0qW28ZwuxuiwM0zKdisK5oSwW1sa3N8UHb1DclJo6kl67yzxe5g@volc-proxy.pjlab.org.cn:13128

# 设置环境变量
export JAX_PLATFORM_NAME=cpu
export CUDA_VISIBLE_DEVICES=0

# 实验参数
POP_SIZE=2
TOTAL_PASSES=100
EVAL_SUBSET=30  # GSM8K 30个 + BFCL 30个
OMEGA=0.7
BETA=0.3
PRUNING_SPARSITY=0.0  # 快速测试不启用剪枝

OUTPUT_DIR="results_bfcl_quick_test"

echo ""
echo "实验配置:"
echo "  Pop size: $POP_SIZE"
echo "  Total forward passes: $TOTAL_PASSES"
echo "  Eval subset size: $EVAL_SUBSET (per task)"
echo "  Omega (fitness): $OMEGA"
echo "  Beta (sparsity): $BETA"
echo "  Pruning: disabled"
echo "  Output dir: $OUTPUT_DIR"
echo ""

# 运行实验
python main_sparsity_aware.py \
    --model1_path models/Qwen2.5-0.5B-Instruct \
    --model2_path models/Qwen2.5-0.5B-Instruct \
    --pop_size $POP_SIZE \
    --total_forward_passes $TOTAL_PASSES \
    --runs 1 \
    --omega $OMEGA \
    --beta $BETA \
    --pruning_sparsity $PRUNING_SPARSITY \
    --eval_subset_size $EVAL_SUBSET \
    --use_bfcl_eval \
    --bfcl_data_path bfcl/data/bfcl_test_simple.json \
    --gsm8k_weight 0.5 \
    --bfcl_weight 0.5 \
    --output_dir $OUTPUT_DIR

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✅ 快速测试实验完成!"
    echo "========================================"
    echo ""
    echo "结果保存在: $OUTPUT_DIR/"
    echo ""
    echo "下一步:"
    echo "  1. 查看结果: ls -lh $OUTPUT_DIR/"
    echo "  2. 绘制曲线: python plot_training_curves.py --input $OUTPUT_DIR/*.pkl"
    echo "  3. 运行完整实验: bash run_bfcl_full_exp.sh"
else
    echo ""
    echo "❌ 实验失败 (exit code: $EXIT_CODE)"
    exit $EXIT_CODE
fi

