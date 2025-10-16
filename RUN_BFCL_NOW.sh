#!/bin/bash
# 🎯 BFCL完整实验 - 一键运行脚本
# 这个脚本会运行完整的GSM8K+BFCL多任务实验

echo "========================================"
echo "🎯 BFCL Multi-Task Experiment"
echo "========================================"

# 1. 设置代理
export https_proxy=https://yefei.p:AolVNTblt0qW28ZwuxuiwM0zKdisK5oSwW1sa3N8UHb1DclJo6kl67yzxe5g@volc-proxy.pjlab.org.cn:13128
export http_proxy=https://yefei.p:AolVNTblt0qW28ZwuxuiwM0zKdisK5oSwW1sa3N8UHb1DclJo6kl67yzxe5g@volc-proxy.pjlab.org.cn:13128

# 2. 设置环境
export JAX_PLATFORM_NAME=cpu
export CUDA_VISIBLE_DEVICES=0

# 3. 实验配置
POP_SIZE=5
TOTAL_PASSES=3000
EVAL_SUBSET=30      # GSM8K采样30个 + BFCL采样30个 = 60个任务
OMEGA=0.7           # Fitness权重
BETA=0.3            # Sparsity权重
PRUNING=0.3         # 30%稀疏度

OUTPUT_DIR="results_bfcl_multitask_pop${POP_SIZE}"

echo ""
echo "📊 实验配置:"
echo "  - Population size: $POP_SIZE"
echo "  - Total passes: $TOTAL_PASSES"
echo "  - Eval subset: $EVAL_SUBSET per task (GSM8K + BFCL)"
echo "  - Omega (fitness): $OMEGA"
echo "  - Beta (sparsity): $BETA"
echo "  - Pruning sparsity: $PRUNING"
echo "  - Output: $OUTPUT_DIR"
echo ""

# 4. 检查数据
if [ ! -f "bfcl/data/bfcl_test_200.json" ]; then
    echo "❌ BFCL数据集不存在"
    echo "请运行: python convert_bfcl_data.py"
    exit 1
fi

echo "✅ BFCL数据集已准备: bfcl/data/bfcl_test_200.json"
echo ""

# 5. 运行实验
echo "🚀 开始运行实验..."
echo "========================================"
echo ""

python main_sparsity_aware.py \
    --model1_path models/Qwen2.5-0.5B-Instruct \
    --model2_path models/Qwen2.5-0.5B-Instruct \
    --pop_size $POP_SIZE \
    --total_forward_passes $TOTAL_PASSES \
    --runs 1 \
    --omega $OMEGA \
    --beta $BETA \
    --pruning_sparsity $PRUNING \
    --eval_subset_size $EVAL_SUBSET \
    --use_bfcl_eval \
    --bfcl_data_path bfcl/data/bfcl_test_200.json \
    --gsm8k_weight 0.5 \
    --bfcl_weight 0.5 \
    --output_dir $OUTPUT_DIR \
    --log_sparsity_stats

EXIT_CODE=$?

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 实验完成!"
    echo "========================================"
    echo ""
    echo "结果保存在: $OUTPUT_DIR/"
    echo ""
    echo "下一步:"
    echo "  1. 查看结果: ls -lh $OUTPUT_DIR/"
    echo "  2. 绘制曲线: python plot_training_curves.py --input $OUTPUT_DIR/*.pkl"
    echo "  3. 分析日志: cat $OUTPUT_DIR/*.json | grep -A5 fitness"
else
    echo "❌ 实验失败 (exit code: $EXIT_CODE)"
    echo "========================================"
    exit $EXIT_CODE
fi

