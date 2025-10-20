#!/bin/bash
# BFCL完整实验
# 任务12: 全量实验 - 运行完整的多任务进化实验（3000步），对比单任务vs多任务效果

echo "========================================"
echo "🎯 BFCL Full Multi-Task Evolution Experiment"
echo "========================================"

# 设置代理
export https_proxy=https://yefei.p:AolVNTblt0qW28ZwuxuiwM0zKdisK5oSwW1sa3N8UHb1DclJo6kl67yzxe5g@volc-proxy.pjlab.org.cn:13128
export http_proxy=https://yefei.p:AolVNTblt0qW28ZwuxuiwM0zKdisK5oSwW1sa3N8UHb1DclJo6kl67yzxe5g@volc-proxy.pjlab.org.cn:13128

# 设置环境变量
export JAX_PLATFORM_NAME=cpu
export CUDA_VISIBLE_DEVICES=0

# 实验参数
POP_SIZE=5
TOTAL_PASSES=3000
EVAL_SUBSET=30  # 加速评估
OMEGA=0.7
BETA=0.3
PRUNING_SPARSITY=0.3  # 30%稀疏度

echo ""
echo "=== 实验A: 多任务 (GSM8K + BFCL) ==="
echo "  Pop size: $POP_SIZE"
echo "  Total passes: $TOTAL_PASSES"
echo "  Eval subset: $EVAL_SUBSET per task"
echo "  Omega: $OMEGA"
echo "  Beta: $BETA"
echo "  Pruning sparsity: $PRUNING_SPARSITY"
echo ""

OUTPUT_DIR_MULTI="results_bfcl_multitask_pop${POP_SIZE}_prune${PRUNING_SPARSITY}"

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
    --output_dir $OUTPUT_DIR_MULTI \
    --log_sparsity_stats

MULTI_EXIT=$?

echo ""
echo "=== 实验B: 单任务 (仅GSM8K Baseline) ==="
echo ""

OUTPUT_DIR_SINGLE="results_gsm8k_baseline_pop${POP_SIZE}_prune${PRUNING_SPARSITY}"

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
    --output_dir $OUTPUT_DIR_SINGLE \
    --log_sparsity_stats

SINGLE_EXIT=$?

echo ""
echo "========================================"
echo "📊 实验完成 - 结果对比"
echo "========================================"

if [ $MULTI_EXIT -eq 0 ] && [ $SINGLE_EXIT -eq 0 ]; then
    echo "✅ 两个实验都成功完成!"
    echo ""
    echo "多任务结果: $OUTPUT_DIR_MULTI/"
    echo "单任务结果: $OUTPUT_DIR_SINGLE/"
    echo ""
    echo "下一步 - 绘制对比图:"
    echo "  python plot_comparison_curves.py \\"
    echo "    --exp1 $OUTPUT_DIR_MULTI/*.pkl \\"
    echo "    --exp2 $OUTPUT_DIR_SINGLE/*.pkl \\"
    echo "    --labels 'Multi-Task (GSM8K+BFCL)' 'Single-Task (GSM8K)'"
else
    echo "❌ 有实验失败"
    echo "  多任务: exit $MULTI_EXIT"
    echo "  单任务: exit $SINGLE_EXIT"
    exit 1
fi

