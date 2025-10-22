#!/usr/bin/env bash
#
# 快速性能分析测试 - 只跑10轮iteration
#

set -e

# 设置实验参数
export MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
export POP_SIZE=8
export FITNESS_PARENTS=300
export RUNS=1
export EVAL_SUBSET=20  # 每个任务20个样本
export NUM_ITERATIONS=10  # 只跑10轮

# 任务配置 - 启用MBPP, Mult4, Boolean
export USE_MBPP_EVAL=true
export USE_MULT4_EVAL=true
export USE_BOOL_EVAL=true

# 任务权重
export GSM8K_WEIGHT=0.5
export MBPP_WEIGHT=0.25
export MULT4_WEIGHT=0.15
export BOOL_WEIGHT=0.10

# Sparsity相关参数
export OMEGA=1.0  # 只看fitness,不看sparsity
export BETA=0.0
export TAU=1.0

# 输出目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="results/profile_test_${TIMESTAMP}"
mkdir -p ${OUTPUT_DIR}

echo "🚀 开始快速性能分析测试"
echo "================================"
echo "模型: ${MODEL_NAME}"
echo "Population: ${POP_SIZE}"
echo "Iterations: ${NUM_ITERATIONS}"
echo "评估样本数/任务: ${EVAL_SUBSET}"
echo "启用任务: GSM8K, MBPP, Mult4, Boolean"
echo "输出目录: ${OUTPUT_DIR}"
echo "================================"

# 运行实验
python3 -u main_sparsity_aware.py \
    --runs ${RUNS} \
    --model1_path ${MODEL_NAME} \
    --model2_path ${MODEL_NAME} \
    --pop_size ${POP_SIZE} \
    --fitness_parents ${FITNESS_PARENTS} \
    --sparsity_aware \
    --omega ${OMEGA} \
    --beta ${BETA} \
    --tau ${TAU} \
    --eval_subset_size ${EVAL_SUBSET} \
    --save_dir ${OUTPUT_DIR} \
    --use_multi_task \
    --gsm8k_weight ${GSM8K_WEIGHT} \
    --use_mbpp_eval \
    --mbpp_weight ${MBPP_WEIGHT} \
    --use_mult4_eval \
    --mult4_weight ${MULT4_WEIGHT} \
    --use_bool_eval \
    --bool_weight ${BOOL_WEIGHT} \
    --num_iterations ${NUM_ITERATIONS} \
    2>&1 | tee ${OUTPUT_DIR}/profile_test.log

echo ""
echo "✅ 测试完成!"
echo "日志文件: ${OUTPUT_DIR}/profile_test.log"
echo ""
echo "请查看日志中的 '📊 任务性能统计' 部分获取详细的耗时和得分信息"

