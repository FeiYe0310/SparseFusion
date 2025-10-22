#!/usr/bin/env bash
#
# 小批次运行测试 - pipsize=12
#

set -e

# 设置实验参数
export MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
export POP_SIZE=8
export FITNESS_PARENTS=300
export RUNS=1
export EVAL_SUBSET=12  # 每个任务12个样本
export NUM_ITERATIONS=30  # 30轮iteration
export TOTAL_FORWARD_PASSES=$((POP_SIZE * NUM_ITERATIONS))  # 8 * 30 = 240

# PIPSize参数
export PIPSIZE=12

# 任务配置 - 全任务启用
export USE_BFCL_EVAL=true
export USE_MBPP_EVAL=true
export USE_MULT4_EVAL=true
export USE_MULT5_EVAL=true
export USE_BOOL_EVAL=true

# 任务权重
export GSM8K_WEIGHT=0.50
export BFCL_WEIGHT=0.00
export MBPP_WEIGHT=0.50
export MULT4_WEIGHT=0.25
export MULT5_WEIGHT=0.00
export BOOL_WEIGHT=0.25

# Sparsity相关参数 (baseline模式)
export OMEGA=1.0  # 只看fitness,不看sparsity
export BETA=0.0
export TAU=1.0
export PRUNING_SPARSITY=0.0  # 不启用pruning

# 输出目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="results/small_batch_pipsize12_${TIMESTAMP}"
mkdir -p ${OUTPUT_DIR}

echo "🚀 开始小批次运行测试 (PIPSize=12)"
echo "================================================"
echo "模型: ${MODEL_NAME}"
echo "Population: ${POP_SIZE}"
echo "Iterations: ${NUM_ITERATIONS}"
echo "PIPSize: ${PIPSIZE}"
echo "评估样本数/任务: ${EVAL_SUBSET}"
echo "启用任务: GSM8K, BFCL, MBPP, Mult4, Mult5, Boolean"
echo "输出目录: ${OUTPUT_DIR}"
echo "================================================"

# 运行实验
python3 -u main_sparsity_aware.py \
    --runs ${RUNS} \
    --model1_path ${MODEL_NAME} \
    --model2_path ${MODEL_NAME} \
    --pop_size ${POP_SIZE} \
    --total_forward_passes ${TOTAL_FORWARD_PASSES} \
    --pipsize ${PIPSIZE} \
    --omega ${OMEGA} \
    --beta ${BETA} \
    --tau ${TAU} \
    --pruning_sparsity ${PRUNING_SPARSITY} \
    --eval_subset_size ${EVAL_SUBSET} \
    --output_dir ${OUTPUT_DIR} \
    --use_multi_task \
    --gsm8k_weight ${GSM8K_WEIGHT} \
    --use_bfcl_eval \
    --bfcl_weight ${BFCL_WEIGHT} \
    --use_mbpp_eval \
    --mbpp_weight ${MBPP_WEIGHT} \
    --use_mult4_eval \
    --mult4_weight ${MULT4_WEIGHT} \
    --use_mult5_eval \
    --mult5_weight ${MULT5_WEIGHT} \
    --use_bool_eval \
    --bool_weight ${BOOL_WEIGHT} \
    2>&1 | tee ${OUTPUT_DIR}/run.log

echo ""
echo "✅ 运行完成!"
echo "日志文件: ${OUTPUT_DIR}/run.log"
echo "结果文件: ${OUTPUT_DIR}/*.pkl"
echo ""
echo "📊 查看任务性能统计，请检查日志中的 '任务性能统计' 部分"

