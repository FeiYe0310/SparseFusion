#!/bin/bash
# BFCL实验 - 动态稀疏度调度（Cosine Annealing with Warm Restarts）
# 配置说明：使用8个GPU进行BFCL测试，结合动态稀疏度调度

echo "=========================================="
echo "🔄 BFCL + Dynamic Sparsity Experiment"
echo "=========================================="

# ============================================
# GPU 配置
# ============================================
export GPUS_PER_NODE=8
export USE_SINGLE_PROCESS_SHARDING=0

# ============================================
# 存储后端配置
# ============================================
export ARCHIVE_BACKEND=gpu

# ============================================
# 遗传算法参数
# ============================================
export POP_SIZE=8                    # 种群大小
export TOTAL_FORWARD_PASSES=3000     # 总前向传播次数
export RUNS=1                        # 运行次数

# ============================================
# 权重参数
# ============================================
export OMEGA=0.9                     # Fitness权重
export BETA=0.1                      # Sparsity权重

# ============================================
# 🔄 动态稀疏度调度配置（NEW！）
# ============================================
export USE_DYNAMIC_SPARSITY=true     # 启用动态稀疏度
export SPARSITY_MIN=0.1              # 最小稀疏度
export SPARSITY_MAX=0.6              # 最大稀疏度
export SPARSITY_T0=100               # 第一个周期的迭代次数
export SPARSITY_T_MULT=2             # 周期长度乘数（2=每次翻倍，1=固定周期）

# 注意：当启用动态稀疏度时，PRUNING_SPARSITY将被忽略
# export PRUNING_SPARSITY=0.2        # 静态剪枝稀疏度（动态模式下不使用）

# ============================================
# 评估配置
# ============================================
export EVAL_SUBSET_SIZE=20           # 评估子集大小
export USE_BFCL_EVAL=true            # 启用BFCL评估
export BFCL_DATA_PATH=bfcl/data/bfcl_test_200.json  # BFCL数据路径

# ============================================
# 评估权重
# ============================================
export GSM8K_WEIGHT=0.5              # GSM8K任务权重
export BFCL_WEIGHT=0.5               # BFCL任务权重

# ============================================
# 模型路径
# ============================================
export MODEL1_PATH=models/Qwen2.5-0.5B-Instruct
export MODEL2_PATH=models/Qwen2.5-0.5B-Instruct

# ============================================
# 输出目录
# ============================================
export OUTPUT_DIR=results_bfcl_dynamic_sparsity_8gpu

# ============================================
# 实验配置总结
# ============================================
echo "配置总结:"
echo "  GPU数量: ${GPUS_PER_NODE}"
echo "  种群大小: ${POP_SIZE}"
echo "  总迭代次数: ${TOTAL_FORWARD_PASSES}"
echo "  权重: ω=${OMEGA}, β=${BETA}"
echo ""
echo "🔄 动态稀疏度配置:"
echo "  稀疏度范围: [${SPARSITY_MIN}, ${SPARSITY_MAX}]"
echo "  第一周期: ${SPARSITY_T0} 迭代"
echo "  周期乘数: ${SPARSITY_T_MULT}x"
echo ""
echo "预计周期安排（前3000次迭代）:"
echo "  周期1: 迭代 0-99    (100次)  - 稀疏度从 ${SPARSITY_MIN} 增长到 ${SPARSITY_MAX}"
echo "  周期2: 迭代 100-299  (200次)  - 稀疏度从 ${SPARSITY_MIN} 增长到 ${SPARSITY_MAX} [重启!]"
echo "  周期3: 迭代 300-699  (400次)  - 稀疏度从 ${SPARSITY_MIN} 增长到 ${SPARSITY_MAX} [重启!]"
echo "  周期4: 迭代 700-1499 (800次)  - 稀疏度从 ${SPARSITY_MIN} 增长到 ${SPARSITY_MAX} [重启!]"
echo "  周期5: 迭代 1500-3000+(1600次) - 稀疏度从 ${SPARSITY_MIN} 增长到 ${SPARSITY_MAX} [重启!]"
echo ""
echo "输出目录: ${OUTPUT_DIR}"
echo "=========================================="

# ============================================
# 执行脚本
# ============================================
bash scripts/experiments/run_bfcl_single_node.sh

echo "=========================================="
echo "✅ 实验完成！"
echo "结果保存在: ${OUTPUT_DIR}"
echo "=========================================="

