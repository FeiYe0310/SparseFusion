#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# 🎯 BFCL完整实验 - 所有参数完整配置版本
# ============================================================================

echo "════════════════════════════════════════════════════════════════════════"
echo "🎯 BFCL Multi-Task Experiment - Complete Configuration"
echo "════════════════════════════════════════════════════════════════════════"

# ============================================================================
# 1. 代理配置
# ============================================================================
export https_proxy=https://yefei.p:AolVNTblt0qW28ZwuxuiwM0zKdisK5oSwW1sa3N8UHb1DclJo6kl67yzxe5g@volc-proxy.pjlab.org.cn:13128
export http_proxy=https://yefei.p:AolVNTblt0qW28ZwuxuiwM0zKdisK5oSwW1sa3N8UHb1DclJo6kl67yzxe5g@volc-proxy.pjlab.org.cn:13128

echo "✅ 代理已配置"

# ============================================================================
# 2. 环境配置
# ============================================================================
ROOTPATH=${ROOTPATH:-/mnt/shared-storage-user/yefei}
cd "$ROOTPATH/SparseFusion"

# HuggingFace配置
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=$ROOTPATH/cache
export WANDB_CACHE_DIR=$ROOTPATH/cache
export TORCH_EXTENSION_DIR=$ROOTPATH/cache
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0

# Python环境
export PATH="$ROOTPATH/miniconda3/bin:$PATH"
export PATH="$ROOTPATH/miniconda3/envs/sparsefusion/bin:$PATH"

# NCCL配置（多GPU支持）
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple

echo "✅ 环境已配置: $ROOTPATH/SparseFusion"
echo ""

# ============================================================================
# 3. 实验参数配置
# ============================================================================

# === GPU配置 ===
GPUS_PER_NODE=1                    # 使用的GPU数量（1=单GPU，>1=多GPU）
ARCHIVE_BACKEND="gpu"              # 存档后端: "cpu" 或 "gpu"

# === 模型路径 ===
MODEL1_PATH="models/Qwen2.5-0.5B-Instruct"
MODEL2_PATH="models/Qwen2.5-0.5B-Instruct"

# === Natural Niches核心参数 ===
POP_SIZE=5                         # 种群大小（建议2-10）
TOTAL_FORWARD_PASSES=3000          # 总迭代次数（forward passes）
RUNS=1                             # 运行次数（多次运行取平均）

# === Sparsity-Aware参数 ===
OMEGA=0.7                          # Fitness权重（0-1，推荐0.6-0.8）
BETA=0.3                           # Sparsity权重（0-1，推荐0.2-0.4）
PRUNING_SPARSITY=0.3               # 剪枝稀疏度（0-1，0.3=30%稀疏）

# === 评估参数 ===
EVAL_SUBSET_SIZE=30                # 每个任务采样数量（GSM8K 30 + BFCL 30 = 60任务）
BATCH_SIZE=32                      # 评估批次大小

# === BFCL多任务参数 ===
USE_BFCL_EVAL=true                 # 启用BFCL评估
BFCL_DATA_PATH="bfcl/data/bfcl_test_200.json"
GSM8K_WEIGHT=0.5                   # GSM8K任务权重
BFCL_WEIGHT=0.5                    # BFCL任务权重

# === 输出配置 ===
OUTPUT_DIR="results_bfcl_full_pop${POP_SIZE}_omega${OMEGA}_beta${BETA}"
LOG_SPARSITY_STATS=true            # 记录稀疏度统计

# ============================================================================
# 4. 打印配置摘要
# ============================================================================
echo "════════════════════════════════════════════════════════════════════════"
echo "📊 实验配置摘要"
echo "════════════════════════════════════════════════════════════════════════"
echo ""
echo "【模型配置】"
echo "  Model 1: $MODEL1_PATH"
echo "  Model 2: $MODEL2_PATH"
echo ""
echo "【Natural Niches参数】"
echo "  Population size: $POP_SIZE"
echo "  Total iterations: $TOTAL_FORWARD_PASSES"
echo "  Runs: $RUNS"
echo ""
echo "【Sparsity-Aware参数】"
echo "  Omega (fitness weight): $OMEGA"
echo "  Beta (sparsity weight): $BETA"
echo "  Pruning sparsity: $PRUNING_SPARSITY (${PRUNING_SPARSITY}0%稀疏)"
echo ""
echo "【多任务评估】"
echo "  Use BFCL: $USE_BFCL_EVAL"
echo "  Eval subset size: $EVAL_SUBSET_SIZE per task"
echo "  Total tasks per iteration: $((EVAL_SUBSET_SIZE * 2)) (GSM8K + BFCL)"
echo "  GSM8K weight: $GSM8K_WEIGHT"
echo "  BFCL weight: $BFCL_WEIGHT"
echo "  BFCL data: $BFCL_DATA_PATH"
echo ""
echo "【硬件配置】"
echo "  GPUs per node: $GPUS_PER_NODE"
echo "  Archive backend: $ARCHIVE_BACKEND"
echo "  Batch size: $BATCH_SIZE"
echo ""
echo "【输出】"
echo "  Output directory: $OUTPUT_DIR"
echo "  Log sparsity stats: $LOG_SPARSITY_STATS"
echo ""
echo "════════════════════════════════════════════════════════════════════════"

# ============================================================================
# 5. 预检查
# ============================================================================
echo ""
echo "🔍 预检查..."
echo "────────────────────────────────────────────────────────────────────────"

# 检查BFCL数据
if [[ ! -f "$BFCL_DATA_PATH" ]]; then
  echo "❌ BFCL数据不存在: $BFCL_DATA_PATH" >&2
  echo "请运行: python tools/convert_bfcl_data.py" >&2
  exit 1
fi
SAMPLE_COUNT=$(python -c "import json; print(len(json.load(open('$BFCL_DATA_PATH'))))" 2>/dev/null || echo "0")
echo "✅ BFCL数据: $BFCL_DATA_PATH ($SAMPLE_COUNT 样本)"

# 检查模型
if [[ -d "$MODEL1_PATH" ]]; then
  echo "✅ Model 1: $MODEL1_PATH"
else
  echo "⚠️  Model 1不存在，将从HuggingFace下载"
fi

# 检查Python环境
python --version 2>/dev/null && echo "✅ Python可用" || { echo "❌ Python不可用" >&2; exit 1; }

# 检查GPU
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
  echo "✅ 检测到 $GPU_COUNT 个GPU"
else
  echo "⚠️  未检测到nvidia-smi"
fi

echo "────────────────────────────────────────────────────────────────────────"
echo ""

# ============================================================================
# 6. 等待确认（可选，自动运行时注释掉）
# ============================================================================
# read -p "按Enter键开始实验，或Ctrl+C取消... " -r
# echo ""

# ============================================================================
# 7. 运行实验
# ============================================================================
echo "════════════════════════════════════════════════════════════════════════"
echo "🚀 开始BFCL多任务实验"
echo "════════════════════════════════════════════════════════════════════════"
echo ""

# 记录开始时间
START_TIME=$(date +%s)

# 根据GPU数量选择运行模式
if (( GPUS_PER_NODE > 1 )); then
  echo "🔧 多GPU模式 (${GPUS_PER_NODE} GPUs)"
  echo "使用torchrun进行分布式训练"
  echo ""
  
  export JAX_PLATFORM_NAME=cpu
  
  torchrun \
    --standalone \
    --nproc_per_node="${GPUS_PER_NODE}" \
    main_sparsity_aware.py \
    --distributed \
    --archive_backend "${ARCHIVE_BACKEND}" \
    --model1_path "$MODEL1_PATH" \
    --model2_path "$MODEL2_PATH" \
    --pop_size "$POP_SIZE" \
    --total_forward_passes "$TOTAL_FORWARD_PASSES" \
    --runs "$RUNS" \
    --omega "$OMEGA" \
    --beta "$BETA" \
    --pruning_sparsity "$PRUNING_SPARSITY" \
    --eval_subset_size "$EVAL_SUBSET_SIZE" \
    --use_bfcl_eval \
    --bfcl_data_path "$BFCL_DATA_PATH" \
    --gsm8k_weight "$GSM8K_WEIGHT" \
    --bfcl_weight "$BFCL_WEIGHT" \
    --output_dir "$OUTPUT_DIR" \
    --log_sparsity_stats
else
  echo "🔧 单GPU模式"
  echo ""
  
  python main_sparsity_aware.py \
    --archive_backend "${ARCHIVE_BACKEND}" \
    --model1_path "$MODEL1_PATH" \
    --model2_path "$MODEL2_PATH" \
    --pop_size "$POP_SIZE" \
    --total_forward_passes "$TOTAL_FORWARD_PASSES" \
    --runs "$RUNS" \
    --omega "$OMEGA" \
    --beta "$BETA" \
    --pruning_sparsity "$PRUNING_SPARSITY" \
    --eval_subset_size "$EVAL_SUBSET_SIZE" \
    --use_bfcl_eval \
    --bfcl_data_path "$BFCL_DATA_PATH" \
    --gsm8k_weight "$GSM8K_WEIGHT" \
    --bfcl_weight "$BFCL_WEIGHT" \
    --output_dir "$OUTPUT_DIR" \
    --log_sparsity_stats
fi

EXIT_CODE=$?

# ============================================================================
# 8. 结果统计
# ============================================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "════════════════════════════════════════════════════════════════════════"

if [ $EXIT_CODE -eq 0 ]; then
  echo "✅ 实验完成！"
  echo "════════════════════════════════════════════════════════════════════════"
  echo ""
  echo "⏱️  运行时间: ${HOURS}h ${MINUTES}m ${SECONDS}s"
  echo "📁 结果目录: $OUTPUT_DIR/"
  echo ""
  
  # 检查输出文件
  if [[ -d "$OUTPUT_DIR" ]]; then
    echo "📊 输出文件:"
    ls -lh "$OUTPUT_DIR/" | grep -v '^total' | awk '{print "  - " $9 " (" $5 ")"}'
    echo ""
  fi
  
  echo "🔍 查看结果:"
  echo "  ls -lh $OUTPUT_DIR/"
  echo ""
  echo "📈 绘制训练曲线:"
  echo "  python tools/plot_training_curves.py --input $OUTPUT_DIR/*.pkl"
  echo ""
  
else
  echo "❌ 实验失败 (exit code: $EXIT_CODE)"
  echo "════════════════════════════════════════════════════════════════════════"
  echo ""
  echo "⏱️  运行时间: ${HOURS}h ${MINUTES}m ${SECONDS}s"
  echo ""
  echo "🔍 查看日志:"
  echo "  tail -100 $OUTPUT_DIR/*.log"
  echo ""
  exit $EXIT_CODE
fi

echo "════════════════════════════════════════════════════════════════════════"
