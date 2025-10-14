#!/usr/bin/env bash
# 这个脚本展示如何用不同的模型运行实验

# 设置工作路径
ROOTPATH=${ROOTPATH:-/mnt/shared-storage-user/yefei}
cd "$ROOTPATH/SparseFusion"

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=$ROOTPATH/cache
export PATH="$ROOTPATH/miniconda3/envs/sparsefusion/bin:$PATH"

# ======== 配置区域 ========
# 选择您想使用的模型组合（取消注释您想要的组合）

# 方案1: 0.5B快速实验（推荐用于调试）
MODEL1="models/Qwen2.5-0.5B-Instruct"
MODEL2="models/Qwen2.5-Coder-0.5B-Instruct"

# 方案2: 1.5B专业模型（更好的性能）
# MODEL1="models/Qwen2.5-Math-1.5B-Instruct"
# MODEL2="models/Qwen2.5-Coder-1.5B-Instruct"

# 方案3: 数学专用组合
# MODEL1="models/Qwen2.5-Math-1.5B-Instruct"
# MODEL2="models/wizardmath_7b"

# 方案4: 混合规模
# MODEL1="models/Qwen2.5-0.5B-Instruct"
# MODEL2="models/Qwen2.5-Math-1.5B-Instruct"

# 其他参数
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
POP_SIZE=${POP_SIZE:-16}
TOTAL_FP=${TOTAL_FP:-100}

# ==========================

echo "=================================="
echo "🚀 启动模型进化实验"
echo "=================================="
echo "模型1: $MODEL1"
echo "模型2: $MODEL2"
echo "GPU数量: $GPUS_PER_NODE"
echo "种群大小: $POP_SIZE"
echo "总前向传递次数: $TOTAL_FP"
echo "=================================="

if (( GPUS_PER_NODE > 1 )); then
  echo "使用 torchrun 多GPU模式"
  exec torchrun \
    --standalone \
    --nproc_per_node="${GPUS_PER_NODE}" \
    run_evolution.py \
    --model1_path "${MODEL1}" \
    --model2_path "${MODEL2}" \
    --pop_size "${POP_SIZE}" \
    --total_forward_passes "${TOTAL_FP}" \
    --use_sharded_archive \
    --archive_backend gpu \
    "$@"
else
  echo "使用单GPU模式"
  exec python run_evolution.py \
    --model1_path "${MODEL1}" \
    --model2_path "${MODEL2}" \
    --pop_size "${POP_SIZE}" \
    --total_forward_passes "${TOTAL_FP}" \
    --use_sharded_archive \
    --archive_backend gpu \
    "$@"
fi







