# 📊 PIPSize=12 小批次运行指南

## 概述

`small_batch_pipsize12.sh` 脚本用于快速测试，使用较小的批次大小（pipsize=12）运行多任务进化实验。

## 🚀 快速开始

### 默认配置运行

```bash
cd /mnt/shared-storage-user/yefei/SparseFusion
bash scripts/experiments/small_batch_pipsize12.sh
```

### 自定义参数运行

```bash
# 修改GPUs数量
GPUS_PER_NODE=4 bash scripts/experiments/small_batch_pipsize12.sh

# 修改迭代次数
TOTAL_FORWARD_PASSES=1200 bash scripts/experiments/small_batch_pipsize12.sh

# 修改pipsize
PIPSIZE=20 bash scripts/experiments/small_batch_pipsize12.sh
```

## 📋 默认配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `GPUS_PER_NODE` | 8 | 使用的GPU数量 |
| `POP_SIZE` | 8 | 种群大小 |
| `TOTAL_FORWARD_PASSES` | 600 | 总评估次数 (约50轮) |
| `PIPSIZE` | 12 | 批次大小 |
| `EVAL_SUBSET_SIZE` | 12 | 每个任务的评估样本数 |
| `OMEGA` | 1.0 | Fitness权重 (baseline模式) |
| `BETA` | 0.0 | Sparsity权重 |
| `PRUNING_SPARSITY` | 0.0 | 不启用剪枝 |

## 🎯 任务配置

默认启用的任务和权重：

- **GSM8K**: 0.50 (数学推理)
- **BFCL**: 0.00 (函数调用，已启用但权重为0)
- **MBPP**: 0.50 (代码生成)
- **Mult4**: 0.25 (4位乘法)
- **Mult5**: 0.00 (5位乘法，已启用但权重为0)
- **Boolean**: 0.25 (布尔逻辑)

## 🔧 自定义任务权重

```bash
# 只测试GSM8K和MBPP
GSM8K_WEIGHT=0.7 \
MBPP_WEIGHT=0.3 \
MULT4_WEIGHT=0.0 \
BOOL_WEIGHT=0.0 \
bash scripts/experiments/small_batch_pipsize12.sh

# 启用BFCL和Mult5
BFCL_WEIGHT=0.3 \
MULT5_WEIGHT=0.2 \
bash scripts/experiments/small_batch_pipsize12.sh
```

## 📊 性能分析

运行后会自动显示每个任务的性能统计：

```
======================================================================
📊 任务性能统计
======================================================================
任务                 耗时(s)         平均得分          得分/秒
----------------------------------------------------------------------
GSM8K                  45.23          0.6500        0.014371
MBPP                   38.67          0.5200        0.013447
Mult4                  12.34          0.8100        0.065640
Boolean                 8.91          0.7300        0.081930
======================================================================
```

## 📁 输出文件

结果保存在 `results/small_batch_pipsize12_YYYYMMDD_HHMMSS/`:

- `run.log` - 完整运行日志
- `*.pkl` - 进化历史记录 (pickle格式)
- `*.json` - 结果摘要 (JSON格式)
- `*.npz` - 最佳模型参数

## 💡 使用场景

1. **快速验证**: 测试新的任务组合或参数设置
2. **性能分析**: 评估不同任务的耗时和基础模型表现
3. **调试**: 小批次运行便于快速定位问题
4. **资源受限**: 在GPU资源有限时进行实验

## 🔄 与完整运行的区别

| 项目 | 小批次 | 完整运行 |
|------|--------|----------|
| PIPSIZE | 12 | 30 |
| EVAL_SUBSET | 12 | 20-30 |
| Forward Passes | 600 | 3000+ |
| 运行时间 | ~1-2小时 | ~8-12小时 |
| 用途 | 快速测试 | 正式实验 |

## 🎓 示例用法

### 示例1: 单GPU快速测试

```bash
GPUS_PER_NODE=1 \
TOTAL_FORWARD_PASSES=200 \
bash scripts/experiments/small_batch_pipsize12.sh
```

### 示例2: 完整多任务测试

```bash
GSM8K_WEIGHT=0.3 \
BFCL_WEIGHT=0.2 \
MBPP_WEIGHT=0.2 \
MULT4_WEIGHT=0.15 \
MULT5_WEIGHT=0.10 \
BOOL_WEIGHT=0.05 \
bash scripts/experiments/small_batch_pipsize12.sh
```

### 示例3: 高迭代快速测试

```bash
TOTAL_FORWARD_PASSES=1200 \
PIPSIZE=20 \
bash scripts/experiments/small_batch_pipsize12.sh
```

## 📝 注意事项

1. **权重和必须为1.0**: 确保所有任务权重相加等于1.0
2. **GPU内存**: pipsize=12 在8GB GPU上可稳定运行
3. **日志查看**: 使用 `tail -f results/.../run.log` 实时查看进度
4. **中断恢复**: 目前不支持断点续训，请确保运行不被中断

## 🆘 常见问题

**Q: 如何只测试特定任务？**
```bash
# 只测试GSM8K
GSM8K_WEIGHT=1.0 \
MBPP_WEIGHT=0.0 \
MULT4_WEIGHT=0.0 \
BOOL_WEIGHT=0.0 \
bash scripts/experiments/small_batch_pipsize12.sh
```

**Q: 如何减少运行时间？**
```bash
# 减少迭代次数
TOTAL_FORWARD_PASSES=200 bash scripts/experiments/small_batch_pipsize12.sh
```

**Q: 如何使用不同的模型？**
```bash
MODEL1_PATH="models/Llama-3.2-1B-Instruct" \
MODEL2_PATH="models/Llama-3.2-1B-Instruct" \
bash scripts/experiments/small_batch_pipsize12.sh
```

