# MBPP快速开始指南

## 🎯 什么是MBPP集成？

MBPP (Mostly Basic Python Problems) 现已集成到SparseFusion多任务评估中，支持：
- ✅ GSM8K（数学推理）
- ✅ BFCL（函数调用）
- ✅ MBPP（代码生成） ← **新增**

## 🚀 5分钟快速体验

### 1. 准备数据（使用示例数据）

```bash
# 示例数据已包含在仓库中
ls mbpp/data/mbpp_test_sample.json
# 包含5个简单的Python编程问题
```

### 2. 运行快速测试

```bash
chmod +x scripts/experiments/run_mbpp_quick_test.sh
bash scripts/experiments/run_mbpp_quick_test.sh
```

**预计耗时：** 10-15分钟  
**配置：** 4个体，50步迭代，每任务5样本

### 3. 查看结果

```bash
python tools/analyze_results.py \
  results/mbpp_quick_test/*.pkl \
  --no-plot
```

## 📊 完整三任务实验

### 命令示例

```bash
python3 main_sparsity_aware.py \
  --runs 1 \
  --model1_path models/Qwen2.5-0.5B-Instruct \
  --model2_path models/Qwen2.5-0.5B-Instruct \
  --pop_size 8 \
  --total_forward_passes 3000 \
  --omega 0.7 --beta 0.3 \
  --pruning_sparsity 0.2 \
  --eval_subset_size 20 \
  --use_bfcl_eval \
  --bfcl_data_path bfcl/data/bfcl_test_200.json \
  --gsm8k_weight 0.4 \
  --bfcl_weight 0.3 \
  --use_mbpp_eval \
  --mbpp_data_path mbpp/data/mbpp_test_sample.json \
  --mbpp_weight 0.3 \
  --output_dir results/三任务实验
```

### 环境变量方式（推荐用于脚本）

```bash
export USE_MBPP_EVAL=true
export MBPP_DATA_PATH=mbpp/data/mbpp_test_sample.json
export MBPP_WEIGHT=0.3
export GSM8K_WEIGHT=0.4
export BFCL_WEIGHT=0.3

bash scripts/experiments/run_bfcl_single_node.sh
```

## 🔧 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--use_mbpp_eval` | 启用MBPP评估 | False |
| `--mbpp_data_path` | MBPP数据路径 | `mbpp/data/mbpp_test.json` |
| `--mbpp_weight` | MBPP任务权重（0-1） | 0.33 |
| `--gsm8k_weight` | GSM8K任务权重 | 0.5 |
| `--bfcl_weight` | BFCL任务权重 | 0.5 |

**注意：** 权重会自动归一化，无需手动保证和为1。

## 📈 权重配置建议

### 场景1：均衡三任务
```bash
--gsm8k_weight 0.4 --bfcl_weight 0.3 --mbpp_weight 0.3
```
适用：全面评估模型能力

### 场景2：强调代码能力
```bash
--gsm8k_weight 0.3 --bfcl_weight 0.3 --mbpp_weight 0.4
```
适用：代码生成任务优先

### 场景3：数学为主
```bash
--gsm8k_weight 0.6 --bfcl_weight 0.2 --mbpp_weight 0.2
```
适用：数学推理任务为主

## 📁 数据准备选项

### 选项1：使用示例数据（最快）
```bash
# 已包含，直接使用
--mbpp_data_path mbpp/data/mbpp_test_sample.json
```

### 选项2：下载完整MBPP
```bash
# 从HuggingFace下载
python -c "
from datasets import load_dataset
ds = load_dataset('mbpp', 'sanitized')
ds['test'].to_json('mbpp/data/mbpp_test_full.json')
"

# 使用完整数据
--mbpp_data_path mbpp/data/mbpp_test_full.json
```

### 选项3：转换自定义数据
```bash
python tools/convert_mbpp_to_simple.py \
  --input your_data.jsonl \
  --output mbpp/data/custom.json \
  --limit 100
```

## 🐛 常见问题

### Q: MBPP评估很慢？
```bash
# 减少样本数加速
--eval_subset_size 10  # 每任务10样本
```

### Q: 测试经常超时？
- 原因：代码生成质量较低或包含死循环
- 解决：使用更大的模型或调整超时参数

### Q: 准确率为0？
- 检查数据格式是否正确
- 查看生成的代码：添加 `--log_sparsity_stats` 查看详细日志
- 尝试降低模型复杂度

## 📖 详细文档

- **完整集成文档：** [docs/MBPP_INTEGRATION.md](docs/MBPP_INTEGRATION.md)
- **BFCL文档：** [docs/BFCL_QUICK_START.md](docs/BFCL_QUICK_START.md)
- **基础使用：** [README.md](README.md)

## 🎯 下一步

1. **运行基准测试**
   ```bash
   bash scripts/experiments/run_mbpp_quick_test.sh
   ```

2. **对比不同配置**
   ```bash
   python tools/plot_checkpoint_comparison.py \
     --baseline results/baseline/*.pkl \
     --sparsity results/mbpp_test/*.pkl \
     --output mbpp_comparison.png
   ```

3. **完整实验**
   - 使用完整MBPP数据集
   - 增加迭代次数到3000+
   - 尝试不同的权重配置

## 💡 提示

- 首次使用建议先跑快速测试验证环境
- MBPP评估需要执行代码，确保有足够的CPU/内存
- 权重配置对最终性能影响很大，建议多试几组
- 使用`--eval_subset_size`可显著加速评估（推荐20-30）

---

**🎉 现在开始体验三任务联合训练！**

