# MBPP集成文档

## 概述

MBPP (Mostly Basic Python Problems) 是一个包含基础Python编程问题的代码生成基准测试集。本文档介绍如何在SparseFusion中使用MBPP进行多任务评估。

## 数据准备

### 1. 数据格式

MBPP数据使用JSON格式，每个样本包含：

```json
{
  "task_id": 1,
  "text": "编写一个函数，检查给定的数字是否是偶数。",
  "code": "def is_even(n):\n    return n % 2 == 0",
  "test_list": [
    "assert is_even(2) == True",
    "assert is_even(3) == False"
  ],
  "test_setup_code": "",
  "challenge_test_list": []
}
```

### 2. 获取MBPP数据

**选项1：使用示例数据（快速测试）**
```bash
# 已包含在仓库中
mbpp/data/mbpp_test_sample.json  # 5个简单样本
```

**选项2：从HuggingFace下载完整数据**
```bash
# 下载MBPP数据集
python -c "
from datasets import load_dataset
ds = load_dataset('mbpp', 'sanitized')
ds['test'].to_json('mbpp/data/mbpp_test_full.json')
"
```

**选项3：转换自定义数据**
```bash
python tools/convert_mbpp_to_simple.py \
  --input your_mbpp.jsonl \
  --output mbpp/data/mbpp_test.json \
  --limit 100
```

## 快速开始

### 1. 快速验证（推荐首次使用）

```bash
# 使用少量样本快速测试MBPP集成
bash scripts/experiments/run_mbpp_quick_test.sh
```

配置：
- 种群大小：4
- 迭代次数：50
- 评估样本：5个/任务
- 三任务：GSM8K (40%) + BFCL (30%) + MBPP (30%)

预计耗时：**10-15分钟**

### 2. 完整实验

```bash
# 三任务完整评估
python3 main_sparsity_aware.py \
  --runs 1 \
  --model1_path models/Qwen2.5-0.5B-Instruct \
  --model2_path models/Qwen2.5-0.5B-Instruct \
  --pop_size 8 \
  --total_forward_passes 3000 \
  --omega 0.7 --beta 0.3 \
  --eval_subset_size 20 \
  --use_bfcl_eval \
  --bfcl_data_path bfcl/data/bfcl_test_200.json \
  --gsm8k_weight 0.4 \
  --bfcl_weight 0.3 \
  --use_mbpp_eval \
  --mbpp_data_path mbpp/data/mbpp_test.json \
  --mbpp_weight 0.3 \
  --output_dir results/mbpp_full_test
```

## 参数说明

### MBPP相关参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--use_mbpp_eval` | flag | False | 启用MBPP评估 |
| `--mbpp_data_path` | str | `mbpp/data/mbpp_test.json` | MBPP数据路径 |
| `--mbpp_weight` | float | 0.33 | MBPP任务权重 |

### 权重配置建议

**三任务均衡（推荐）：**
```bash
--gsm8k_weight 0.4 --bfcl_weight 0.3 --mbpp_weight 0.3
```

**强调数学推理：**
```bash
--gsm8k_weight 0.6 --bfcl_weight 0.2 --mbpp_weight 0.2
```

**强调代码生成：**
```bash
--gsm8k_weight 0.3 --bfcl_weight 0.3 --mbpp_weight 0.4
```

## 评估机制

### 代码生成与执行

1. **生成阶段**
   - 模型根据问题描述生成Python函数
   - 使用低温度（0.2）确保稳定性
   - 最大生成长度：512 tokens

2. **测试执行**
   - 在隔离的临时目录中运行代码
   - 超时限制：10秒/样本
   - 执行样本的`test_list`中的所有断言

3. **判分**
   - 所有测试通过 → 1.0分
   - 任意测试失败/超时/异常 → 0.0分
   - 最终准确率 = 通过样本数 / 总样本数

### 安全机制

- **进程隔离**：每个测试在独立的subprocess中运行
- **超时控制**：防止无限循环
- **临时文件**：自动清理，避免污染
- **禁用字节码**：设置`PYTHONDONTWRITEBYTECODE=1`

## 结果分析

### 查看实验摘要

```bash
python tools/analyze_results.py \
  results/mbpp_full_test/*.pkl \
  --no-plot
```

输出示例：
```
[Run 1] 迭代步数: 300
  archive_fitness_mean: 0.4521
  archive_sparsity_mean: 0.2134
  archive_total_score_mean: 0.3856
```

### 可视化对比

```bash
# 对比不同权重配置
python tools/plot_checkpoint_comparison.py \
  --baseline results/gsm8k_only/result.pkl \
  --sparsity results/mbpp_multi_task/result.pkl \
  --output comparison_mbpp.png
```

## 常见问题

### Q1: MBPP评估很慢？

**原因**：代码执行需要启动subprocess。

**解决**：
```bash
# 减少评估样本数
--eval_subset_size 10  # 每任务只评估10个样本

# 减少迭代次数（快速测试）
--total_forward_passes 100
```

### Q2: 测试经常超时？

**原因**：生成的代码可能包含死循环或低效算法。

**解决**：
- 调整超时参数（在`natural_niches_sparsity_aware_fn.py`中修改`timeout`参数）
- 使用更强的基础模型
- 增加温度多样性（修改`temperature`参数）

### Q3: 准确率很低？

**可能原因**：
1. 模型太小（0.5B可能不足以生成复杂代码）
2. 权重配置不当
3. prompt模板需要优化

**建议**：
- 使用更大的模型（如7B）
- 调整`mbpp_data_utils.py`中的prompt模板
- 增加`eval_subset_size`观察更多样本

### Q4: 如何自定义prompt？

修改 `mbpp_data_utils.py` 中的 `_build_prompt` 方法：

```python
def _build_prompt(self, item: Dict) -> str:
    text = item.get("text", "")
    # 自定义你的prompt格式
    prompt = f"请用Python实现：{text}\n\n代码："
    return prompt
```

## 性能基准

基于Qwen2.5-0.5B-Instruct的参考性能：

| 配置 | GSM8K Pass@1 | BFCL Acc | MBPP Pass@1 | 总评估时间 |
|------|-------------|----------|-------------|-----------|
| 单任务（仅GSM8K） | ~35% | - | - | ~30min |
| 双任务（GSM8K+BFCL） | ~32% | ~28% | - | ~45min |
| 三任务（+MBPP） | ~30% | ~25% | ~15% | ~60min |

*注：以上数据基于eval_subset_size=20，total_forward_passes=3000*

## 下一步

1. **扩展数据集**
   - MBPP-Plus（更严格的测试）
   - HumanEval（更复杂的算法题）

2. **优化执行**
   - 批量执行多个测试
   - 使用Docker沙箱增强安全性

3. **改进评估**
   - 支持pass@k (k>1)
   - 代码相似度评估（非仅通过率）

## 参考资料

- [MBPP论文](https://arxiv.org/abs/2108.07732)
- [HuggingFace MBPP数据集](https://huggingface.co/datasets/mbpp)
- [EvalPlus项目](https://github.com/evalplus/evalplus)

