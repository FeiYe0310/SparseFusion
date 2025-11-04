# Configuration System

统一的配置管理系统，集中管理所有超参数和设置。

## 📁 结构

```
config/
├── __init__.py              # 配置模块入口
├── model_config.py          # 模型相关配置
├── training_config.py       # 训练相关配置
├── evaluation_config.py     # 评估相关配置
├── sparsity_config.py       # 稀疏度相关配置
├── task_config.py           # 多任务配置
├── presets.py               # 预设配置
└── README.md                # 本文件
```

## 🚀 快速开始

### 方式1: 使用预设配置

```python
from config.presets import load_preset, print_config

# 加载预设
config = load_preset("multi_task")

# 查看配置
print_config(config)

# 使用配置
model_cfg = config["model"]
training_cfg = config["training"]
eval_cfg = config["evaluation"]
sparsity_cfg = config["sparsity"]
task_cfg = config["task"]
```

### 方式2: 自定义配置

```python
from config import (
    ModelConfig,
    TrainingConfig,
    EvaluationConfig,
    SparsityConfig,
    TaskConfig,
)

# 创建自定义配置
model_cfg = ModelConfig(
    model1_path="models/wizardmath_7b",
    model2_path="models/agentevol-7b",
)

training_cfg = TrainingConfig(
    pop_size=16,
    total_forward_passes=10000,
    distributed=True,
)

eval_cfg = EvaluationConfig(
    batch_size=16,
    eval_subset_size=15,
)

sparsity_cfg = SparsityConfig(
    omega=0.5,
    beta=0.5,
    use_dynamic_sparsity=True,
)

task_cfg = TaskConfig(
    use_gsm8k=True,
    use_mbpp_eval=True,
    gsm8k_weight=0.5,
    mbpp_weight=0.5,
)
```

### 方式3: 修改预设配置

```python
from config.presets import load_preset

# 加载预设
config = load_preset("default")

# 修改部分参数
config["training"].pop_size = 32
config["evaluation"].batch_size = 32
config["sparsity"].use_dynamic_sparsity = True
```

## 📋 可用预设

### `default`
默认配置，适合单任务GSM8K训练：
- Population size: 20
- Batch size: 16
- No dynamic sparsity

### `quick_test`
快速测试配置，用于调试：
- Debug models (BERT)
- Small population (8)
- 100 forward passes
- Small batch size (4)

### `multi_task`
多任务学习配置（GSM8K + MBPP）：
- Population: 16
- Dynamic sparsity enabled
- Few-shot prompting enabled

### `dynamic_sparsity`
动态稀疏度调度：
- Cosine annealing with warm restarts
- Sparsity range: [0.1, 0.6]
- Wanda pruning

### `high_performance`
高性能配置：
- Large batch size (32)
- Distributed training
- GPU archive backend

## 🔧 配置类详解

### ModelConfig
- `model1_path`, `model2_path`: 模型路径
- `debug_models`: 使用调试模型
- `max_new_tokens`: 最大生成token数
- `temperature`, `top_p`: 采样参数

### TrainingConfig
- `pop_size`: 种群大小
- `total_forward_passes`: 总前向传播次数
- `runs`: 独立运行次数
- `distributed`: 分布式训练
- `archive_backend`: Archive后端 ("gpu"/"cpu")

### EvaluationConfig
- `batch_size`: 批次大小
- `eval_subset_size`: 评估子集大小
- `gsm8k_qwen_chat`: GSM8K使用Qwen chat模板
- `mbpp_qwen_chat`: MBPP使用Qwen chat模板
- `*_few_shot_k`: Few-shot示例数量

### SparsityConfig
- `omega`, `beta`: 适应度和稀疏度权重
- `tau`: Softmax温度
- `pruning_sparsity`: 目标稀疏度
- `use_dynamic_sparsity`: 启用动态稀疏度
- `sparsity_min`, `sparsity_max`: 稀疏度范围

### TaskConfig
- `use_gsm8k`, `use_mbpp_eval`, etc.: 启用任务
- `gsm8k_weight`, `mbpp_weight`, etc.: 任务权重
- `*_data_path`: 数据集路径

## 💡 最佳实践

1. **开始新实验时**，先选择一个预设：
   ```python
   config = load_preset("multi_task")
   ```

2. **调整参数**，只修改需要改的：
   ```python
   config["training"].pop_size = 24
   config["evaluation"].batch_size = 24
   ```

3. **记录配置**到日志：
   ```python
   import json
   
   config_dict = {
       name: cfg.to_dict()
       for name, cfg in config.items()
   }
   
   with open("experiment_config.json", "w") as f:
       json.dump(config_dict, f, indent=2)
   ```

## 📝 扩展配置

如需添加新参数：

1. 在对应的配置类中添加字段
2. 更新 `to_dict()` 方法
3. 如需验证，在 `__post_init__()` 中添加

示例：
```python
@dataclass
class ModelConfig:
    # 新增参数
    use_flash_attention: bool = False
    
    def to_dict(self):
        return {
            # ... existing fields ...
            "use_flash_attention": self.use_flash_attention,
        }
```

