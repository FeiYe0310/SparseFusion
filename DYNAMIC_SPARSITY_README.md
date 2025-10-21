# 🔄 动态稀疏度调度 - 使用指南

## 概述

动态稀疏度调度是基于 **Cosine Annealing with Warm Restarts (SGDR)** 论文实现的一种自适应剪枝策略，通过周期性地调整剪枝稀疏度，增强进化算法在不同稀疏生态位上的探索能力。

**论文**: [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983)

**核心修改**: 我们使用 `sin(π/2)` 替代原文的余弦函数，实现从低稀疏度到高稀疏度的 **warm-up** 效果。

---

## 设计思路

### 1. 为什么需要动态稀疏度？

在传统的固定稀疏度剪枝中：
- 稀疏度设置过高：可能过早地破坏有潜力的模型
- 稀疏度设置过低：探索空间受限，无法充分利用稀疏性优势

动态稀疏度调度通过**周期性地变化稀疏度**，解决这个两难问题：
- **初期低稀疏度**：保护高fitness但低sparsity的优秀个体
- **逐渐增加稀疏度**：探索更稀疏的模型空间
- **周期性重启**：避免陷入局部最优，增加种群多样性

### 2. 数学公式

动态稀疏度在第 `t` 次迭代时的值为：

```
η_t = η_min + 0.5 × (η_max - η_min) × (1 + sin(T_cur/T_i × π/2))
```

其中：
- `η_min`: 稀疏度最小值（例如 0.1）
- `η_max`: 稀疏度最大值（例如 0.6）
- `T_cur`: 当前周期内已经过的迭代次数
- `T_i`: 当前周期的总迭代次数

每个周期结束后：
- 稀疏度重启到 `η_min`
- 下一周期长度 `T_i` 乘以 `T_mult`（例如 2 表示每次翻倍）

### 3. 调度示例

假设配置为：
- `η_min = 0.1`
- `η_max = 0.6`
- `T_0 = 100`（第一周期100次迭代）
- `T_mult = 2`（每次周期翻倍）

稀疏度变化如下：

```
迭代 0-99:    周期1 (100次)  稀疏度: 0.10 → 0.60
迭代 100-299:  周期2 (200次)  稀疏度: 0.10 → 0.60  [重启!]
迭代 300-699:  周期3 (400次)  稀疏度: 0.10 → 0.60  [重启!]
迭代 700-1499: 周期4 (800次)  稀疏度: 0.10 → 0.60  [重启!]
迭代 1500-...: 周期5 (1600次) 稀疏度: 0.10 → 0.60  [重启!]
```

---

## 使用方法

### 方法1: 使用预配置脚本（推荐）

```bash
# 直接运行动态稀疏度实验
bash run_bfcl_dynamic_sparsity.sh
```

### 方法2: 自定义配置

```bash
export USE_DYNAMIC_SPARSITY=true
export SPARSITY_MIN=0.1
export SPARSITY_MAX=0.6
export SPARSITY_T0=100
export SPARSITY_T_MULT=2

bash scripts/experiments/run_bfcl_single_node.sh
```

### 方法3: 命令行参数

```bash
python main_sparsity_aware.py \
  --model1_path models/Qwen2.5-0.5B-Instruct \
  --model2_path models/Qwen2.5-0.5B-Instruct \
  --pop_size 8 \
  --total_forward_passes 3000 \
  --runs 1 \
  --omega 0.9 \
  --beta 0.1 \
  --use_dynamic_sparsity \
  --sparsity_min 0.1 \
  --sparsity_max 0.6 \
  --sparsity_t0 100 \
  --sparsity_t_mult 2 \
  --eval_subset_size 20 \
  --use_bfcl_eval \
  --bfcl_data_path bfcl/data/bfcl_test_200.json \
  --gsm8k_weight 0.5 \
  --bfcl_weight 0.5 \
  --output_dir results_dynamic
```

---

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| `--use_dynamic_sparsity` | flag | False | 启用动态稀疏度调度（会覆盖 `--pruning_sparsity`） |
| `--sparsity_min` | float | 0.1 | 稀疏度最小值（周期开始时的值） |
| `--sparsity_max` | float | 0.6 | 稀疏度最大值（周期结束时的值） |
| `--sparsity_t0` | int | 100 | 第一个周期的迭代次数 |
| `--sparsity_t_mult` | int | 2 | 周期长度乘数（1=固定周期，2=每次翻倍） |

### 参数选择建议

#### 1. `sparsity_min` 和 `sparsity_max`
- **小模型（<1B参数）**: `[0.1, 0.5]`
- **中等模型（1-7B参数）**: `[0.1, 0.6]`
- **大模型（>7B参数）**: `[0.2, 0.7]`

#### 2. `sparsity_t0`（第一周期长度）
- **快速探索**: `50-100` 次迭代
- **平衡模式**: `100-200` 次迭代
- **稳定收敛**: `200-500` 次迭代

建议：`T_0 ≈ total_iterations / 30`

#### 3. `sparsity_t_mult`（周期乘数）
- `T_mult = 1`: 固定周期（每个周期长度相同）
  - 优点：重启频繁，探索多样
  - 缺点：后期可能不稳定
- `T_mult = 2`: 指数增长（推荐）
  - 优点：初期快速探索，后期稳定收敛
  - 缺点：后期重启次数减少
- `T_mult = 3`: 快速增长
  - 优点：后期非常稳定
  - 缺点：重启次数很少

---

## 实验对比

### 对比1: 动态 vs 固定稀疏度

**固定稀疏度 (pruning_sparsity=0.3)**:
```bash
export PRUNING_SPARSITY=0.3
export USE_DYNAMIC_SPARSITY=false
bash scripts/experiments/run_bfcl_single_node.sh
```

**动态稀疏度 (0.1→0.6)**:
```bash
export USE_DYNAMIC_SPARSITY=true
export SPARSITY_MIN=0.1
export SPARSITY_MAX=0.6
bash scripts/experiments/run_bfcl_single_node.sh
```

### 对比2: 不同周期策略

**固定周期 (T_mult=1)**:
```bash
export SPARSITY_T_MULT=1  # 每100次迭代重启
```

**指数增长 (T_mult=2)**:
```bash
export SPARSITY_T_MULT=2  # 100 → 200 → 400 → ...
```

---

## 日志输出

启用动态稀疏度后，您会看到如下日志：

```
🔄 Dynamic Sparsity ENABLED: method=wanda
   Sparsity range: [0.10, 0.60]
   First cycle: 100 iterations
   Cycle multiplier: 2x

🔄 [Iter 0] Dynamic sparsity: 0.1000
🔄 [Iter 10] Dynamic sparsity: 0.1244
...
🔄 [Iter 90] Dynamic sparsity: 0.5756
🔄 [Iter 100] Dynamic sparsity: 0.1000  # 重启！
```

---

## 常见问题

### Q1: 动态稀疏度会覆盖 `--pruning_sparsity` 吗？

**是的**。当启用 `--use_dynamic_sparsity` 时，`--pruning_sparsity` 参数会被忽略。

### Q2: 如何选择合适的稀疏度范围？

建议从较大范围开始（例如 `[0.1, 0.6]`），观察实验结果后再调整：
- 如果模型在高稀疏度时表现崩溃，减小 `sparsity_max`
- 如果低稀疏度阶段收益不明显，增大 `sparsity_min`

### Q3: 周期长度应该如何设置？

经验法则：
```
T_0 = total_iterations / (10 × log2(total_iterations))
```

例如：
- 1000次迭代: `T_0 ≈ 100`
- 3000次迭代: `T_0 ≈ 200`
- 10000次迭代: `T_0 ≈ 300`

### Q4: 动态稀疏度适用于所有任务吗？

不一定。建议先在小规模实验上对比：
- 任务对稀疏度敏感：动态调度效果好
- 任务对稀疏度不敏感：固定稀疏度可能更稳定

---

## 实现细节

### 核心函数

```python
def calculate_dynamic_sparsity(
    current_iteration: int,
    eta_min: float,
    eta_max: float,
    t0: int,
    t_mult: int
) -> float:
    """
    计算当前迭代的动态稀疏度
    """
    t_i = float(t0)
    t_cur = float(current_iteration)
    
    # 找到当前周期
    while t_cur >= t_i:
        t_cur -= t_i
        t_i *= t_mult
    
    # 正弦warm-up公式
    sparsity_ratio = 0.5 * (1 + math.sin((t_cur / t_i) * math.pi / 2))
    return eta_min + (eta_max - eta_min) * sparsity_ratio
```

### 集成位置

在 `natural_niches_sparsity_aware_fn.py` 的主循环中（第1240-1253行）：

```python
if enable_pruning:
    # 动态计算当前稀疏度
    if use_dynamic_sparsity:
        current_pruning_sparsity = calculate_dynamic_sparsity(
            current_iteration=i,
            eta_min=sparsity_min,
            eta_max=sparsity_max,
            t0=sparsity_t0,
            t_mult=sparsity_t_mult
        )
    else:
        current_pruning_sparsity = pruning_sparsity
    
    # 使用动态稀疏度进行剪枝
    prune_with_wanda(..., current_pruning_sparsity, ...)
```

---

## 相关文件

- `natural_niches_sparsity_aware_fn.py`: 核心实现（第119-193行）
- `main_sparsity_aware.py`: 命令行参数（第80-90行）
- `run_bfcl_dynamic_sparsity.sh`: 预配置运行脚本
- `scripts/experiments/run_bfcl_single_node.sh`: 通用启动脚本（支持动态稀疏度）

---

## 引用

如果您在研究中使用了动态稀疏度调度，请引用以下论文：

```bibtex
@inproceedings{loshchilov2017sgdr,
  title={SGDR: Stochastic gradient descent with warm restarts},
  author={Loshchilov, Ilya and Hutter, Frank},
  booktitle={International Conference on Learning Representations},
  year={2017}
}
```

---

**祝实验顺利！** 🚀

如有问题，请查看日志输出或联系开发者。


