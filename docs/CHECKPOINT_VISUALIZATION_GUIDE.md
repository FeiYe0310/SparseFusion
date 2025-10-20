# Checkpoint可视化对比工具使用指南

## 📊 功能简介

这个工具用于比较两个checkpoint的性能，并生成详细的可视化图表，包括：

1. **Fitness Evolution**: 显示GSM8K准确率随训练步数的变化
2. **Sparsity Evolution**: 显示模型稀疏度随训练步数的变化
3. **Total Score Evolution**: 显示总分数（ω·fitness + β·sparsity）的演化
4. **Pareto Front**: 显示最终archive中个体的分布（fitness vs sparsity）

## 🚀 快速开始

### 方法1: 使用默认路径（最简单）

```bash
cd /mnt/shared-storage-user/yefei/SparseFusion
bash scripts/experiments/run_plot_comparison.sh
```

这将使用以下默认checkpoint：
- **Baseline**: `results/checkpoints/checkpoint_run1_step3000_20251017_021740.pkl`
- **Sparsity-Aware**: `results/sparsity_aware_w0.80_b0.20_t1.00_prune_wanda_0.30.pkl`
- **输出**: `comparison_baseline_vs_sparsity.png`

### 方法2: 自定义路径

```bash
BASELINE_CHECKPOINT="/path/to/baseline.pkl" \
SPARSITY_CHECKPOINT="/path/to/sparsity.pkl" \
OUTPUT_PLOT="my_comparison.png" \
bash scripts/experiments/run_plot_comparison.sh
```

### 方法3: 直接使用Python脚本

```bash
python tools/plot_checkpoint_comparison.py \
    --baseline /path/to/baseline.pkl \
    --sparsity /path/to/sparsity.pkl \
    --output comparison.png
```

## 📈 输出示例

运行后会生成：

1. **PNG图表文件**（4个子图）：
   - 左上：Fitness Evolution（蓝色=baseline, 红色=sparsity-aware）
   - 右上：Sparsity Evolution
   - 左下：Total Score Evolution
   - 右下：Pareto Front Distribution

2. **终端统计摘要**：
   ```
   ============================================================
   FINAL STATISTICS SUMMARY
   ============================================================
   
   📊 Baseline (Final):
     Max Fitness:      0.4567
     Mean Fitness:     0.3821
     Max Sparsity:     0.1234
     Mean Sparsity:    0.0956
     Max Total Score:  0.4123
     Mean Total Score: 0.3456
   
   📊 Sparsity-Aware (Final):
     Max Fitness:      0.4892
     Mean Fitness:     0.4156
     Max Sparsity:     0.3456
     Mean Sparsity:    0.2987
     Max Total Score:  0.5234
     Mean Total Score: 0.4567
   
   📈 Improvement (Sparsity-Aware vs Baseline):
     Fitness:      +7.12%
     Sparsity:     +180.13%
     Total Score:  +26.94%
   ============================================================
   ```

## 🔧 命令行参数

### run_plot_comparison.sh 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `BASELINE_CHECKPOINT` | Baseline checkpoint路径 | `results/checkpoints/checkpoint_run1_step3000_20251017_021740.pkl` |
| `SPARSITY_CHECKPOINT` | Sparsity-aware checkpoint路径 | `results/sparsity_aware_w0.80_b0.20_t1.00_prune_wanda_0.30.pkl` |
| `OUTPUT_PLOT` | 输出图表文件名 | `comparison_baseline_vs_sparsity.png` |
| `ROOTPATH` | 工作根目录 | `/mnt/shared-storage-user/yefei` |

### tools/plot_checkpoint_comparison.py 参数

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `--baseline` | str | ✅ | Baseline checkpoint路径 |
| `--sparsity` | str | ✅ | Sparsity-aware checkpoint路径 |
| `--output` | str | ❌ | 输出文件名（默认：checkpoint_comparison.png）|

## 📝 Checkpoint格式要求

工具期望checkpoint文件是pickle格式（.pkl），包含以下结构：

```python
{
    'history': [
        {
            'forward_passes': int,
            'fitness': [float, ...],        # 每个个体的fitness
            'sparsity': [float, ...],       # 每个个体的sparsity
            'total_score': [float, ...],    # 每个个体的total score
        },
        ...
    ],
    'archive': {
        'individual_id': {
            'fitness': float,
            'sparsity': float,
            'total_score': float,
        },
        ...
    }
}
```

## 💡 使用场景

### 场景1: 对比Baseline vs Sparsity-Aware

```bash
# 默认就是这个场景
bash scripts/experiments/run_plot_comparison.sh
```

### 场景2: 对比不同超参数配置

```bash
python tools/plot_checkpoint_comparison.py \
    --baseline results/sparsity_w0.50_b0.50.pkl \
    --sparsity results/sparsity_w0.80_b0.20.pkl \
    --output compare_hyperparams.png
```

### 场景3: 对比不同剪枝方法

```bash
python tools/plot_checkpoint_comparison.py \
    --baseline results/prune_magnitude_0.30.pkl \
    --sparsity results/prune_wanda_0.30.pkl \
    --output compare_pruning_methods.png
```

## 🎨 图表自定义

如需修改图表样式，可以编辑 `tools/plot_checkpoint_comparison.py` 中的 `plot_comparison()` 函数：

- **颜色**: 修改 `'b-'`, `'r-'` 等颜色代码
- **线型**: 修改 `'-'`, `'--'` 等线型
- **标记**: 修改 `marker='o'`, `marker='^'` 等标记样式
- **DPI**: 修改 `plt.savefig(..., dpi=300)` 中的DPI值
- **图表大小**: 修改 `figsize=(16, 12)` 参数

## ❓ 常见问题

### Q1: 报错 "Checkpoint not found"
**A**: 检查checkpoint路径是否正确，确保文件存在：
```bash
ls -lh /path/to/your/checkpoint.pkl
```

### Q2: 报错 "KeyError: 'history'"
**A**: Checkpoint文件可能不包含history数据。确认checkpoint是用 `natural_niches_sparsity_aware_fn.py` 生成的。

### Q3: 图表中某些曲线为空
**A**: 正常现象。如果某个checkpoint没有记录某个指标（如baseline的sparsity可能都是0），对应曲线会很平。

### Q4: 如何在本地查看生成的图表？
**A**: 使用scp下载到本地：
```bash
scp user@server:/path/to/comparison.png ./
```

## 📚 相关文件

- `tools/plot_checkpoint_comparison.py`: Python绘图脚本
- `run_plot_comparison.sh`: Shell启动脚本
- `natural_niches_sparsity_aware_fn.py`: 生成checkpoint的主程序
- `main_sparsity_aware.py`: 实验入口

## 🔗 配合使用的脚本

1. **训练Baseline**:
   ```bash
   bash scripts/experiments/run_baseline.sh
   ```

2. **训练Sparsity-Aware**:
   ```bash
   bash scripts/experiments/run_sparsity_single_node.sh
   ```

3. **可视化对比**:
   ```bash
   bash scripts/experiments/run_plot_comparison.sh
   ```

---

**创建日期**: 2025-10-17  
**作者**: SparseFusion Team  
**版本**: 1.0
