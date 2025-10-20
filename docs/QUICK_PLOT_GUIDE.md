# 快速绘图指南 📊

## 🎯 一键运行（在服务器上）

```bash
cd /mnt/shared-storage-user/yefei/SparseFusion
bash scripts/experiments/run_plot_comparison.sh
```

这将对比以下两个checkpoints：
- **Baseline**: `results/checkpoints/checkpoint_run1_step3000_20251017_021740.pkl`
- **Sparsity-Aware**: `results/sparsity_aware_w0.80_b0.20_t1.00_prune_wanda_0.30.pkl`

生成的图表：`comparison_baseline_vs_sparsity.png`

## 📊 输出内容

生成的图表包含4个子图：

1. **Fitness Evolution** (左上)
   - 蓝色 = Baseline
   - 红色 = Sparsity-Aware
   - 实线 = Max fitness
   - 虚线 = Mean fitness

2. **Sparsity Evolution** (右上)
   - 显示稀疏度随训练的变化

3. **Total Score Evolution** (左下)
   - 显示总分数 (ω·fitness + β·sparsity)

4. **Pareto Front** (右下)
   - 最终archive中个体的分布
   - X轴 = Sparsity, Y轴 = Fitness

## 📈 统计摘要

运行后会在终端显示：
- 两个模型的最终指标对比
- 改进百分比（Sparsity-Aware vs Baseline）

## 🔧 自定义对比

### 对比不同的checkpoints：

```bash
python tools/plot_checkpoint_comparison.py \
    --baseline /path/to/checkpoint1.pkl \
    --sparsity /path/to/checkpoint2.pkl \
    --output my_comparison.png
```

### 使用环境变量：

```bash
BASELINE_CHECKPOINT="/path/to/baseline.pkl" \
SPARSITY_CHECKPOINT="/path/to/sparsity.pkl" \
OUTPUT_PLOT="my_plot.png" \
bash scripts/experiments/run_plot_comparison.sh
```

## 📥 下载图表到本地

```bash
scp user@server:/mnt/shared-storage-user/yefei/SparseFusion/comparison_baseline_vs_sparsity.png ./
```

## ❓ 常见问题

**Q: 报错 "ModuleNotFoundError: No module named 'matplotlib'"**
```bash
pip install matplotlib
```

**Q: 图表显示为空**
- 检查checkpoint文件是否包含 `history` 数据
- 确认checkpoint路径正确

**Q: 格式不兼容**
- 脚本已支持dict和list两种checkpoint格式
- 如仍有问题，请检查checkpoint结构

---

详细文档请参阅：`CHECKPOINT_VISUALIZATION_GUIDE.md`

