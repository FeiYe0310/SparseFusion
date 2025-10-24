# SparseFusion: Sparsity-Aware Natural Niches

This repository contains code for model fusion using Natural Niches evolutionary algorithm, with additional support for **sparsity-aware selection** and **Wanda pruning**.

## 🎯 Features

- **Original Natural Niches**: Evolutionary model fusion algorithm
- **Sparsity-Aware Selection**: NEW! Combines fitness and sparsity scores for model selection
- **Wanda Pruning Integration**: NEW! Active pruning during evolution
- **Distributed Training**: Multi-GPU support via PyTorch DDP

---

## 📦 Installation

### Using Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate natural_niches
```

### Using Pip

```bash
pip install torch jax jaxlib transformers datasets numpy tqdm matplotlib
```

---

## 🚀 Quick Start

### 1. Run Original Natural Niches

```bash
python run_evolution.py \
    --model1_path models/Qwen2.5-Math-1.5B-Instruct \
    --model2_path models/Qwen2.5-Coder-1.5B-Instruct \
    --pop_size 16 \
    --total_forward_passes 100
```

### 2. Run Sparsity-Aware Version (NEW!)

#### Option A: Sparsity Scoring Only (No Active Pruning)

```bash
python main_sparsity_aware.py \
    --debug_models \
    --pop_size 16 \
    --total_forward_passes 100 \
    --omega 0.5 \
    --beta 0.5 \
    --pruning_sparsity 0.0
```

#### Option B: Sparsity Scoring + Wanda Pruning

```bash
python main_sparsity_aware.py \
    --debug_models \
    --pop_size 16 \
    --total_forward_passes 100 \
    --omega 0.5 \
    --beta 0.5 \
    --pruning_sparsity 0.5 \
    --pruning_method wanda \
    --log_sparsity_stats
```

### 3. Multi-GPU Distributed Training

```bash
JAX_PLATFORM_NAME=cpu torchrun --nproc_per_node=4 main_sparsity_aware.py \
    --distributed \
    --pop_size 16 \
    --total_forward_passes 1000 \
    --omega 0.5 \
    --beta 0.5 \
    --pruning_sparsity 0.5 \
    --model1_path models/model1 \
    --model2_path models/model2 \
    --archive_backend cpu
```

---

## 📖 Parameter Guide

### Sparsity-Aware Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--omega` | 0.5 | Weight for fitness component in total score |
| `--beta` | 0.5 | Weight for sparsity component in total score |
| `--tau` | 1.0 | Softmax temperature for sparsity scores |
| `--epsilon` | 1e-10 | Threshold for considering parameters as zero |
| `--alpha` | 1.0 | Fitness normalization exponent |

### Pruning Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--pruning_sparsity` | 0.0 | Target sparsity (0.0 = disabled, 0.5 = 50% sparse) |
| `--pruning_method` | wanda | Pruning method: `wanda` or `magnitude` |
| `--log_sparsity_stats` | False | Log detailed sparsity statistics |

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--pop_size` | 16 | Population size |
| `--total_forward_passes` | 100 | Number of iterations |
| `--runs` | 1 | Number of independent runs |
| `--distributed` | False | Enable multi-GPU training |
| `--archive_backend` | gpu | Archive storage: `gpu` or `cpu` |

---

## 🧪 Example Use Cases

### Case 1: Balance Performance and Sparsity

```bash
python main_sparsity_aware.py \
    --pop_size 16 \
    --total_forward_passes 500 \
    --omega 0.5 \
    --beta 0.5 \
    --pruning_sparsity 0.5
```

**Expected**: ~50% sparsity while maintaining good performance.

---

### Case 2: Prioritize Sparsity

```bash
python main_sparsity_aware.py \
    --pop_size 16 \
    --total_forward_passes 500 \
    --omega 0.2 \
    --beta 0.8 \
    --pruning_sparsity 0.7 \
    --tau 0.5
```

**Expected**: >70% sparsity, lower performance.

---

### Case 3: Prioritize Performance

```bash
python main_sparsity_aware.py \
    --pop_size 16 \
    --total_forward_passes 500 \
    --omega 0.8 \
    --beta 0.2 \
    --pruning_sparsity 0.3 \
    --tau 2.0
```

**Expected**: ~30% sparsity, higher performance.

---

## 📊 How It Works

### Total Score Formula

The sparsity-aware version combines two scores:

```
Total Score = ω × Fitness + β × Sparsity
```

Where:
- **Fitness**: Normalized performance on tasks
- **Sparsity**: Softmax-normalized proportion of near-zero parameters
- **ω (omega)**: Weight for fitness (default: 0.5)
- **β (beta)**: Weight for sparsity (default: 0.5)

### Evolution Process

```
For each iteration:
  1. Select parents based on Total Score
  2. Apply Wanda pruning (if enabled)
  3. Crossover
  4. Mutation
  5. Evaluate child
  6. Update archive based on Total Score
```

---

## 📁 Project Structure

```
SparseFusion/
├── docs/                      # Reference guides and deployment notes
├── scripts/
│   ├── deploy/                # Sync and infrastructure helpers
│   ├── experiments/           # Common experiment launchers
│   ├── tests/                 # Shell-based integration checks
│   ├── run_merge.sh
│   └── run_with_models.sh
├── tools/                     # Standalone analysis & plotting utilities
├── bfcl/                      # BFCL benchmark assets
├── datasets/
├── lib/                       # Wanda pruning utilities
├── results/                   # Experiment outputs (gitignored)
├── config.py                  # Path configuration
├── main.py                    # Natural Niches entry point
├── main_sparsity_aware.py     # Sparsity-aware entry point
├── run_evolution.py           # Torchrun-ready launcher
└── requirements.txt           # Dependencies
```

Extensive setup and deployment documentation now lives under `docs/`, while shell launchers are grouped in `scripts/` and analysis helpers moved to `tools/` for easier discovery.

---

## 🔬 Running Experiments

### Comparative Experiments

Run baseline vs. sparsity-aware methods:

```bash
# Baseline
python run_evolution.py --debug_models --pop_size 16 --total_forward_passes 100

# Sparsity-aware (scoring only)
python main_sparsity_aware.py --debug_models --pop_size 16 --total_forward_passes 100 \
    --omega 0.5 --beta 0.5 --pruning_sparsity 0.0

# Sparsity-aware (with pruning)
python main_sparsity_aware.py --debug_models --pop_size 16 --total_forward_passes 100 \
    --omega 0.5 --beta 0.5 --pruning_sparsity 0.5
```

---

## 🐛 Test

```bash
PYTHONPATH="$PWD" pytest tests/test_helper_fn.py
```

## 🐛 Troubleshooting

### ImportError: No module named 'lib'

**Solution**: Ensure you're running from the project root directory:
```bash
cd /path/to/SparseFusion
python main_sparsity_aware.py ...
```

### CUDA Out of Memory

**Solution**: Use CPU backend for archive:
```bash
python main_sparsity_aware.py --archive_backend cpu ...
```

### Wanda Pruning Fails

**Solution**: The code will automatically skip pruning and continue. Check logs for warnings.

---

## 📝 Citation

If you use this code, please cite:

```bibtex
@article{sakana2025m2n2,
  title={Competition and Attraction Improve Model Fusion},
  author={Abrantes, Jo\~{a}o and Lange, Robert and Tang, Yujin},
  booktitle={Proceedings of the 2025 genetic and evolutionary computation conference},
  pages={1217--1225},
  year={2025}
}
```

---

## 📄 License

Apache-2.0 License

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📧 Contact

For questions or issues, please open an issue on GitHub.
