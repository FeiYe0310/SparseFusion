#!/usr/bin/env python3
"""直接测试plot_detailed_comparison.py中的load函数"""

import sys
sys.path.insert(0, '/mnt/shared-storage-user/yefei/SparseFusion')

from plot_detailed_comparison import load_and_extract_data

baseline_path = '/mnt/shared-storage-user/yefei/SparseFusion/results/checkpoints/checkpoint_run1_step3000_20251017_021740.pkl'
sparsity_path = '/mnt/shared-storage-user/yefei/SparseFusion/results/sparsity_aware_w0.80_b0.20_t1.00_prune_wanda_0.30.pkl'

print("="*70)
print("Testing load_and_extract_data function")
print("="*70)

print("\n--- Loading Baseline ---")
baseline_data = load_and_extract_data(baseline_path)
print(f"Iterations: {len(baseline_data['iterations'])}")
print(f"Individuals: {len(baseline_data['individuals'])}")
print(f"Individual IDs: {baseline_data['individuals']}")
if len(baseline_data['iterations']) > 0:
    print(f"First iteration: {baseline_data['iterations'][0]}")
    print(f"Last iteration: {baseline_data['iterations'][-1]}")

print("\n--- Loading Sparsity-Aware ---")
sparsity_data = load_and_extract_data(sparsity_path)
print(f"Iterations: {len(sparsity_data['iterations'])}")
print(f"Individuals: {len(sparsity_data['individuals'])}")
print(f"Individual IDs: {sparsity_data['individuals']}")
if len(sparsity_data['iterations']) > 0:
    print(f"First iteration: {sparsity_data['iterations'][0]}")
    print(f"Last iteration: {sparsity_data['iterations'][-1]}")

