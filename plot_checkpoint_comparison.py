#!/usr/bin/env python3
"""
Checkpoint Comparison Visualization Script

比较两个checkpoint的性能：
1. Baseline checkpoint (无sparsity-aware)
2. Sparsity-aware checkpoint (带剪枝)

绘制：
- Fitness evolution over forward passes
- Sparsity evolution over forward passes
- Total score (omega * fitness + beta * sparsity) evolution
- Pareto front comparison
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def load_checkpoint(checkpoint_path):
    """加载checkpoint文件"""
    print(f"Loading checkpoint: {checkpoint_path}")
    with open(checkpoint_path, 'rb') as f:
        data = pickle.load(f)
    
    # 打印checkpoint信息
    print(f"  Keys: {list(data.keys())}")
    if 'history' in data:
        print(f"  History steps: {len(data['history'])}")
    if 'archive' in data:
        print(f"  Archive size: {len(data['archive'])}")
    
    return data


def extract_history(checkpoint_data):
    """从checkpoint提取历史数据"""
    history = checkpoint_data.get('history', [])
    
    forward_passes = []
    max_fitness = []
    mean_fitness = []
    max_sparsity = []
    mean_sparsity = []
    max_total_score = []
    mean_total_score = []
    
    for entry in history:
        forward_passes.append(entry.get('forward_passes', 0))
        
        # Fitness统计
        fitness_vals = entry.get('fitness', [])
        if fitness_vals:
            max_fitness.append(np.max(fitness_vals))
            mean_fitness.append(np.mean(fitness_vals))
        else:
            max_fitness.append(0)
            mean_fitness.append(0)
        
        # Sparsity统计
        sparsity_vals = entry.get('sparsity', [])
        if sparsity_vals:
            max_sparsity.append(np.max(sparsity_vals))
            mean_sparsity.append(np.mean(sparsity_vals))
        else:
            max_sparsity.append(0)
            mean_sparsity.append(0)
        
        # Total score统计
        total_score_vals = entry.get('total_score', [])
        if total_score_vals:
            max_total_score.append(np.max(total_score_vals))
            mean_total_score.append(np.mean(total_score_vals))
        else:
            max_total_score.append(0)
            mean_total_score.append(0)
    
    return {
        'forward_passes': np.array(forward_passes),
        'max_fitness': np.array(max_fitness),
        'mean_fitness': np.array(mean_fitness),
        'max_sparsity': np.array(max_sparsity),
        'mean_sparsity': np.array(mean_sparsity),
        'max_total_score': np.array(max_total_score),
        'mean_total_score': np.array(mean_total_score),
    }


def extract_archive_data(checkpoint_data):
    """从checkpoint提取archive中的个体数据"""
    archive = checkpoint_data.get('archive', {})
    
    fitness_list = []
    sparsity_list = []
    total_score_list = []
    
    for individual in archive.values():
        if isinstance(individual, dict):
            fitness = individual.get('fitness', 0)
            sparsity = individual.get('sparsity', 0)
            total_score = individual.get('total_score', 0)
            
            fitness_list.append(fitness)
            sparsity_list.append(sparsity)
            total_score_list.append(total_score)
    
    return {
        'fitness': np.array(fitness_list),
        'sparsity': np.array(sparsity_list),
        'total_score': np.array(total_score_list),
    }


def plot_comparison(baseline_data, sparsity_data, output_path):
    """绘制对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Baseline vs Sparsity-Aware Checkpoint Comparison', fontsize=16, fontweight='bold')
    
    # ========== 子图1: Fitness Evolution ==========
    ax1 = axes[0, 0]
    
    if len(baseline_data['forward_passes']) > 0:
        ax1.plot(baseline_data['forward_passes'], baseline_data['max_fitness'], 
                 'b-', linewidth=2, label='Baseline Max', alpha=0.8)
        ax1.plot(baseline_data['forward_passes'], baseline_data['mean_fitness'], 
                 'b--', linewidth=1.5, label='Baseline Mean', alpha=0.6)
    
    if len(sparsity_data['forward_passes']) > 0:
        ax1.plot(sparsity_data['forward_passes'], sparsity_data['max_fitness'], 
                 'r-', linewidth=2, label='Sparsity-Aware Max', alpha=0.8)
        ax1.plot(sparsity_data['forward_passes'], sparsity_data['mean_fitness'], 
                 'r--', linewidth=1.5, label='Sparsity-Aware Mean', alpha=0.6)
    
    ax1.set_xlabel('Forward Passes', fontsize=12)
    ax1.set_ylabel('Fitness (GSM8K Accuracy)', fontsize=12)
    ax1.set_title('Fitness Evolution', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # ========== 子图2: Sparsity Evolution ==========
    ax2 = axes[0, 1]
    
    if len(baseline_data['forward_passes']) > 0:
        ax2.plot(baseline_data['forward_passes'], baseline_data['max_sparsity'], 
                 'b-', linewidth=2, label='Baseline Max', alpha=0.8)
        ax2.plot(baseline_data['forward_passes'], baseline_data['mean_sparsity'], 
                 'b--', linewidth=1.5, label='Baseline Mean', alpha=0.6)
    
    if len(sparsity_data['forward_passes']) > 0:
        ax2.plot(sparsity_data['forward_passes'], sparsity_data['max_sparsity'], 
                 'r-', linewidth=2, label='Sparsity-Aware Max', alpha=0.8)
        ax2.plot(sparsity_data['forward_passes'], sparsity_data['mean_sparsity'], 
                 'r--', linewidth=1.5, label='Sparsity-Aware Mean', alpha=0.6)
    
    ax2.set_xlabel('Forward Passes', fontsize=12)
    ax2.set_ylabel('Sparsity', fontsize=12)
    ax2.set_title('Sparsity Evolution', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # ========== 子图3: Total Score Evolution ==========
    ax3 = axes[1, 0]
    
    if len(baseline_data['forward_passes']) > 0:
        ax3.plot(baseline_data['forward_passes'], baseline_data['max_total_score'], 
                 'b-', linewidth=2, label='Baseline Max', alpha=0.8)
        ax3.plot(baseline_data['forward_passes'], baseline_data['mean_total_score'], 
                 'b--', linewidth=1.5, label='Baseline Mean', alpha=0.6)
    
    if len(sparsity_data['forward_passes']) > 0:
        ax3.plot(sparsity_data['forward_passes'], sparsity_data['max_total_score'], 
                 'r-', linewidth=2, label='Sparsity-Aware Max', alpha=0.8)
        ax3.plot(sparsity_data['forward_passes'], sparsity_data['mean_total_score'], 
                 'r--', linewidth=1.5, label='Sparsity-Aware Mean', alpha=0.6)
    
    ax3.set_xlabel('Forward Passes', fontsize=12)
    ax3.set_ylabel('Total Score (ω·fitness + β·sparsity)', fontsize=12)
    ax3.set_title('Total Score Evolution', fontsize=14, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # ========== 子图4: Pareto Front (Archive) ==========
    ax4 = axes[1, 1]
    
    # 绘制baseline的archive
    if baseline_data.get('archive') is not None and len(baseline_data['archive']['fitness']) > 0:
        ax4.scatter(baseline_data['archive']['sparsity'], 
                   baseline_data['archive']['fitness'],
                   c='blue', s=50, alpha=0.6, label='Baseline Archive', marker='o')
    
    # 绘制sparsity-aware的archive
    if sparsity_data.get('archive') is not None and len(sparsity_data['archive']['fitness']) > 0:
        ax4.scatter(sparsity_data['archive']['sparsity'], 
                   sparsity_data['archive']['fitness'],
                   c='red', s=50, alpha=0.6, label='Sparsity-Aware Archive', marker='^')
    
    ax4.set_xlabel('Sparsity', fontsize=12)
    ax4.set_ylabel('Fitness (GSM8K Accuracy)', fontsize=12)
    ax4.set_title('Final Archive Distribution (Pareto Front)', fontsize=14, fontweight='bold')
    ax4.legend(loc='best', fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {output_path}")
    
    # 显示统计摘要
    print("\n" + "="*60)
    print("FINAL STATISTICS SUMMARY")
    print("="*60)
    
    if len(baseline_data['forward_passes']) > 0:
        print(f"\n📊 Baseline (Final):")
        print(f"  Max Fitness:      {baseline_data['max_fitness'][-1]:.4f}")
        print(f"  Mean Fitness:     {baseline_data['mean_fitness'][-1]:.4f}")
        print(f"  Max Sparsity:     {baseline_data['max_sparsity'][-1]:.4f}")
        print(f"  Mean Sparsity:    {baseline_data['mean_sparsity'][-1]:.4f}")
        print(f"  Max Total Score:  {baseline_data['max_total_score'][-1]:.4f}")
        print(f"  Mean Total Score: {baseline_data['mean_total_score'][-1]:.4f}")
    
    if len(sparsity_data['forward_passes']) > 0:
        print(f"\n📊 Sparsity-Aware (Final):")
        print(f"  Max Fitness:      {sparsity_data['max_fitness'][-1]:.4f}")
        print(f"  Mean Fitness:     {sparsity_data['mean_fitness'][-1]:.4f}")
        print(f"  Max Sparsity:     {sparsity_data['max_sparsity'][-1]:.4f}")
        print(f"  Mean Sparsity:    {sparsity_data['mean_sparsity'][-1]:.4f}")
        print(f"  Max Total Score:  {sparsity_data['max_total_score'][-1]:.4f}")
        print(f"  Mean Total Score: {sparsity_data['mean_total_score'][-1]:.4f}")
    
    # 计算改进百分比
    if len(baseline_data['forward_passes']) > 0 and len(sparsity_data['forward_passes']) > 0:
        fitness_improvement = (sparsity_data['max_fitness'][-1] - baseline_data['max_fitness'][-1]) / baseline_data['max_fitness'][-1] * 100
        sparsity_improvement = (sparsity_data['max_sparsity'][-1] - baseline_data['max_sparsity'][-1]) / max(baseline_data['max_sparsity'][-1], 1e-6) * 100
        total_improvement = (sparsity_data['max_total_score'][-1] - baseline_data['max_total_score'][-1]) / baseline_data['max_total_score'][-1] * 100
        
        print(f"\n📈 Improvement (Sparsity-Aware vs Baseline):")
        print(f"  Fitness:      {fitness_improvement:+.2f}%")
        print(f"  Sparsity:     {sparsity_improvement:+.2f}%")
        print(f"  Total Score:  {total_improvement:+.2f}%")
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Compare two checkpoints and visualize results')
    parser.add_argument('--baseline', type=str, required=True,
                       help='Path to baseline checkpoint (.pkl)')
    parser.add_argument('--sparsity', type=str, required=True,
                       help='Path to sparsity-aware checkpoint (.pkl)')
    parser.add_argument('--output', type=str, default='checkpoint_comparison.png',
                       help='Output plot filename (default: checkpoint_comparison.png)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("CHECKPOINT COMPARISON TOOL")
    print("="*60)
    
    # 加载checkpoints
    baseline_checkpoint = load_checkpoint(args.baseline)
    sparsity_checkpoint = load_checkpoint(args.sparsity)
    
    # 提取历史数据
    print("\nExtracting history data...")
    baseline_history = extract_history(baseline_checkpoint)
    sparsity_history = extract_history(sparsity_checkpoint)
    
    # 提取archive数据
    print("Extracting archive data...")
    baseline_archive = extract_archive_data(baseline_checkpoint)
    sparsity_archive = extract_archive_data(sparsity_checkpoint)
    
    # 合并数据
    baseline_data = {**baseline_history, 'archive': baseline_archive}
    sparsity_data = {**sparsity_history, 'archive': sparsity_archive}
    
    # 绘制对比图
    print(f"\nGenerating comparison plot...")
    plot_comparison(baseline_data, sparsity_data, args.output)


if __name__ == '__main__':
    main()

