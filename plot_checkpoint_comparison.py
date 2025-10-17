#!/usr/bin/env python3
"""
Checkpoint Comparison Visualization Script

比较两个checkpoint的性能：
1. Baseline checkpoint (无sparsity-aware)
2. Sparsity-aware checkpoint (带剪枝)

绘制：
- Fitness (test_accuracy) evolution over iterations
- Sparsity evolution over iterations (if available)
- Pareto front comparison
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from collections import defaultdict


def load_checkpoint(checkpoint_path):
    """加载checkpoint文件"""
    print(f"Loading checkpoint: {checkpoint_path}")
    with open(checkpoint_path, 'rb') as f:
        data = pickle.load(f)
    
    # 打印checkpoint信息
    if isinstance(data, dict):
        print(f"  Type: dict")
        print(f"  Keys: {list(data.keys())}")
    elif isinstance(data, list):
        print(f"  Type: list")
        print(f"  Length: {len(data)}")
    else:
        print(f"  Type: {type(data)}")
    
    return data


def extract_history_from_results(data):
    """
    从results字段提取历史数据
    
    data可能是：
    1. dict with 'results' key containing list of records
    2. list of records directly
    """
    records = []
    
    if isinstance(data, dict):
        if 'results' in data:
            results_data = data['results']
            if isinstance(results_data, dict) and 'evaluation_history' in results_data:
                records = results_data['evaluation_history']
            elif isinstance(results_data, list):
                records = results_data
        elif 'evaluation_history' in data:
            records = data['evaluation_history']
    elif isinstance(data, list):
        records = data
    
    # 按iteration分组统计
    iteration_data = defaultdict(lambda: {'fitness': [], 'sparsity': []})
    
    for record in records:
        if not isinstance(record, dict):
            continue
        
        iteration = record.get('iteration', 0)
        fitness = record.get('test_accuracy', 0)
        sparsity = record.get('sparsity', None)
        
        iteration_data[iteration]['fitness'].append(fitness)
        if sparsity is not None:
            iteration_data[iteration]['sparsity'].append(sparsity)
    
    # 转换为数组
    iterations = sorted(iteration_data.keys())
    
    max_fitness = []
    mean_fitness = []
    max_sparsity = []
    mean_sparsity = []
    
    for it in iterations:
        fitness_vals = iteration_data[it]['fitness']
        sparsity_vals = iteration_data[it]['sparsity']
        
        if fitness_vals:
            max_fitness.append(np.max(fitness_vals))
            mean_fitness.append(np.mean(fitness_vals))
        else:
            max_fitness.append(0)
            mean_fitness.append(0)
        
        if sparsity_vals:
            max_sparsity.append(np.max(sparsity_vals))
            mean_sparsity.append(np.mean(sparsity_vals))
        else:
            max_sparsity.append(0)
            mean_sparsity.append(0)
    
    return {
        'iterations': np.array(iterations),
        'max_fitness': np.array(max_fitness),
        'mean_fitness': np.array(mean_fitness),
        'max_sparsity': np.array(max_sparsity),
        'mean_sparsity': np.array(mean_sparsity),
    }


def extract_final_population(data):
    """提取最终population的fitness和sparsity"""
    records = []
    
    if isinstance(data, dict):
        if 'results' in data:
            results_data = data['results']
            if isinstance(results_data, dict) and 'evaluation_history' in results_data:
                records = results_data['evaluation_history']
            elif isinstance(results_data, list):
                records = results_data
        elif 'evaluation_history' in data:
            records = data['evaluation_history']
    elif isinstance(data, list):
        records = data
    
    if not records:
        return {'fitness': np.array([]), 'sparsity': np.array([])}
    
    # 获取最后一个iteration的所有个体
    max_iteration = max(r.get('iteration', 0) for r in records if isinstance(r, dict))
    
    final_fitness = []
    final_sparsity = []
    
    for record in records:
        if not isinstance(record, dict):
            continue
        if record.get('iteration', 0) == max_iteration:
            fitness = record.get('test_accuracy', 0)
            sparsity = record.get('sparsity', None)
            
            final_fitness.append(fitness)
            if sparsity is not None:
                final_sparsity.append(sparsity)
            else:
                final_sparsity.append(0)
    
    return {
        'fitness': np.array(final_fitness),
        'sparsity': np.array(final_sparsity),
    }


def plot_comparison(baseline_data, sparsity_data, output_path):
    """绘制对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Baseline vs Sparsity-Aware Checkpoint Comparison', fontsize=16, fontweight='bold')
    
    # ========== 子图1: Fitness Evolution ==========
    ax1 = axes[0, 0]
    
    if len(baseline_data['iterations']) > 0:
        ax1.plot(baseline_data['iterations'], baseline_data['max_fitness'], 
                 'b-', linewidth=2, label='Baseline Max', alpha=0.8)
        ax1.plot(baseline_data['iterations'], baseline_data['mean_fitness'], 
                 'b--', linewidth=1.5, label='Baseline Mean', alpha=0.6)
    
    if len(sparsity_data['iterations']) > 0:
        ax1.plot(sparsity_data['iterations'], sparsity_data['max_fitness'], 
                 'r-', linewidth=2, label='Sparsity-Aware Max', alpha=0.8)
        ax1.plot(sparsity_data['iterations'], sparsity_data['mean_fitness'], 
                 'r--', linewidth=1.5, label='Sparsity-Aware Mean', alpha=0.6)
    
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Fitness (Test Accuracy)', fontsize=12)
    ax1.set_title('Fitness Evolution', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # ========== 子图2: Sparsity Evolution ==========
    ax2 = axes[0, 1]
    
    has_sparsity_data = False
    
    if len(baseline_data['iterations']) > 0 and np.any(baseline_data['max_sparsity'] > 0):
        ax2.plot(baseline_data['iterations'], baseline_data['max_sparsity'], 
                 'b-', linewidth=2, label='Baseline Max', alpha=0.8)
        ax2.plot(baseline_data['iterations'], baseline_data['mean_sparsity'], 
                 'b--', linewidth=1.5, label='Baseline Mean', alpha=0.6)
        has_sparsity_data = True
    
    if len(sparsity_data['iterations']) > 0 and np.any(sparsity_data['max_sparsity'] > 0):
        ax2.plot(sparsity_data['iterations'], sparsity_data['max_sparsity'], 
                 'r-', linewidth=2, label='Sparsity-Aware Max', alpha=0.8)
        ax2.plot(sparsity_data['iterations'], sparsity_data['mean_sparsity'], 
                 'r--', linewidth=1.5, label='Sparsity-Aware Mean', alpha=0.6)
        has_sparsity_data = True
    
    if not has_sparsity_data:
        ax2.text(0.5, 0.5, 'No Sparsity Data Available', 
                ha='center', va='center', fontsize=14, transform=ax2.transAxes)
    
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Sparsity', fontsize=12)
    ax2.set_title('Sparsity Evolution', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # ========== 子图3: Max Fitness Comparison ==========
    ax3 = axes[1, 0]
    
    if len(baseline_data['iterations']) > 0 and len(sparsity_data['iterations']) > 0:
        x_labels = ['Baseline', 'Sparsity-Aware']
        max_vals = [baseline_data['max_fitness'][-1], sparsity_data['max_fitness'][-1]]
        colors = ['blue', 'red']
        
        bars = ax3.bar(x_labels, max_vals, color=colors, alpha=0.7, edgecolor='black')
        
        # 添加数值标签
        for bar, val in zip(bars, max_vals):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax3.set_ylabel('Max Fitness', fontsize=12)
        ax3.set_title('Final Max Fitness Comparison', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
    
    # ========== 子图4: Final Population Distribution ==========
    ax4 = axes[1, 1]
    
    baseline_pop = baseline_data.get('final_population', {'fitness': np.array([]), 'sparsity': np.array([])})
    sparsity_pop = sparsity_data.get('final_population', {'fitness': np.array([]), 'sparsity': np.array([])})
    
    has_pop_data = False
    
    if len(baseline_pop['fitness']) > 0:
        # 如果sparsity全为0，使用个体编号代替
        x_vals = baseline_pop['sparsity'] if np.any(baseline_pop['sparsity'] > 0) else np.arange(len(baseline_pop['fitness']))
        ax4.scatter(x_vals, baseline_pop['fitness'],
                   c='blue', s=100, alpha=0.6, label='Baseline', marker='o', edgecolors='black')
        has_pop_data = True
    
    if len(sparsity_pop['fitness']) > 0:
        x_vals = sparsity_pop['sparsity'] if np.any(sparsity_pop['sparsity'] > 0) else np.arange(len(sparsity_pop['fitness']))
        ax4.scatter(x_vals, sparsity_pop['fitness'],
                   c='red', s=100, alpha=0.6, label='Sparsity-Aware', marker='^', edgecolors='black')
        has_pop_data = True
    
    if has_pop_data:
        # 判断x轴标签
        if np.any(baseline_pop.get('sparsity', []) > 0) or np.any(sparsity_pop.get('sparsity', []) > 0):
            ax4.set_xlabel('Sparsity', fontsize=12)
        else:
            ax4.set_xlabel('Individual Index', fontsize=12)
        
        ax4.set_ylabel('Fitness (Test Accuracy)', fontsize=12)
        ax4.set_title('Final Population Distribution', fontsize=14, fontweight='bold')
        ax4.legend(loc='best', fontsize=10)
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No Population Data', 
                ha='center', va='center', fontsize=14, transform=ax4.transAxes)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {output_path}")
    
    # 显示统计摘要
    print("\n" + "="*60)
    print("FINAL STATISTICS SUMMARY")
    print("="*60)
    
    if len(baseline_data['iterations']) > 0:
        print(f"\n📊 Baseline (Final):")
        print(f"  Iteration:        {baseline_data['iterations'][-1]}")
        print(f"  Max Fitness:      {baseline_data['max_fitness'][-1]:.4f}")
        print(f"  Mean Fitness:     {baseline_data['mean_fitness'][-1]:.4f}")
        if np.any(baseline_data['max_sparsity'] > 0):
            print(f"  Max Sparsity:     {baseline_data['max_sparsity'][-1]:.6f}")
            print(f"  Mean Sparsity:    {baseline_data['mean_sparsity'][-1]:.6f}")
    
    if len(sparsity_data['iterations']) > 0:
        print(f"\n📊 Sparsity-Aware (Final):")
        print(f"  Iteration:        {sparsity_data['iterations'][-1]}")
        print(f"  Max Fitness:      {sparsity_data['max_fitness'][-1]:.4f}")
        print(f"  Mean Fitness:     {sparsity_data['mean_fitness'][-1]:.4f}")
        if np.any(sparsity_data['max_sparsity'] > 0):
            print(f"  Max Sparsity:     {sparsity_data['max_sparsity'][-1]:.6f}")
            print(f"  Mean Sparsity:    {sparsity_data['mean_sparsity'][-1]:.6f}")
    
    # 计算改进百分比
    if len(baseline_data['iterations']) > 0 and len(sparsity_data['iterations']) > 0:
        fitness_improvement = (sparsity_data['max_fitness'][-1] - baseline_data['max_fitness'][-1]) / baseline_data['max_fitness'][-1] * 100
        
        print(f"\n📈 Improvement (Sparsity-Aware vs Baseline):")
        print(f"  Max Fitness:  {fitness_improvement:+.2f}%")
        
        if np.any(baseline_data['max_sparsity'] > 0) and np.any(sparsity_data['max_sparsity'] > 0):
            sparsity_improvement = (sparsity_data['max_sparsity'][-1] - baseline_data['max_sparsity'][-1]) / max(baseline_data['max_sparsity'][-1], 1e-9) * 100
            print(f"  Max Sparsity: {sparsity_improvement:+.2f}%")
    
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
    baseline_history = extract_history_from_results(baseline_checkpoint)
    sparsity_history = extract_history_from_results(sparsity_checkpoint)
    
    # 提取最终population
    print("Extracting final population...")
    baseline_population = extract_final_population(baseline_checkpoint)
    sparsity_population = extract_final_population(sparsity_checkpoint)
    
    # 合并数据
    baseline_data = {**baseline_history, 'final_population': baseline_population}
    sparsity_data = {**sparsity_history, 'final_population': sparsity_population}
    
    # 绘制对比图
    print(f"\nGenerating comparison plot...")
    plot_comparison(baseline_data, sparsity_data, args.output)


if __name__ == '__main__':
    main()
