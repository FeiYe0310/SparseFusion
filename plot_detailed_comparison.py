#!/usr/bin/env python3
"""
Detailed Checkpoint Comparison with Individual Trajectories

生成详细的对比图表，包括：
1. 每个individual的fitness演化轨迹
2. Population的max/mean/min统计
3. Individual级别的详细分析
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from collections import defaultdict


def load_and_extract_data(checkpoint_path):
    """加载checkpoint并提取数据"""
    print(f"Loading: {checkpoint_path}")
    with open(checkpoint_path, 'rb') as f:
        data = pickle.load(f)
    
    records = []
    
    # 提取records
    if isinstance(data, dict) and 'results' in data:
        results = data['results']
        if isinstance(results, list) and len(results) > 0:
            # Baseline格式: results[0]可能是dict with test_evaluations
            if isinstance(results[0], dict) and 'test_evaluations' in results[0]:
                records = results[0]['test_evaluations']
            else:
                # 直接是records list
                records = results
    elif isinstance(data, list) and len(data) > 0:
        first_elem = data[0]
        # Sparsity-aware格式: list[0]是defaultdict with test_evaluations
        if hasattr(first_elem, 'keys') and 'test_evaluations' in first_elem:
            records = first_elem['test_evaluations']
    
    print(f"  Found {len(records)} evaluation records")
    
    # 按iteration和individual组织数据
    iteration_data = defaultdict(dict)  # {iteration: {individual: fitness}}
    
    for record in records:
        if not isinstance(record, dict):
            continue
        
        iteration = record.get('iteration', 0)
        individual = record.get('individual', 0)
        fitness = record.get('test_accuracy', 0)
        
        iteration_data[iteration][individual] = fitness
    
    # 转换为数组格式
    iterations = sorted(iteration_data.keys())
    
    # 获取所有individual IDs
    all_individuals = set()
    for it_data in iteration_data.values():
        all_individuals.update(it_data.keys())
    all_individuals = sorted(all_individuals)
    
    # 构建每个individual的轨迹
    individual_trajectories = {ind: [] for ind in all_individuals}
    population_stats = {
        'iterations': [],
        'max_fitness': [],
        'mean_fitness': [],
        'min_fitness': []
    }
    
    for it in iterations:
        fitness_values = []
        for ind in all_individuals:
            fitness = iteration_data[it].get(ind, 0)
            individual_trajectories[ind].append(fitness)
            fitness_values.append(fitness)
        
        population_stats['iterations'].append(it)
        if fitness_values:
            population_stats['max_fitness'].append(np.max(fitness_values))
            population_stats['mean_fitness'].append(np.mean(fitness_values))
            population_stats['min_fitness'].append(np.min(fitness_values))
        else:
            population_stats['max_fitness'].append(0)
            population_stats['mean_fitness'].append(0)
            population_stats['min_fitness'].append(0)
    
    # 转换为numpy数组
    for key in population_stats:
        population_stats[key] = np.array(population_stats[key])
    
    for ind in individual_trajectories:
        individual_trajectories[ind] = np.array(individual_trajectories[ind])
    
    return {
        'iterations': iterations,
        'individuals': all_individuals,
        'trajectories': individual_trajectories,
        'stats': population_stats
    }


def plot_detailed_comparison(baseline_data, sparsity_data, output_path):
    """生成详细对比图"""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Detailed Baseline vs Sparsity-Aware Comparison', 
                 fontsize=16, fontweight='bold')
    
    # ========== 子图1: Baseline Individual Trajectories ==========
    ax1 = fig.add_subplot(gs[0, 0])
    if baseline_data and len(baseline_data['iterations']) > 0:
        for ind in baseline_data['individuals']:
            ax1.plot(baseline_data['iterations'], 
                    baseline_data['trajectories'][ind],
                    alpha=0.6, linewidth=1, label=f'Ind {ind}')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Fitness')
        ax1.set_title('Baseline: Individual Trajectories')
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'No Baseline Data', ha='center', va='center', 
                transform=ax1.transAxes)
    
    # ========== 子图2: Sparsity-Aware Individual Trajectories ==========
    ax2 = fig.add_subplot(gs[0, 1])
    if sparsity_data and len(sparsity_data['iterations']) > 0:
        for ind in sparsity_data['individuals']:
            ax2.plot(sparsity_data['iterations'], 
                    sparsity_data['trajectories'][ind],
                    alpha=0.6, linewidth=1, label=f'Ind {ind}')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Fitness')
        ax2.set_title('Sparsity-Aware: Individual Trajectories')
        ax2.legend(loc='best', fontsize=8)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No Sparsity Data', ha='center', va='center',
                transform=ax2.transAxes)
    
    # ========== 子图3: Population Statistics Comparison ==========
    ax3 = fig.add_subplot(gs[0, 2])
    if baseline_data and len(baseline_data['iterations']) > 0:
        ax3.plot(baseline_data['stats']['iterations'], 
                baseline_data['stats']['max_fitness'],
                'b-', linewidth=2, label='Baseline Max', alpha=0.8)
        ax3.plot(baseline_data['stats']['iterations'], 
                baseline_data['stats']['mean_fitness'],
                'b--', linewidth=1.5, label='Baseline Mean', alpha=0.6)
    
    if sparsity_data and len(sparsity_data['iterations']) > 0:
        ax3.plot(sparsity_data['stats']['iterations'], 
                sparsity_data['stats']['max_fitness'],
                'r-', linewidth=2, label='Sparsity Max', alpha=0.8)
        ax3.plot(sparsity_data['stats']['iterations'], 
                sparsity_data['stats']['mean_fitness'],
                'r--', linewidth=1.5, label='Sparsity Mean', alpha=0.6)
    
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Fitness')
    ax3.set_title('Population Statistics')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # ========== 子图4: Baseline Max/Mean/Min with Shading ==========
    ax4 = fig.add_subplot(gs[1, 0])
    if baseline_data and len(baseline_data['iterations']) > 0:
        stats = baseline_data['stats']
        ax4.plot(stats['iterations'], stats['max_fitness'], 
                'b-', linewidth=2, label='Max')
        ax4.plot(stats['iterations'], stats['mean_fitness'], 
                'g-', linewidth=2, label='Mean')
        ax4.plot(stats['iterations'], stats['min_fitness'], 
                'r-', linewidth=2, label='Min')
        ax4.fill_between(stats['iterations'], 
                        stats['min_fitness'], 
                        stats['max_fitness'],
                        alpha=0.2, color='blue')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Fitness')
        ax4.set_title('Baseline: Population Range')
        ax4.legend(loc='best')
        ax4.grid(True, alpha=0.3)
    
    # ========== 子图5: Sparsity-Aware Max/Mean/Min with Shading ==========
    ax5 = fig.add_subplot(gs[1, 1])
    if sparsity_data and len(sparsity_data['iterations']) > 0:
        stats = sparsity_data['stats']
        ax5.plot(stats['iterations'], stats['max_fitness'], 
                'b-', linewidth=2, label='Max')
        ax5.plot(stats['iterations'], stats['mean_fitness'], 
                'g-', linewidth=2, label='Mean')
        ax5.plot(stats['iterations'], stats['min_fitness'], 
                'r-', linewidth=2, label='Min')
        ax5.fill_between(stats['iterations'], 
                        stats['min_fitness'], 
                        stats['max_fitness'],
                        alpha=0.2, color='red')
        ax5.set_xlabel('Iteration')
        ax5.set_ylabel('Fitness')
        ax5.set_title('Sparsity-Aware: Population Range')
        ax5.legend(loc='best')
        ax5.grid(True, alpha=0.3)
    
    # ========== 子图6: Best Individual Comparison ==========
    ax6 = fig.add_subplot(gs[1, 2])
    if baseline_data and len(baseline_data['iterations']) > 0:
        ax6.plot(baseline_data['stats']['iterations'], 
                baseline_data['stats']['max_fitness'],
                'b-', linewidth=2, label='Baseline Best', alpha=0.8)
    
    if sparsity_data and len(sparsity_data['iterations']) > 0:
        ax6.plot(sparsity_data['stats']['iterations'], 
                sparsity_data['stats']['max_fitness'],
                'r-', linewidth=2, label='Sparsity Best', alpha=0.8)
    
    ax6.set_xlabel('Iteration')
    ax6.set_ylabel('Max Fitness')
    ax6.set_title('Best Individual Comparison')
    ax6.legend(loc='best')
    ax6.grid(True, alpha=0.3)
    
    # ========== 子图7: Final Population Distribution (Baseline) ==========
    ax7 = fig.add_subplot(gs[2, 0])
    if baseline_data and len(baseline_data['iterations']) > 0:
        final_fitness = [baseline_data['trajectories'][ind][-1] 
                        for ind in baseline_data['individuals']]
        ax7.bar(baseline_data['individuals'], final_fitness, 
               color='blue', alpha=0.7, edgecolor='black')
        ax7.set_xlabel('Individual ID')
        ax7.set_ylabel('Final Fitness')
        ax7.set_title('Baseline: Final Population')
        ax7.grid(True, alpha=0.3, axis='y')
    
    # ========== 子图8: Final Population Distribution (Sparsity) ==========
    ax8 = fig.add_subplot(gs[2, 1])
    if sparsity_data and len(sparsity_data['iterations']) > 0:
        final_fitness = [sparsity_data['trajectories'][ind][-1] 
                        for ind in sparsity_data['individuals']]
        ax8.bar(sparsity_data['individuals'], final_fitness, 
               color='red', alpha=0.7, edgecolor='black')
        ax8.set_xlabel('Individual ID')
        ax8.set_ylabel('Final Fitness')
        ax8.set_title('Sparsity-Aware: Final Population')
        ax8.grid(True, alpha=0.3, axis='y')
    
    # ========== 子图9: Statistics Table ==========
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    table_data = []
    table_data.append(['Metric', 'Baseline', 'Sparsity', 'Δ'])
    
    if baseline_data and sparsity_data:
        if len(baseline_data['iterations']) > 0 and len(sparsity_data['iterations']) > 0:
            b_max = baseline_data['stats']['max_fitness'][-1]
            s_max = sparsity_data['stats']['max_fitness'][-1]
            b_mean = baseline_data['stats']['mean_fitness'][-1]
            s_mean = sparsity_data['stats']['mean_fitness'][-1]
            
            table_data.append(['Max Fitness', 
                             f'{b_max:.4f}', 
                             f'{s_max:.4f}',
                             f'{s_max-b_max:+.4f}'])
            table_data.append(['Mean Fitness', 
                             f'{b_mean:.4f}', 
                             f'{s_mean:.4f}',
                             f'{s_mean-b_mean:+.4f}'])
            
            if b_max > 0:
                improvement = (s_max - b_max) / b_max * 100
                table_data.append(['Improvement', '', '', f'{improvement:+.2f}%'])
    
    table = ax9.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # 设置表头样式
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Detailed plot saved to: {output_path}")
    
    # 打印详细统计
    print("\n" + "="*70)
    print("DETAILED STATISTICS")
    print("="*70)
    
    if baseline_data and len(baseline_data['iterations']) > 0:
        print(f"\n📊 Baseline:")
        print(f"  Iterations: {len(baseline_data['iterations'])}")
        print(f"  Individuals: {len(baseline_data['individuals'])}")
        print(f"  Final Max Fitness: {baseline_data['stats']['max_fitness'][-1]:.4f}")
        print(f"  Final Mean Fitness: {baseline_data['stats']['mean_fitness'][-1]:.4f}")
        print(f"  Final Min Fitness: {baseline_data['stats']['min_fitness'][-1]:.4f}")
    
    if sparsity_data and len(sparsity_data['iterations']) > 0:
        print(f"\n📊 Sparsity-Aware:")
        print(f"  Iterations: {len(sparsity_data['iterations'])}")
        print(f"  Individuals: {len(sparsity_data['individuals'])}")
        print(f"  Final Max Fitness: {sparsity_data['stats']['max_fitness'][-1]:.4f}")
        print(f"  Final Mean Fitness: {sparsity_data['stats']['mean_fitness'][-1]:.4f}")
        print(f"  Final Min Fitness: {sparsity_data['stats']['min_fitness'][-1]:.4f}")
    
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Detailed checkpoint comparison')
    parser.add_argument('--baseline', type=str, required=True)
    parser.add_argument('--sparsity', type=str, required=True)
    parser.add_argument('--output', type=str, default='detailed_comparison.png')
    
    args = parser.parse_args()
    
    print("="*70)
    print("DETAILED CHECKPOINT COMPARISON")
    print("="*70)
    
    baseline_data = load_and_extract_data(args.baseline)
    sparsity_data = load_and_extract_data(args.sparsity)
    
    plot_detailed_comparison(baseline_data, sparsity_data, args.output)


if __name__ == '__main__':
    main()

