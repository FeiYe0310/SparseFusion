#!/usr/bin/env python3
"""
分析并可视化SparseFusion实验结果

使用方法:
    python analyze_results.py results/sparsity_aware_w0.50_b0.50_t1.00_prune_wanda_0.20.pkl
"""

import pickle
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_results(file_path):
    """加载结果文件"""
    file_path = Path(file_path)
    
    if file_path.suffix == '.pkl':
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    elif file_path.suffix == '.json':
        with open(file_path, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f"不支持的文件格式: {file_path.suffix}")


def plot_training_curves(results, save_dir=None):
    """绘制训练曲线"""
    n_runs = len(results)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('SparseFusion Training Curves', fontsize=16)
    
    for run_idx in range(n_runs):
        run_data = results[run_idx]
        if "iterations" not in run_data:
            print(f"警告: Run {run_idx} 没有迭代数据")
            continue
            
        iterations = [item["iteration"] for item in run_data["iterations"]]
        
        # Plot 1: Archive Fitness
        archive_fitness_mean = [item["archive_fitness_mean"] for item in run_data["iterations"]]
        archive_fitness_max = [item["archive_fitness_max"] for item in run_data["iterations"]]
        
        axes[0, 0].plot(iterations, archive_fitness_mean, label=f'Run {run_idx+1} (mean)', alpha=0.7)
        axes[0, 0].plot(iterations, archive_fitness_max, label=f'Run {run_idx+1} (max)', alpha=0.7, linestyle='--')
    
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Fitness')
    axes[0, 0].set_title('Archive Fitness Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    for run_idx in range(n_runs):
        run_data = results[run_idx]
        if "iterations" not in run_data:
            continue
            
        iterations = [item["iteration"] for item in run_data["iterations"]]
        
        # Plot 2: Archive Sparsity
        archive_sparsity_mean = [item["archive_sparsity_mean"] for item in run_data["iterations"]]
        archive_sparsity_min = [item["archive_sparsity_min"] for item in run_data["iterations"]]
        
        axes[0, 1].plot(iterations, archive_sparsity_mean, label=f'Run {run_idx+1} (mean)', alpha=0.7)
        axes[0, 1].plot(iterations, archive_sparsity_min, label=f'Run {run_idx+1} (min)', alpha=0.7, linestyle='--')
    
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Sparsity')
    axes[0, 1].set_title('Archive Sparsity Over Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    for run_idx in range(n_runs):
        run_data = results[run_idx]
        if "iterations" not in run_data:
            continue
            
        iterations = [item["iteration"] for item in run_data["iterations"]]
        
        # Plot 3: Archive Total Score
        archive_total_score_mean = [item["archive_total_score_mean"] for item in run_data["iterations"]]
        archive_total_score_max = [item["archive_total_score_max"] for item in run_data["iterations"]]
        
        axes[1, 0].plot(iterations, archive_total_score_mean, label=f'Run {run_idx+1} (mean)', alpha=0.7)
        axes[1, 0].plot(iterations, archive_total_score_max, label=f'Run {run_idx+1} (max)', alpha=0.7, linestyle='--')
    
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Total Score')
    axes[1, 0].set_title('Archive Total Score Over Time')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Test Accuracy (每10步评估)
    for run_idx in range(n_runs):
        run_data = results[run_idx]
        if "test_evaluations" not in run_data:
            continue
            
        # 计算每次测试时archive的平均准确率
        test_iterations = sorted(set(item["iteration"] for item in run_data["test_evaluations"]))
        test_acc_means = []
        for iter_num in test_iterations:
            iter_accs = [item["test_accuracy"] for item in run_data["test_evaluations"] 
                        if item["iteration"] == iter_num]
            test_acc_means.append(np.mean(iter_accs))
        
        axes[1, 1].plot(test_iterations, test_acc_means, label=f'Run {run_idx+1}', alpha=0.7, marker='o')
    
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Test Accuracy')
    axes[1, 1].set_title('Test Accuracy (evaluated every 10 steps)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / "training_curves.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 图表已保存到: {save_path}")
    
    plt.show()


def print_summary(results):
    """打印结果摘要"""
    print("\n" + "=" * 70)
    print("实验结果摘要")
    print("=" * 70)
    
    for run_idx in range(len(results)):
        run_data = results[run_idx]
        print(f"\n--- Run {run_idx + 1} ---")
        
        if "final_best_model" in run_data:
            best = run_data["final_best_model"]
            print(f"最佳模型:")
            print(f"  - Fitness: {best['fitness']:.4f}")
            print(f"  - Sparsity: {best['sparsity']:.4f}")
            print(f"  - Total Score: {best['total_score']:.4f}")
            print(f"  - 保存路径: {best['save_path']}")
        
        if "iterations" in run_data and len(run_data["iterations"]) > 0:
            n_iters = len(run_data["iterations"])
            final_iter = run_data["iterations"][-1]
            print(f"\n最终迭代 (第{n_iters}步):")
            print(f"  - Archive Fitness (mean): {final_iter['archive_fitness_mean']:.4f}")
            print(f"  - Archive Fitness (max): {final_iter['archive_fitness_max']:.4f}")
            print(f"  - Archive Sparsity (mean): {final_iter['archive_sparsity_mean']:.4f}")
            print(f"  - Archive Sparsity (min): {final_iter['archive_sparsity_min']:.4f}")
        
        if "test_evaluations" in run_data and len(run_data["test_evaluations"]) > 0:
            test_accs = [item["test_accuracy"] for item in run_data["test_evaluations"]]
            print(f"\n测试集表现:")
            print(f"  - 平均准确率: {np.mean(test_accs):.4f}")
            print(f"  - 最佳准确率: {np.max(test_accs):.4f}")
            print(f"  - 评估次数: {len(set(item['iteration'] for item in run_data['test_evaluations']))}")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description='分析SparseFusion实验结果')
    parser.add_argument('result_file', type=str, help='结果文件路径 (.pkl 或 .json)')
    parser.add_argument('--no-plot', action='store_true', help='不绘制图表')
    parser.add_argument('--save-dir', type=str, default=None, help='保存图表的目录')
    
    args = parser.parse_args()
    
    # 加载结果
    print(f"加载结果文件: {args.result_file}")
    results = load_results(args.result_file)
    
    # 打印摘要
    print_summary(results)
    
    # 绘制图表
    if not args.no_plot:
        plot_training_curves(results, save_dir=args.save_dir)


if __name__ == "__main__":
    main()

