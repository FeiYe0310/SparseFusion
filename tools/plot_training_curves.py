#!/usr/bin/env python3
"""
实时画图工具 - 可以在训练过程中查看进度

用法:
    # 画最新的checkpoint
    python tools/plot_training_curves.py
    
    # 画指定的checkpoint
    python tools/plot_training_curves.py results/checkpoint_run1_20231016.pkl
    
    # 持续监控（每30秒刷新）
    python tools/plot_training_curves.py --watch
"""

import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import glob
import os
import time
from pathlib import Path

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)


def find_latest_checkpoint(results_dir="results"):
    """查找最新的checkpoint文件"""
    patterns = [
        f"{results_dir}*/checkpoint*.pkl",
        f"{results_dir}/checkpoint*.pkl",
        "checkpoint*.pkl"
    ]
    
    all_files = []
    for pattern in patterns:
        all_files.extend(glob.glob(pattern))
    
    if not all_files:
        return None
    
    # 按修改时间排序，返回最新的
    latest = max(all_files, key=os.path.getmtime)
    return latest


def load_checkpoint(checkpoint_path):
    """加载checkpoint文件"""
    try:
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"❌ 无法加载 {checkpoint_path}: {e}")
        return None


def plot_training_curves(data, save_path=None):
    """画训练曲线"""
    results = data.get('results', [])
    
    if not results or not results[0].get('iterations'):
        print("⚠️ 没有找到迭代数据")
        return
    
    # 提取数据
    all_runs_data = []
    for run_idx, run_data in enumerate(results):
        iterations = run_data.get('iterations', [])
        if not iterations:
            continue
        
        run_stats = {
            'iteration': [it['iteration'] for it in iterations],
            'child_fitness': [it['child_fitness'] for it in iterations],
            'child_sparsity': [it['child_sparsity'] for it in iterations],
            'archive_fitness_mean': [it['archive_fitness_mean'] for it in iterations],
            'archive_fitness_max': [it['archive_fitness_max'] for it in iterations],
            'archive_sparsity_mean': [it['archive_sparsity_mean'] for it in iterations],
            'archive_sparsity_min': [it['archive_sparsity_min'] for it in iterations],
            'archive_total_score_mean': [it['archive_total_score_mean'] for it in iterations],
            'archive_total_score_max': [it['archive_total_score_max'] for it in iterations],
        }
        all_runs_data.append(run_stats)
    
    if not all_runs_data:
        print("⚠️ 没有有效的run数据")
        return
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('🔬 Sparsity-Aware Training Progress', fontsize=16, fontweight='bold')
    
    # 1. Fitness曲线
    ax = axes[0, 0]
    for i, run_data in enumerate(all_runs_data):
        ax.plot(run_data['iteration'], run_data['archive_fitness_max'], 
                label=f'Run {i+1} (Max)', linewidth=2, alpha=0.8)
        ax.plot(run_data['iteration'], run_data['archive_fitness_mean'], 
                label=f'Run {i+1} (Mean)', linewidth=1.5, alpha=0.6, linestyle='--')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Fitness (Accuracy)', fontsize=12)
    ax.set_title('📈 Fitness Evolution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Sparsity曲线
    ax = axes[0, 1]
    for i, run_data in enumerate(all_runs_data):
        ax.plot(run_data['iteration'], run_data['archive_sparsity_min'], 
                label=f'Run {i+1} (Min)', linewidth=2, alpha=0.8)
        ax.plot(run_data['iteration'], run_data['archive_sparsity_mean'], 
                label=f'Run {i+1} (Mean)', linewidth=1.5, alpha=0.6, linestyle='--')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Sparsity', fontsize=12)
    ax.set_title('🎯 Sparsity Evolution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Total Score曲线
    ax = axes[1, 0]
    for i, run_data in enumerate(all_runs_data):
        ax.plot(run_data['iteration'], run_data['archive_total_score_max'], 
                label=f'Run {i+1} (Max)', linewidth=2, alpha=0.8)
        ax.plot(run_data['iteration'], run_data['archive_total_score_mean'], 
                label=f'Run {i+1} (Mean)', linewidth=1.5, alpha=0.6, linestyle='--')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Total Score (ω×Fitness + β×Sparsity)', fontsize=12)
    ax.set_title('🏆 Total Score Evolution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Trade-off散点图（最新状态）
    ax = axes[1, 1]
    for i, run_data in enumerate(all_runs_data):
        if len(run_data['iteration']) > 0:
            # 画最后10个点的轨迹
            n_points = min(10, len(run_data['iteration']))
            last_fitness = run_data['archive_fitness_max'][-n_points:]
            last_sparsity = run_data['archive_sparsity_min'][-n_points:]
            
            # 渐变颜色表示时间
            colors = plt.cm.viridis(np.linspace(0, 1, n_points))
            ax.scatter(last_sparsity, last_fitness, c=colors, s=100, 
                      alpha=0.7, edgecolors='black', linewidths=1.5,
                      label=f'Run {i+1}')
            
            # 最新点标记
            ax.scatter(last_sparsity[-1], last_fitness[-1], 
                      marker='*', s=400, edgecolors='red', linewidths=2,
                      c=colors[-1], zorder=10)
    
    ax.set_xlabel('Sparsity (Lower = Denser)', fontsize=12)
    ax.set_ylabel('Fitness (Accuracy)', fontsize=12)
    ax.set_title('🎨 Fitness-Sparsity Trade-off (Last 10 iters)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ 图表已保存到: {save_path}")
    else:
        plt.show()
    
    # 打印统计信息
    print("\n" + "="*60)
    print("📊 训练统计")
    print("="*60)
    for i, run_data in enumerate(all_runs_data):
        n_iters = len(run_data['iteration'])
        latest_fitness = run_data['archive_fitness_max'][-1]
        latest_sparsity = run_data['archive_sparsity_min'][-1]
        latest_total = run_data['archive_total_score_max'][-1]
        
        print(f"\nRun {i+1}:")
        print(f"  - 已完成迭代: {n_iters}")
        print(f"  - 当前最佳Fitness: {latest_fitness:.4f}")
        print(f"  - 当前最低Sparsity: {latest_sparsity:.4f}")
        print(f"  - 当前最高Total Score: {latest_total:.4f}")
    print("="*60)


def watch_and_plot(checkpoint_path=None, interval=30):
    """持续监控并更新图表"""
    print("🔄 进入监控模式（Ctrl+C退出）")
    print(f"   刷新间隔: {interval}秒")
    
    try:
        while True:
            # 查找checkpoint
            if checkpoint_path is None:
                path = find_latest_checkpoint()
            else:
                path = checkpoint_path
            
            if path is None:
                print(f"⏳ 等待checkpoint文件出现... (搜索路径: results*/checkpoint*.pkl)")
                time.sleep(interval)
                continue
            
            # 获取文件修改时间
            mtime = os.path.getmtime(path)
            print(f"\n📂 加载: {path}")
            print(f"   最后修改: {time.ctime(mtime)}")
            
            # 加载并画图
            data = load_checkpoint(path)
            if data:
                # 生成保存路径
                save_path = path.replace('.pkl', '_plot.png')
                plot_training_curves(data, save_path=save_path)
            
            print(f"\n⏰ {interval}秒后刷新...")
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n👋 监控已停止")


def main():
    parser = argparse.ArgumentParser(
        description='实时画训练曲线',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 画最新checkpoint
  python tools/plot_training_curves.py
  
  # 画指定checkpoint
  python tools/plot_training_curves.py results/checkpoint_run1.pkl
  
  # 持续监控（每30秒刷新）
  python tools/plot_training_curves.py --watch
  
  # 自定义刷新间隔
  python tools/plot_training_curves.py --watch --interval 60
        """
    )
    
    parser.add_argument('checkpoint', nargs='?', default=None,
                       help='Checkpoint文件路径（不指定则自动查找最新的）')
    parser.add_argument('--watch', action='store_true',
                       help='持续监控模式')
    parser.add_argument('--interval', type=int, default=30,
                       help='监控模式的刷新间隔（秒）')
    parser.add_argument('--save', type=str, default=None,
                       help='保存图表到指定路径')
    
    args = parser.parse_args()
    
    # 监控模式
    if args.watch:
        watch_and_plot(args.checkpoint, args.interval)
        return
    
    # 单次画图
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint()
        if checkpoint_path is None:
            print("❌ 没有找到checkpoint文件")
            print("   搜索路径: results*/checkpoint*.pkl, checkpoint*.pkl")
            return
    
    print(f"📂 加载: {checkpoint_path}")
    data = load_checkpoint(checkpoint_path)
    
    if data:
        save_path = args.save
        if save_path is None and not args.watch:
            # 默认保存路径
            save_path = checkpoint_path.replace('.pkl', '_plot.png')
        
        plot_training_curves(data, save_path=save_path)


if __name__ == '__main__':
    main()


