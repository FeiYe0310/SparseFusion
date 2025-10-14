#!/usr/bin/env python3
"""
实时绘制训练曲线 - 用于向老板汇报进展

使用方法:
  python plot_training_curves.py results/checkpoints/checkpoint_run1_stepXXX.pkl
  或
  python plot_training_curves.py --live  # 自动找最新checkpoint并持续更新
"""

import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import glob
import os
from pathlib import Path
import time

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def find_latest_checkpoint(checkpoint_dir="results/checkpoints"):
    """找到最新的checkpoint文件"""
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pkl"))
    if not checkpoints:
        return None
    
    # 按修改时间排序，返回最新的
    latest = max(checkpoints, key=os.path.getmtime)
    return latest

def load_checkpoint(checkpoint_path):
    """加载checkpoint数据"""
    with open(checkpoint_path, 'rb') as f:
        data = pickle.load(f)
    return data

def extract_training_data(checkpoint_data):
    """从checkpoint提取训练数据"""
    results = checkpoint_data.get("results", {})
    
    # 提取iteration数据
    iterations_data = []
    test_data = []
    
    for run_idx, run_data in results.items():
        if "iterations" in run_data:
            for iter_stat in run_data["iterations"]:
                iterations_data.append(iter_stat)
        
        if "test_evaluations" in run_data:
            for test_stat in run_data["test_evaluations"]:
                test_data.append(test_stat)
    
    return iterations_data, test_data

def plot_curves(iterations_data, test_data, save_path="training_curves.png", title_suffix=""):
    """绘制训练曲线"""
    
    if not iterations_data:
        print("❌ 没有训练数据，无法绘图")
        return False
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'SparseFusion Training Progress{title_suffix}', fontsize=20, fontweight='bold')
    
    # 提取数据
    steps = [d['iteration'] for d in iterations_data]
    
    # 1. Training Accuracy (Child Fitness)
    train_acc = [d.get('child_fitness', 0) for d in iterations_data]
    axes[0, 0].plot(steps, train_acc, 'b-', linewidth=2, label='Training Accuracy', alpha=0.7)
    if len(train_acc) > 10:
        # 添加移动平均线
        window = min(50, len(train_acc) // 5)
        if window > 1:
            smooth = np.convolve(train_acc, np.ones(window)/window, mode='valid')
            axes[0, 0].plot(steps[window-1:], smooth, 'r-', linewidth=2.5, label=f'{window}-step Moving Avg')
    axes[0, 0].set_xlabel('Training Steps', fontsize=12)
    axes[0, 0].set_ylabel('Accuracy', fontsize=12)
    axes[0, 0].set_title('Training Accuracy (Child Fitness)', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Archive Performance
    archive_mean = [d.get('archive_fitness_mean', 0) for d in iterations_data]
    archive_max = [d.get('archive_fitness_max', 0) for d in iterations_data]
    axes[0, 1].plot(steps, archive_mean, 'g-', linewidth=2, label='Archive Mean Fitness', alpha=0.7)
    axes[0, 1].plot(steps, archive_max, 'orange', linewidth=2, label='Archive Max Fitness', alpha=0.7)
    axes[0, 1].set_xlabel('Training Steps', fontsize=12)
    axes[0, 1].set_ylabel('Fitness', fontsize=12)
    axes[0, 1].set_title('Archive Performance', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Sparsity Evolution
    child_sparsity = [d.get('child_sparsity', 0) for d in iterations_data]
    archive_sparsity = [d.get('archive_sparsity_mean', 0) for d in iterations_data]
    axes[1, 0].plot(steps, child_sparsity, 'm-', linewidth=2, label='Child Sparsity', alpha=0.7)
    axes[1, 0].plot(steps, archive_sparsity, 'c-', linewidth=2, label='Archive Mean Sparsity', alpha=0.7)
    axes[1, 0].set_xlabel('Training Steps', fontsize=12)
    axes[1, 0].set_ylabel('Sparsity Ratio', fontsize=12)
    axes[1, 0].set_title('Model Sparsity Evolution', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Test/Validation Accuracy
    if test_data:
        # 按iteration分组，计算平均
        test_by_iter = {}
        for t in test_data:
            iter_num = t['iteration']
            if iter_num not in test_by_iter:
                test_by_iter[iter_num] = []
            test_by_iter[iter_num].append(t['test_accuracy'])
        
        test_steps = sorted(test_by_iter.keys())
        test_mean = [np.mean(test_by_iter[s]) for s in test_steps]
        test_max = [np.max(test_by_iter[s]) for s in test_steps]
        
        axes[1, 1].plot(test_steps, test_mean, 'r-', linewidth=2, marker='o', 
                       markersize=6, label='Test Mean Acc', alpha=0.7)
        axes[1, 1].plot(test_steps, test_max, 'darkred', linewidth=2, marker='s', 
                       markersize=6, label='Test Max Acc', alpha=0.7)
        axes[1, 1].set_xlabel('Training Steps', fontsize=12)
        axes[1, 1].set_ylabel('Test Accuracy', fontsize=12)
        axes[1, 1].set_title('Validation/Test Accuracy', fontsize=14, fontweight='bold')
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No Test Data Yet\n(Will appear after 10 steps)', 
                       ha='center', va='center', fontsize=14, transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Validation/Test Accuracy (Pending)', fontsize=14, fontweight='bold')
    
    # 添加统计信息
    stats_text = f"Total Steps: {max(steps)}\n"
    stats_text += f"Current Train Acc: {train_acc[-1]:.4f}\n"
    stats_text += f"Best Archive Acc: {max(archive_max):.4f}\n"
    if test_data:
        all_test_acc = [t['test_accuracy'] for t in test_data]
        stats_text += f"Best Test Acc: {max(all_test_acc):.4f}"
    
    fig.text(0.98, 0.02, stats_text, fontsize=11, ha='right', va='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 曲线图已保存: {save_path}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='绘制SparseFusion训练曲线')
    parser.add_argument('checkpoint', nargs='?', help='Checkpoint文件路径')
    parser.add_argument('--live', action='store_true', help='实时模式：自动找最新checkpoint并持续更新')
    parser.add_argument('--interval', type=int, default=60, help='实时模式下的更新间隔（秒）')
    parser.add_argument('--output', '-o', default='training_curves.png', help='输出图片路径')
    
    args = parser.parse_args()
    
    if args.live:
        print("🔄 实时监控模式启动...")
        print(f"   更新间隔: {args.interval}秒")
        print(f"   输出文件: {args.output}")
        print("   按Ctrl+C停止\n")
        
        last_checkpoint = None
        iteration = 0
        
        while True:
            try:
                latest = find_latest_checkpoint()
                
                if latest and latest != last_checkpoint:
                    print(f"[{time.strftime('%H:%M:%S')}] 发现新checkpoint: {os.path.basename(latest)}")
                    
                    data = load_checkpoint(latest)
                    iterations_data, test_data = extract_training_data(data)
                    
                    if iterations_data:
                        current_step = data.get('iteration', 0)
                        suffix = f" - Step {current_step}"
                        plot_curves(iterations_data, test_data, args.output, suffix)
                        print(f"   ✅ 已更新 (步数: {current_step}, 数据点: {len(iterations_data)})")
                    
                    last_checkpoint = latest
                    iteration += 1
                
                time.sleep(args.interval)
                
            except KeyboardInterrupt:
                print("\n\n👋 实时监控已停止")
                break
            except Exception as e:
                print(f"⚠️  错误: {e}")
                time.sleep(args.interval)
    
    else:
        # 单次模式
        checkpoint_path = args.checkpoint
        
        if not checkpoint_path:
            # 尝试自动找最新的
            checkpoint_path = find_latest_checkpoint()
            if not checkpoint_path:
                print("❌ 未找到checkpoint文件")
                print("   请指定文件路径，或使用 --live 进入实时模式")
                return
            print(f"📂 使用最新checkpoint: {checkpoint_path}")
        
        if not os.path.exists(checkpoint_path):
            print(f"❌ 文件不存在: {checkpoint_path}")
            return
        
        print(f"📊 加载数据: {checkpoint_path}")
        data = load_checkpoint(checkpoint_path)
        iterations_data, test_data = extract_training_data(data)
        
        print(f"   - 训练数据点: {len(iterations_data)}")
        print(f"   - 测试数据点: {len(test_data)}")
        
        if iterations_data:
            current_step = data.get('iteration', 0)
            suffix = f" - Step {current_step}"
            plot_curves(iterations_data, test_data, args.output, suffix)
        else:
            print("❌ 没有可用的训练数据")

if __name__ == "__main__":
    main()


