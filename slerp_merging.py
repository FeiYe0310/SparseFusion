"""
SLERP模型融合脚本 - 修复版
使用球面线性插值(SLERP)融合两个预训练模型，支持权重调整和文件命名
"""

import os
import sys
import pickle
import time
from datetime import datetime

# ========== 全局配置参数 ==========
FUSION_WEIGHT = 0.5  # 🎯 可调整的融合权重 (0.0-1.0)

# 设置路径
sys.path.append('/fs-computility/pdz-grp1/yefei.p/Niches_nature/natural_niches')

import jax
import jax.numpy as jnp
from helper_fn import slerp
from data import load_data
from model import mlp, get_acc

def slerp_fusion_complete():
    """完整的SLERP融合流程"""
    
    print("🔀 SLERP模型融合测试")
    print("=" * 50)
    print(f"🎯 融合权重: {FUSION_WEIGHT:.2f}")
    print(f"   (0.0=完全Model1, 1.0=完全Model2)")
    print("=" * 50)
    
    # 1. 加载预训练模型
    print("\n📂 加载预训练模型...")
    pkl_path = "/fs-computility/pdz-grp1/yefei.p/Niches_nature/natural_niches/results/simple_pretrained_models.pkl"
    
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    
    model_1 = data["model_1_params"]  # 0-4专家
    model_2 = data["model_2_params"]  # 5-9专家
    original_performance = data.get("performance_results", {})
    
    print(f"✅ 模型加载完成")
    print(f"   Model 1 (0-4专家): {model_1.shape}")
    print(f"   Model 2 (5-9专家): {model_2.shape}")
    
    # 2. 加载测试数据
    print(f"\n📊 加载测试数据...")
    (x_train, y_train), (x_test, y_test) = load_data()
    print(f"✅ 数据加载完成: 测试集 {x_test.shape}")
    
    # 3. 显示原始模型性能
    print(f"\n📈 原始模型性能:")
    if original_performance:
        model_1_perf = original_performance.get("model_1", {})
        model_2_perf = original_performance.get("model_2", {})
        
        if model_1_perf:
            print(f"   Model 1 (0-4专家):")
            print(f"     0-4区间: {model_1_perf.get('acc_on_0_to_4', 'N/A'):.4f}")
            print(f"     5-9区间: {model_1_perf.get('acc_on_5_to_9', 'N/A'):.4f}")
        
        if model_2_perf:
            print(f"   Model 2 (5-9专家):")
            print(f"     0-4区间: {model_2_perf.get('acc_on_0_to_4', 'N/A'):.4f}")
            print(f"     5-9区间: {model_2_perf.get('acc_on_5_to_9', 'N/A'):.4f}")
    
    # 4. SLERP融合
    print(f"\n🔀 执行SLERP融合 (权重={FUSION_WEIGHT:.2f})...")
    start_time = time.time()
    
    merged_model = slerp(FUSION_WEIGHT, model_1, model_2)
    
    fusion_time = time.time() - start_time
    print(f"✅ 融合完成，耗时: {fusion_time:.4f}秒")
    print(f"   融合模型形状: {merged_model.shape}")
    
    # 5. 评估融合模型
    print(f"\n📊 评估融合模型...")
    
    # 分组测试数据
    mask_0_to_4 = y_test < 5
    mask_5_to_9 = y_test >= 5
    
    x_test_0_to_4 = x_test[mask_0_to_4]
    y_test_0_to_4 = y_test[mask_0_to_4]
    x_test_5_to_9 = x_test[mask_5_to_9] 
    y_test_5_to_9 = y_test[mask_5_to_9]
    
    print(f"   0-4组样本数: {len(x_test_0_to_4)}")
    print(f"   5-9组样本数: {len(x_test_5_to_9)}")
    
    # 计算准确率
    acc_0_to_4 = get_acc(mlp(merged_model, x_test_0_to_4), y_test_0_to_4)
    acc_5_to_9 = get_acc(mlp(merged_model, x_test_5_to_9), y_test_5_to_9)
    overall_acc = (acc_0_to_4 + acc_5_to_9) / 2.0
    
    # 6. 显示结果
    print(f"\n🏆 融合结果 (权重={FUSION_WEIGHT:.2f}):")
    print("=" * 40)
    print(f"   0-4区间准确率: {acc_0_to_4:.4f}")
    print(f"   5-9区间准确率: {acc_5_to_9:.4f}")
    print(f"   平均准确率:     {overall_acc:.4f}")
    
    # 7. 保存结果 - 文件名包含权重信息
    print(f"\n💾 保存结果...")
    
    results_data = {
        "fusion_weight": FUSION_WEIGHT,
        "original_models": {
            "model_1_params": model_1,
            "model_2_params": model_2,
            "performance": original_performance
        },
        "fusion_result": {
            "model_params": merged_model,
            "acc_0_to_4": float(acc_0_to_4),
            "acc_5_to_9": float(acc_5_to_9),
            "overall_acc": float(overall_acc),
            "fusion_time": fusion_time
        },
        "config": {
            "fusion_weight": FUSION_WEIGHT,
            "fusion_method": "SLERP"
        },
        "timestamp": datetime.now().isoformat()
    }
    
    results_dir = "/fs-computility/pdz-grp1/yefei.p/Niches_nature/natural_niches/results"
    
    # 📁 文件名包含权重信息
    weight_str = f"{FUSION_WEIGHT:.2f}".replace(".", "p")  # 0.5 -> 0p5
    output_file = os.path.join(results_dir, f"slerp_fusion_w{weight_str}.pkl")
    
    with open(output_file, "wb") as f:
        pickle.dump(results_data, f)
    
    print(f"✅ 结果已保存: slerp_fusion_w{weight_str}.pkl")
    
    # 8. 总结
    print(f"\n🎉 SLERP融合完成!")
    print(f"   最终权重: {FUSION_WEIGHT:.2f}")
    print(f"   最终性能: {overall_acc:.4f}")
    print(f"   结果文件: slerp_fusion_w{weight_str}.pkl")
    
    return results_data

def change_weight_and_test():
    """修改权重并测试多个值"""
    
    global FUSION_WEIGHT
    
    print("🎯 多权重SLERP融合测试")
    print("=" * 50)
    
    # 测试多个权重值
    test_weights = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    print(f"📊 将测试权重: {test_weights}")
    print()
    
    # 加载模型和数据
    pkl_path = "/fs-computility/pdz-grp1/yefei.p/Niches_nature/natural_niches/results/simple_pretrained_models.pkl"
    
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    
    model_1 = data["model_1_params"]
    model_2 = data["model_2_params"]
    
    (_, _), (x_test, y_test) = load_data()
    
    # 分组数据
    mask_0_to_4 = y_test < 5
    mask_5_to_9 = y_test >= 5
    
    x_test_0_to_4 = x_test[mask_0_to_4]
    y_test_0_to_4 = y_test[mask_0_to_4]
    x_test_5_to_9 = x_test[mask_5_to_9]
    y_test_5_to_9 = y_test[mask_5_to_9]
    
    print("多权重融合结果:")
    print("权重    0-4区间    5-9区间    平均")
    print("=" * 40)
    
    best_weight = 0.5
    best_avg = 0.0
    
    for weight in test_weights:
        FUSION_WEIGHT = weight
        
        # SLERP融合
        merged_model = slerp(weight, model_1, model_2)
        
        # 评估
        acc_0_to_4 = get_acc(mlp(merged_model, x_test_0_to_4), y_test_0_to_4)
        acc_5_to_9 = get_acc(mlp(merged_model, x_test_5_to_9), y_test_5_to_9)
        avg_acc = (acc_0_to_4 + acc_5_to_9) / 2.0
        
        print(f"{weight:.1f}     {acc_0_to_4:.4f}     {acc_5_to_9:.4f}     {avg_acc:.4f}")
        
        # 保存每个权重的结果
        weight_str = f"{weight:.1f}".replace(".", "p")
        results_data = {
            "fusion_weight": weight,
            "fusion_result": {
                "model_params": merged_model,
                "acc_0_to_4": float(acc_0_to_4),
                "acc_5_to_9": float(acc_5_to_9),
                "overall_acc": float(avg_acc)
            }
        }
        
        output_file = f"/fs-computility/pdz-grp1/yefei.p/Niches_nature/natural_niches/results/slerp_fusion_w{weight_str}.pkl"
        with open(output_file, "wb") as f:
            pickle.dump(results_data, f)
        
        if avg_acc > best_avg:
            best_avg = avg_acc
            best_weight = weight
    
    print(f"\n🏆 最佳权重: {best_weight:.1f} (平均准确率: {best_avg:.4f})")
    
    # 设置为最佳权重
    FUSION_WEIGHT = best_weight
    return best_weight

if __name__ == "__main__":
    print("🔀 Natural Niches SLERP模型融合 - 修复版")
    print()
    
    try:
        print("运行模式选择:")
        print("1 - 使用当前权重进行单次融合")
        print("2 - 测试多个权重并找到最佳值")
        print("3 - 快速融合(最简洁输出)")
        print()
        
        # 强制刷新输出缓冲区
        sys.stdout.flush()
        
        # 获取用户选择，如果没有输入则使用默认值
        try:
            choice = input(f"请选择模式 (1/2/3) [默认: 1]: ").strip() or "1"
        except (EOFError, KeyboardInterrupt):
            choice = "1"  # 默认选择
            print("使用默认模式 1")
        
        if choice == "2":
            print("\n执行多权重测试...")
            best_weight = change_weight_and_test()
            print(f"\n使用最佳权重 {best_weight:.1f} 进行最终融合...")
            slerp_fusion_complete()
            
        elif choice == "3":
            print(f"\n快速融合 (权重={FUSION_WEIGHT:.2f})...")
            
            # 简化版本
            pkl_path = "/fs-computility/pdz-grp1/yefei.p/Niches_nature/natural_niches/results/simple_pretrained_models.pkl"
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            
            model_1 = data["model_1_params"]
            model_2 = data["model_2_params"]
            
            (_, _), (x_test, y_test) = load_data()
            
            mask_0_to_4 = y_test < 5
            mask_5_to_9 = y_test >= 5
            
            merged_model = slerp(FUSION_WEIGHT, model_1, model_2)
            
            acc_0_to_4 = get_acc(mlp(merged_model, x_test[mask_0_to_4]), y_test[mask_0_to_4])
            acc_5_to_9 = get_acc(mlp(merged_model, x_test[mask_5_to_9]), y_test[mask_5_to_9])
            
            print(f"权重 {FUSION_WEIGHT:.2f}:")
            print(f"  0-4区间: {acc_0_to_4:.4f}")
            print(f"  5-9区间: {acc_5_to_9:.4f}")
            
        else:  # 默认选择1
            print(f"\n执行单次融合 (权重={FUSION_WEIGHT:.2f})...")
            slerp_fusion_complete()
        
        print("\n🎉 所有任务完成!")
        
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()