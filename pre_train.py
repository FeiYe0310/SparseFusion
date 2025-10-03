"""
Natural Niches 预训练模型生成 - 简洁版
只测试每个模型在0-4和5-9两个区间的准确率
"""

import os
import sys
import pickle
import time
from datetime import datetime

# 设置路径和JAX环境
sys.path.append('/fs-computility/pdz-grp1/yefei.p/Niches_nature/natural_niches')
os.environ['JAX_PLATFORM_NAME'] = 'cpu'  # 如果要用GPU，删除这行

# 导入repo中现有的函数
from helper_fn import get_pre_trained_models
from data import load_data
from model import mlp, get_acc, num_params

def evaluate_model_on_groups(model_params, x_test, y_test, model_name):
    """评估模型在两个数字区间的性能"""
    
    print(f"\n{model_name} 区间测试:")
    
    # 0-4区间测试
    mask_0_to_4 = y_test < 5
    x_test_0_to_4 = x_test[mask_0_to_4]
    y_test_0_to_4 = y_test[mask_0_to_4]
    
    logits_0_to_4 = mlp(model_params, x_test_0_to_4)
    acc_0_to_4 = get_acc(logits_0_to_4, y_test_0_to_4)
    
    # 5-9区间测试
    mask_5_to_9 = y_test >= 5
    x_test_5_to_9 = x_test[mask_5_to_9]
    y_test_5_to_9 = y_test[mask_5_to_9]
    
    logits_5_to_9 = mlp(model_params, x_test_5_to_9)
    acc_5_to_9 = get_acc(logits_5_to_9, y_test_5_to_9)
    
    print(f"  数字0-4区间准确率: {acc_0_to_4:.4f}")
    print(f"  数字5-9区间准确率: {acc_5_to_9:.4f}")
    
    return float(acc_0_to_4), float(acc_5_to_9)

def generate_pretrained_models_simple():
    """生成预训练模型并进行区间测试"""
    
    print("=" * 50)
    print("Natural Niches 预训练模型生成")
    print("=" * 50)
    
    # 1. 加载数据
    print("\n1. 加载数据集...")
    (x_train, y_train), (x_test, y_test) = load_data()
    print(f"   训练集: {x_train.shape}")
    print(f"   测试集: {x_test.shape}")
    
    # 2. 数据分布
    print(f"\n2. 数据分布:")
    train_0_to_4 = int(jnp.sum(y_train < 5))
    train_5_to_9 = int(jnp.sum(y_train >= 5))
    test_0_to_4 = int(jnp.sum(y_test < 5))
    test_5_to_9 = int(jnp.sum(y_test >= 5))
    
    print(f"   训练集 0-4: {train_0_to_4}个样本")
    print(f"   训练集 5-9: {train_5_to_9}个样本")
    print(f"   测试集 0-4: {test_0_to_4}个样本")
    print(f"   测试集 5-9: {test_5_to_9}个样本")
    
    # 3. 训练模型
    print(f"\n3. 训练专门化模型...")
    print(f"   模型参数总数: {num_params}")
    
    start_time = time.time()
    model_1, model_2 = get_pre_trained_models()
    training_time = time.time() - start_time
    
    print(f"   训练完成，耗时: {training_time:.2f}秒")
    
    # 4. 区间测试
    print(f"\n4. 区间准确率测试:")
    print("=" * 30)
    
    # 测试Model 1 (应该是0-4专家)
    acc_1_on_0_to_4, acc_1_on_5_to_9 = evaluate_model_on_groups(
        model_1, x_test, y_test, "Model 1 (0-4专家)"
    )
    
    # 测试Model 2 (应该是5-9专家)
    acc_2_on_0_to_4, acc_2_on_5_to_9 = evaluate_model_on_groups(
        model_2, x_test, y_test, "Model 2 (5-9专家)"
    )
    
    # 5. 专门化效果分析
    print(f"\n5. 专门化效果:")
    print("=" * 30)
    
    model_1_specialization = acc_1_on_0_to_4 - acc_1_on_5_to_9
    model_2_specialization = acc_2_on_5_to_9 - acc_2_on_0_to_4
    
    print(f"Model 1 专门化效果: {model_1_specialization:.4f} (目标-非目标)")
    print(f"Model 2 专门化效果: {model_2_specialization:.4f} (目标-非目标)")
    
    avg_specialization = (model_1_specialization + model_2_specialization) / 2
    print(f"平均专门化效果: {avg_specialization:.4f}")
    
    # 判断专门化质量
    if avg_specialization > 0.3:
        quality = "优秀 ✅"
    elif avg_specialization > 0.2:
        quality = "良好 ✅"
    elif avg_specialization > 0.1:
        quality = "中等 ⚠️"
    else:
        quality = "较差 ❌"
    
    print(f"专门化质量: {quality}")
    
    # 6. 保存模型
    print(f"\n6. 保存模型...")
    
    results_dir = "/fs-computility/pdz-grp1/yefei.p/Niches_nature/natural_niches/results"
    os.makedirs(results_dir, exist_ok=True)
    
    model_data = {
        "model_1_params": model_1,
        "model_2_params": model_2,
        "performance_results": {
            "model_1": {
                "acc_on_0_to_4": acc_1_on_0_to_4,
                "acc_on_5_to_9": acc_1_on_5_to_9,
                "specialization_effect": model_1_specialization
            },
            "model_2": {
                "acc_on_0_to_4": acc_2_on_0_to_4,
                "acc_on_5_to_9": acc_2_on_5_to_9,
                "specialization_effect": model_2_specialization
            },
            "summary": {
                "avg_specialization_effect": avg_specialization,
                "specialization_quality": quality
            }
        },
        "data_info": {
            "train_0_to_4_samples": train_0_to_4,
            "train_5_to_9_samples": train_5_to_9,
            "test_0_to_4_samples": test_0_to_4,
            "test_5_to_9_samples": test_5_to_9
        },
        "training_info": {
            "training_time_seconds": training_time,
            "num_params": num_params,
            "timestamp": datetime.now().isoformat()
        }
    }
    
    output_file = os.path.join(results_dir, "simple_pretrained_models.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(model_data, f)
    
    print(f"   ✅ 模型已保存: {output_file}")
    
    # 7. 总结报告
    print(f"\n" + "=" * 50)
    print("训练总结")
    print("=" * 50)
    print(f"Model 1 (0-4专家):")
    print(f"  ├─ 在0-4区间: {acc_1_on_0_to_4:.4f}")
    print(f"  └─ 在5-9区间: {acc_1_on_5_to_9:.4f}")
    
    print(f"Model 2 (5-9专家):")
    print(f"  ├─ 在0-4区间: {acc_2_on_0_to_4:.4f}")
    print(f"  └─ 在5-9区间: {acc_2_on_5_to_9:.4f}")
    
    print(f"专门化效果: {avg_specialization:.4f} ({quality})")
    
    return model_data

def load_and_test_models():
    """加载已训练模型并快速测试"""
    
    results_dir = "/fs-computility/pdz-grp1/yefei.p/Niches_nature/natural_niches/results"
    model_file = os.path.join(results_dir, "simple_pretrained_models.pkl")
    
    if os.path.exists(model_file):
        with open(model_file, "rb") as f:
            data = pickle.load(f)
        
        print("✅ 已加载预训练模型")
        print("性能摘要:")
        
        results = data['performance_results']
        print(f"Model 1: 0-4区间 {results['model_1']['acc_on_0_to_4']:.4f}, 5-9区间 {results['model_1']['acc_on_5_to_9']:.4f}")
        print(f"Model 2: 0-4区间 {results['model_2']['acc_on_0_to_4']:.4f}, 5-9区间 {results['model_2']['acc_on_5_to_9']:.4f}")
        
        return data
    else:
        print("❌ 未找到预训练模型，请先运行训练")
        return None

if __name__ == "__main__":
    try:
        import jax.numpy as jnp
        
        print("Natural Niches 简洁版预训练模型生成")
        print("只测试0-4和5-9两个区间的准确率\n")
        
        # 生成模型
        model_data = generate_pretrained_models_simple()
        
        print(f"\n🎉 预训练完成!")
        print("可以进行后续的Natural Niches进化实验")
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()