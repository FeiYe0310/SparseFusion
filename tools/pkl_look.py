"""
加载和探索simple_pretrained_models.pkl文件
"""

import pickle
import sys
import os
import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jrandom
import jax.tree_util as jtree
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jrandom
import jax.tree_util as jtree

# 加载PKL文件
def load_and_explore_pkl():
    pkl_file = "/fs-computility/pdz-grp1/yefei.p/Niches_nature/natural_niches/results/simple_pretrained_models.pkl"
    
    print("=" * 50)
    print("PKL文件分析")
    print("=" * 50)
    
    # 1. 检查文件存在性和大小
    if os.path.exists(pkl_file):
        file_size = os.path.getsize(pkl_file)
        print(f"✅ 文件存在")
        print(f"📁 文件路径: {pkl_file}")
        print(f"📏 文件大小: {file_size:,} 字节 ({file_size/1024:.1f} KB)")
    else:
        print(f"❌ 文件不存在: {pkl_file}")
        return None
    
    # 2. 加载PKL文件
    try:
        with open(pkl_file, "rb") as f:  # 'rb' = read binary
            data = pickle.load(f)
        print(f"✅ PKL文件加载成功")
    except Exception as e:
        print(f"❌ PKL文件加载失败: {e}")
        return None
    
    # 3. 分析文件内容结构
    print(f"\n📊 PKL文件内容分析:")
    print(f"   数据类型: {type(data)}")
    
    if isinstance(data, dict):
        print(f"   字典键数量: {len(data)}")
        print(f"   字典键列表:")
        for key in data.keys():
            print(f"     - {key}: {type(data[key])}")
    
    return data

# 运行分析
data = load_and_explore_pkl()
"""
深入分析您的预训练模型数据
"""

def analyze_pretrained_models(data):
    """详细分析预训练模型PKL文件"""
    
    if data is None:
        return
    
    print(f"\n" + "=" * 50)
    print("预训练模型详细分析")
    print("=" * 50)
    
    # 1. 模型参数分析
    if 'model_1_params' in data:
        model_1 = data['model_1_params']
        print(f"\n🤖 Model 1 (数字0-4专家):")
        print(f"   参数类型: {type(model_1)}")
        print(f"   参数形状: {model_1.shape}")
        print(f"   参数数量: {model_1.size:,}")
        print(f"   数据类型: {model_1.dtype}")
        print(f"   参数范围: [{model_1.min():.6f}, {model_1.max():.6f}]")
        print(f"   参数均值: {model_1.mean():.6f}")
        print(f"   参数标准差: {model_1.std():.6f}")
    
    if 'model_2_params' in data:
        model_2 = data['model_2_params']
        print(f"\n🤖 Model 2 (数字5-9专家):")
        print(f"   参数类型: {type(model_2)}")
        print(f"   参数形状: {model_2.shape}")
        print(f"   参数数量: {model_2.size:,}")
        print(f"   数据类型: {model_2.dtype}")
        print(f"   参数范围: [{model_2.min():.6f}, {model_2.max():.6f}]")
        print(f"   参数均值: {model_2.mean():.6f}")
        print(f"   参数标准差: {model_2.std():.6f}")
    
    # 2. 性能结果分析
    if 'performance_results' in data:
        perf = data['performance_results']
        print(f"\n📈 性能结果:")
        
        if 'model_1' in perf:
            m1_perf = perf['model_1']
            print(f"   Model 1性能:")
            for key, value in m1_perf.items():
                print(f"     {key}: {value:.4f}")
        
        if 'model_2' in perf:
            m2_perf = perf['model_2']
            print(f"   Model 2性能:")
            for key, value in m2_perf.items():
                print(f"     {key}: {value:.4f}")
    
    # 3. 元数据分析
    print(f"\n📋 元数据信息:")
    for key, value in data.items():
        if key not in ['model_1_params', 'model_2_params', 'performance_results']:
            print(f"   {key}: {value}")

# 运行详细分析
if data:
    analyze_pretrained_models(data)

"""
使用PKL文件中的模型进行预测
"""

def use_models_for_prediction():
    """加载模型并进行预测演示"""
    
    # 设置环境
    sys.path.append('/fs-computility/pdz-grp1/yefei.p/Niches_nature/natural_niches')
    os.environ['JAX_PLATFORM_NAME'] = 'cpu'
    
    # 加载模型数据
    pkl_file = "/fs-computility/pdz-grp1/yefei.p/Niches_nature/natural_niches/results/simple_pretrained_models.pkl"
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)
    
    # 获取模型参数
    model_1 = data['model_1_params']
    model_2 = data['model_2_params']
    
    print(f"\n" + "=" * 50)
    print("模型预测演示")
    print("=" * 50)
    
    try:
        # 导入必要的函数
        from model import mlp, get_acc
        from data import load_data
        import jax.numpy as jnp
        
        # 加载测试数据
        (x_train, y_train), (x_test, y_test) = load_data()
        
        # 预测几个样本
        print(f"\n🔮 样本预测演示 (前10个测试样本):")
        print("样本ID | 真实标签 | Model1预测 | Model2预测 | 推荐专家")
        print("-" * 60)
        
        for i in range(min(10, len(x_test))):
            # 单个样本
            x_sample = x_test[i:i+1]
            true_label = y_test[i]
            
            # Model 1预测
            logits_1 = mlp(model_1, x_sample)
            pred_1 = jnp.argmax(logits_1, axis=1)[0]
            conf_1 = jnp.max(jax.nn.softmax(logits_1, axis=1))
            
            # Model 2预测
            logits_2 = mlp(model_2, x_sample)
            pred_2 = jnp.argmax(logits_2, axis=1)[0]
            conf_2 = jnp.max(jax.nn.softmax(logits_2, axis=1))
            
            # 确定推荐专家
            if true_label < 5:
                expert = "Model1"
                expert_pred = pred_1
                expert_conf = conf_1
            else:
                expert = "Model2"
                expert_pred = pred_2
                expert_conf = conf_2
            
            print(f"   {i:2d}   |    {true_label}     |     {pred_1}      |     {pred_2}      |  {expert}")
        
        # 整体性能评估
        print(f"\n📊 整体性能评估:")
        
        # 分组数据
        mask_0_to_4 = y_test < 5
        mask_5_to_9 = y_test >= 5
        
        x_test_0_to_4 = x_test[mask_0_to_4]
        y_test_0_to_4 = y_test[mask_0_to_4]
        x_test_5_to_9 = x_test[mask_5_to_9]
        y_test_5_to_9 = y_test[mask_5_to_9]
        
        # Model 1评估
        acc_1_on_0_to_4 = get_acc(mlp(model_1, x_test_0_to_4), y_test_0_to_4)
        acc_1_on_5_to_9 = get_acc(mlp(model_1, x_test_5_to_9), y_test_5_to_9)
        
        # Model 2评估
        acc_2_on_0_to_4 = get_acc(mlp(model_2, x_test_0_to_4), y_test_0_to_4)
        acc_2_on_5_to_9 = get_acc(mlp(model_2, x_test_5_to_9), y_test_5_to_9)
        
        print(f"   Model 1: 0-4区间={acc_1_on_0_to_4:.4f}, 5-9区间={acc_1_on_5_to_9:.4f}")
        print(f"   Model 2: 0-4区间={acc_2_on_0_to_4:.4f}, 5-9区间={acc_2_on_5_to_9:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 预测演示失败: {e}")
        import traceback
        traceback.print_exc()
        return False

# 运行预测演示
use_models_for_prediction()

"""
PKL文件的常用操作大全
"""

def pkl_operations_guide():
    """PKL文件操作指南"""
    
    pkl_file = "/fs-computility/pdz-grp1/yefei.p/Niches_nature/natural_niches/results/simple_pretrained_models.pkl"
    
    print("=" * 50)
    print("PKL文件操作指南")
    print("=" * 50)
    
    # 1. 加载PKL文件
    print(f"\n1️⃣ 加载PKL文件:")
    print(f"```python")
    print(f"import pickle")
    print(f"with open('{pkl_file}', 'rb') as f:")
    print(f"    data = pickle.load(f)")
    print(f"```")
    
    # 2. 保存PKL文件
    print(f"\n2️⃣ 保存PKL文件:")
    print(f"```python")
    print(f"import pickle")
    print(f"data = {{'model': model, 'results': results}}")
    print(f"with open('my_data.pkl', 'wb') as f:")
    print(f"    pickle.dump(data, f)")
    print(f"```")
    
    # 3. 查看PKL文件内容
    print(f"\n3️⃣ 查看PKL文件内容:")
    print(f"```python")
    print(f"# 查看数据类型")
    print(f"print(type(data))")
    print(f"")
    print(f"# 如果是字典，查看键")
    print(f"if isinstance(data, dict):")
    print(f"    print(data.keys())")
    print(f"```")
    
    # 4. 提取特定数据
    print(f"\n4️⃣ 提取特定数据:")
    print(f"```python")
    print(f"# 提取模型参数")
    print(f"model_1 = data['model_1_params']")
    print(f"model_2 = data['model_2_params']")
    print(f"")
    print(f"# 提取性能结果")
    print(f"performance = data['performance_results']")
    print(f"```")
    
    # 5. 修改和重新保存
    print(f"\n5️⃣ 修改和重新保存:")
    print(f"```python")
    print(f"# 添加新数据")
    print(f"data['new_info'] = 'some additional data'")
    print(f"")
    print(f"# 重新保存")
    print(f"with open('updated_data.pkl', 'wb') as f:")
    print(f"    pickle.dump(data, f)")
    print(f"```")
    
    # 6. 安全加载（处理版本兼容性）
    print(f"\n6️⃣ 安全加载（处理版本兼容性）:")
    print(f"```python")
    print(f"import pickle")
    print(f"try:")
    print(f"    with open('data.pkl', 'rb') as f:")
    print(f"        data = pickle.load(f)")
    print(f"except Exception as e:")
    print(f"    print(f'加载失败: {{e}}')")
    print(f"```")

# 运行操作指南
pkl_operations_guide()

# 运行这个脚本来分析您的PKL文件
if __name__ == "__main__":
    print("🔍 开始分析您的PKL文件...")
    
    # 1. 加载和基本分析
    data = load_and_explore_pkl()
    
    if data:
        # 2. 详细分析
        analyze_pretrained_models(data)
        
        # 3. 预测演示
        print(f"\n" + "="*50)
        use_models_for_prediction()
        
        print(f"\n🎉 PKL文件分析完成！")
        print(f"您现在已经了解了:")
        print(f"  ✅ PKL文件的结构和内容")
        print(f"  ✅ 两个专门化模型的参数")
        print(f"  ✅ 模型的性能表现")
        print(f"  ✅ 如何使用模型进行预测")
    else:
        print(f"❌ 无法分析PKL文件")
        