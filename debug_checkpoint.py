#!/usr/bin/env python3
"""
Debug script to inspect checkpoint structure
"""

import pickle
import sys

def inspect_checkpoint(path):
    """详细检查checkpoint结构"""
    print(f"\n{'='*70}")
    print(f"检查 checkpoint: {path}")
    print('='*70)
    
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"✅ 文件加载成功")
        print(f"📦 数据类型: {type(data)}")
        
        if isinstance(data, dict):
            print(f"📋 Keys: {list(data.keys())}")
            print()
            
            # 检查每个key
            for key in data.keys():
                val = data[key]
                print(f"  [{key}]:")
                print(f"    类型: {type(val)}")
                
                if isinstance(val, (list, dict)):
                    print(f"    长度: {len(val)}")
                    
                    # 如果是history
                    if key == 'history' and len(val) > 0:
                        first = val[0] if isinstance(val, list) else list(val.values())[0]
                        if isinstance(first, dict):
                            print(f"    第一个entry的keys: {list(first.keys())}")
                    
                    # 如果是archive
                    elif key == 'archive' and len(val) > 0:
                        if isinstance(val, dict):
                            first_key = list(val.keys())[0]
                            first_val = val[first_key]
                            print(f"    第一个individual: {type(first_val)}")
                            if isinstance(first_val, dict):
                                print(f"    Individual keys: {list(first_val.keys())}")
                else:
                    print(f"    值: {val}")
                print()
        
        elif isinstance(data, list):
            print(f"📏 列表长度: {len(data)}")
            if len(data) > 0:
                print(f"📋 第一个元素类型: {type(data[0])}")
                if isinstance(data[0], dict):
                    print(f"📋 第一个元素keys: {list(data[0].keys())}")
                    print(f"\n第一个元素内容:")
                    for k, v in data[0].items():
                        if isinstance(v, (list, dict)):
                            print(f"  {k}: {type(v)} (长度={len(v)})")
                        else:
                            print(f"  {k}: {v}")
                
                if len(data) > 1:
                    print(f"\n最后一个元素类型: {type(data[-1])}")
                    if isinstance(data[-1], dict):
                        print(f"最后一个元素keys: {list(data[-1].keys())}")
        
        else:
            print(f"❓ 未知数据结构: {type(data)}")
        
        print('='*70)
        return data
        
    except FileNotFoundError:
        print(f"❌ 错误: 文件不存在")
        print('='*70)
        return None
    except Exception as e:
        print(f"❌ 错误: {e}")
        print('='*70)
        return None


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python debug_checkpoint.py <checkpoint_path1> [checkpoint_path2]")
        sys.exit(1)
    
    for path in sys.argv[1:]:
        inspect_checkpoint(path)

