#!/usr/bin/env python3
import pickle

print("="*70)
print("CHECKPOINT 1: Baseline")
print("="*70)
with open('/mnt/shared-storage-user/yefei/SparseFusion/results/checkpoints/checkpoint_run1_step3000_20251017_021740.pkl', 'rb') as f:
    data1 = pickle.load(f)

print(f"Type: {type(data1)}")
print(f"Keys: {list(data1.keys())}")
print(f"\nresults type: {type(data1['results'])}")

if isinstance(data1['results'], tuple):
    print(f"results is tuple, length: {len(data1['results'])}")
    print(f"results[0] type: {type(data1['results'][0])}")
    print(f"results[1] type: {type(data1['results'][1])}")
    if isinstance(data1['results'][1], dict):
        print(f"results[1] keys: {list(data1['results'][1].keys())}")
        if 'evaluation_history' in data1['results'][1]:
            history = data1['results'][1]['evaluation_history']
            print(f"evaluation_history length: {len(history)}")
            if len(history) > 0:
                print(f"First 3 records:")
                for i in range(min(3, len(history))):
                    print(f"  {history[i]}")

print("\n" + "="*70)
print("CHECKPOINT 2: Sparsity-aware")
print("="*70)
with open('/mnt/shared-storage-user/yefei/SparseFusion/results/sparsity_aware_w0.80_b0.20_t1.00_prune_wanda_0.30.pkl', 'rb') as f:
    data2 = pickle.load(f)

print(f"Type: {type(data2)}")
print(f"Length: {len(data2)}")
print(f"\nFirst element type: {type(data2[0])}")

if isinstance(data2[0], tuple):
    print(f"First element is tuple, length: {len(data2[0])}")
    print(f"data2[0][0] type: {type(data2[0][0])}")
    print(f"data2[0][1] type: {type(data2[0][1])}")
    
    if isinstance(data2[0][1], dict):
        print(f"data2[0][1] keys: {list(data2[0][1].keys())}")
        if 'evaluation_history' in data2[0][1]:
            history = data2[0][1]['evaluation_history']
            print(f"evaluation_history length: {len(history)}")
            if len(history) > 0:
                print(f"First 3 records:")
                for i in range(min(3, len(history))):
                    print(f"  {history[i]}")
                print(f"\nLast 3 records:")
                for i in range(max(0, len(history)-3), len(history)):
                    print(f"  {history[i]}")

