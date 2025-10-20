#!/usr/bin/env python3
import pickle
from collections import defaultdict

baseline_path = '/mnt/shared-storage-user/yefei/SparseFusion/results/checkpoints/checkpoint_run1_step3000_20251017_021740.pkl'

with open(baseline_path, 'rb') as f:
    data = pickle.load(f)

print("="*70)
print("BASELINE DATA EXTRACTION DEBUG")
print("="*70)

print(f"\n1. Top level type: {type(data)}")
print(f"   Has 'results' key: {'results' in data}")

results = data['results']
print(f"\n2. results type: {type(results)}")
print(f"   results is list: {isinstance(results, list)}")
print(f"   results length: {len(results)}")

if len(results) > 0:
    first = results[0]
    print(f"\n3. results[0] type: {type(first)}")
    print(f"   results[0] is dict: {isinstance(first, dict)}")
    print(f"   results[0] is defaultdict: {isinstance(first, defaultdict)}")
    print(f"   results[0] has keys: {hasattr(first, 'keys')}")
    
    if hasattr(first, 'keys'):
        print(f"   results[0] keys: {list(first.keys())}")
        print(f"   Has 'test_evaluations': {'test_evaluations' in first}")
        
        if 'test_evaluations' in first:
            evals = first['test_evaluations']
            print(f"\n4. test_evaluations type: {type(evals)}")
            print(f"   test_evaluations length: {len(evals)}")
            print(f"   First 3 records: {evals[:3]}")
            print(f"   Last 3 records: {evals[-3:]}")

print("\n" + "="*70)
print("EXTRACTION LOGIC TEST")
print("="*70)

records = []

if isinstance(data, dict) and 'results' in data:
    results = data['results']
    print(f"✓ Found results key, type: {type(results)}")
    
    if isinstance(results, list) and len(results) > 0:
        print(f"✓ results is list with {len(results)} elements")
        
        first_elem = results[0]
        print(f"✓ results[0] type: {type(first_elem)}")
        print(f"  isinstance(results[0], dict): {isinstance(first_elem, dict)}")
        
        if isinstance(first_elem, dict) and 'test_evaluations' in first_elem:
            print(f"✓ Found test_evaluations in results[0]")
            records = first_elem['test_evaluations']
        else:
            print(f"✗ No test_evaluations, using results directly")
            records = results

print(f"\nFinal extracted records: {len(records)}")
if len(records) > 0:
    print(f"First record: {records[0]}")
    print(f"Last record: {records[-1]}")

