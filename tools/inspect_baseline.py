#!/usr/bin/env python3
import pickle

with open('/mnt/shared-storage-user/yefei/SparseFusion/results/checkpoints/checkpoint_run1_step3000_20251017_021740.pkl', 'rb') as f:
    data = pickle.load(f)

print("="*70)
print("BASELINE CHECKPOINT STRUCTURE")
print("="*70)

print(f"\nTop level type: {type(data)}")
print(f"Top level keys: {list(data.keys())}")

print(f"\n--- Checking 'results' field ---")
results = data['results']
print(f"results type: {type(results)}")
print(f"results length: {len(results)}")

if len(results) > 0:
    print(f"\nresults[0] type: {type(results[0])}")
    print(f"results[0]: {results[0]}")
    
    print(f"\nresults[1] type: {type(results[1])}")
    print(f"results[1]: {results[1]}")
    
    print(f"\nresults[-1] type: {type(results[-1])}")
    print(f"results[-1]: {results[-1]}")

print(f"\n--- Checking 'scores' field ---")
scores = data['scores']
print(f"scores type: {type(scores)}")
if hasattr(scores, 'shape'):
    print(f"scores shape: {scores.shape}")
    print(f"scores sample: {scores[:5]}")
elif isinstance(scores, list):
    print(f"scores length: {len(scores)}")
    print(f"scores sample: {scores[:5]}")
else:
    print(f"scores: {scores}")

print(f"\n--- Checking 'archive' field ---")
archive = data['archive']
print(f"archive type: {type(archive)}")
if hasattr(archive, 'shape'):
    print(f"archive shape: {archive.shape}")
elif isinstance(archive, dict):
    print(f"archive keys: {list(archive.keys())}")
    if len(archive) > 0:
        first_key = list(archive.keys())[0]
        print(f"archive[{first_key}]: {archive[first_key]}")

print("\n" + "="*70)

