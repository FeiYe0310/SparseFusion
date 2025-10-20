#!/usr/bin/env python3
"""
Analyze Sparsity-Aware Natural Niches Experiments
"""

import os
import sys
import pickle
import glob
from typing import Dict, List, Tuple
import numpy as np


def load_pickle_safe(filepath: str) -> dict:
    """Load a pickle file safely."""
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"⚠️  Error loading {filepath}: {e}")
        return None


def analyze_experiment(exp_dir: str) -> Dict[str, Dict]:
    """
    Analyze all experiments in a directory.
    
    Returns:
        Dictionary mapping experiment names to their statistics
    """
    results = {}
    
    # Find all .pkl files
    pkl_files = glob.glob(os.path.join(exp_dir, "*.pkl"))
    
    if not pkl_files:
        print(f"❌ No .pkl files found in {exp_dir}")
        return results
    
    print(f"\n📂 Found {len(pkl_files)} result files:")
    for pkl_file in pkl_files:
        print(f"   - {os.path.basename(pkl_file)}")
    
    for pkl_file in sorted(pkl_files):
        exp_name = os.path.splitext(os.path.basename(pkl_file))[0]
        data = load_pickle_safe(pkl_file)
        
        if data is None:
            continue
        
        # Extract statistics
        # Note: The structure depends on what's saved in the pickle file
        # For now, we'll extract basic info if available
        
        results[exp_name] = {
            'file': pkl_file,
            'data': data,
            'status': 'loaded'
        }
    
    return results


def print_summary(results: Dict[str, Dict]):
    """Print a summary of all experiments."""
    
    print("\n" + "=" * 70)
    print("📊 EXPERIMENT SUMMARY")
    print("=" * 70)
    
    if not results:
        print("No results to display.")
        return
    
    # Create a table
    headers = ["Experiment", "Status", "File Size"]
    rows = []
    
    for exp_name, exp_data in sorted(results.items()):
        file_size = os.path.getsize(exp_data['file'])
        size_mb = file_size / (1024 * 1024)
        
        rows.append([
            exp_name,
            "✅ Loaded" if exp_data['status'] == 'loaded' else "❌ Failed",
            f"{size_mb:.2f} MB"
        ])
    
    # Print table
    col_widths = [
        max(len(headers[0]), max(len(row[0]) for row in rows)),
        max(len(headers[1]), max(len(row[1]) for row in rows)),
        max(len(headers[2]), max(len(row[2]) for row in rows)),
    ]
    
    # Header
    print("\n" + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)))
    print("-" * (sum(col_widths) + 6))
    
    # Rows
    for row in rows:
        print(" | ".join(str(cell).ljust(w) for cell, w in zip(row, col_widths)))
    
    print("\n" + "=" * 70)


def extract_model_info(exp_dir: str):
    """Extract and display information about saved models."""
    
    print("\n" + "=" * 70)
    print("🏆 BEST MODELS")
    print("=" * 70)
    
    # Look for .npz files (saved models)
    model_files = glob.glob(os.path.join(exp_dir, "../results/*.npz"))
    
    if not model_files:
        print("\nNo saved models found.")
        return
    
    print(f"\nFound {len(model_files)} saved model(s):")
    
    for model_file in sorted(model_files):
        print(f"\n📦 {os.path.basename(model_file)}")
        try:
            data = np.load(model_file)
            params = data['params']
            
            # Calculate sparsity
            near_zero = np.abs(params) < 1e-10
            sparsity = np.mean(near_zero)
            
            print(f"   - Parameters: {len(params):,}")
            print(f"   - Sparsity: {sparsity:.4f} ({sparsity*100:.2f}%)")
            print(f"   - Non-zero params: {np.sum(~near_zero):,}")
            print(f"   - Size: {os.path.getsize(model_file) / (1024*1024):.2f} MB")
            
        except Exception as e:
            print(f"   ⚠️  Error loading model: {e}")
    
    print("\n" + "=" * 70)


def compare_configurations(results: Dict[str, Dict]):
    """Compare different omega/beta configurations."""
    
    print("\n" + "=" * 70)
    print("⚖️  CONFIGURATION COMPARISON")
    print("=" * 70)
    
    # Group experiments by type
    configs = {
        'baseline': None,
        'balanced': None,
        'performance_focused': None,
        'sparsity_focused': None,
        'extreme_sparsity': None,
    }
    
    for exp_name in results:
        name_lower = exp_name.lower()
        for config_type in configs:
            if config_type in name_lower:
                configs[config_type] = exp_name
                break
    
    print("\n📋 Detected configurations:")
    for config_type, exp_name in configs.items():
        if exp_name:
            print(f"   ✅ {config_type.replace('_', ' ').title()}: {exp_name}")
        else:
            print(f"   ❌ {config_type.replace('_', ' ').title()}: Not found")
    
    print("\n💡 Recommendations:")
    print("   - Compare baseline vs sparsity-aware configurations")
    print("   - Analyze trade-off between performance and sparsity")
    print("   - Identify the Pareto frontier")
    
    print("\n" + "=" * 70)


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_sparsity_experiments.py <experiment_directory>")
        print("\nExample:")
        print("  python analyze_sparsity_experiments.py experiments/sparsity_comparison_20251011_120000")
        sys.exit(1)
    
    exp_dir = sys.argv[1]
    
    if not os.path.isdir(exp_dir):
        print(f"❌ Error: Directory not found: {exp_dir}")
        sys.exit(1)
    
    print("\n" + "╔" + "=" * 68 + "╗")
    print("║" + " Sparsity-Aware Natural Niches Experiment Analyzer ".center(68) + "║")
    print("╚" + "=" * 68 + "╝")
    print(f"\n📁 Analyzing directory: {exp_dir}")
    
    # Load and analyze results
    results = analyze_experiment(exp_dir)
    
    # Print summary
    print_summary(results)
    
    # Compare configurations
    compare_configurations(results)
    
    # Extract model information
    extract_model_info(exp_dir)
    
    print("\n✅ Analysis complete!")
    print("\n💡 Next steps:")
    print("   1. Visualize results: python plot_sparsity_comparison.py " + exp_dir)
    print("   2. Inspect individual logs: cat " + exp_dir + "/<experiment>.log")
    print("   3. Load results in Python:")
    print("      >>> import pickle")
    print(f"      >>> with open('{exp_dir}/<experiment>.pkl', 'rb') as f:")
    print("      ...     data = pickle.load(f)")
    print("")


if __name__ == "__main__":
    main()

