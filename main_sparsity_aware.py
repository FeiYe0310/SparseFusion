import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"
import pickle
import argparse

from natural_niches_sparsity_aware_fn import run_natural_niches_sparsity_aware


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run Natural Niches with Sparsity-Aware Selection"
    )

    # Original arguments
    parser.add_argument("--pop_size", type=int, default=20,
                        help="Population size for the archive")
    parser.add_argument("--runs", type=int, default=10,
                        help="Number of independent runs")
    parser.add_argument("--store_train_results", action="store_true",
                        help="Store training results")
    parser.add_argument("--no_crossover", action="store_true",
                        help="Disable crossover operation")
    parser.add_argument("--no_splitpoint", action="store_true",
                        help="Disable splitpoint in crossover")
    parser.add_argument("--no_matchmaker", action="store_true",
                        help="Disable matchmaker for parent selection")
    parser.add_argument("--use_pre_trained", action="store_true",
                        help="Use pre-trained models")
    parser.add_argument("--total_forward_passes", type=int, default=50000,
                        help="Total number of forward passes")
    parser.add_argument("--model1_path", type=str, default="models/wizardmath_7b",
                        help="Path to first model")
    parser.add_argument("--model2_path", type=str, default="models/agentevol-7b",
                        help="Path to second model")
    parser.add_argument("--distributed", action="store_true",
                        help="Enable multi-node distributed execution")
    parser.add_argument("--archive_backend", type=str, default="gpu",
                        choices=["gpu", "cpu"],
                        help="Where to place the evolutionary archive")
    parser.add_argument("--debug_models", action="store_true",
                        help="Use lightweight BERTOverflow checkpoints for debugging")

    # Sparsity-Aware specific arguments
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Fitness normalization exponent (default: 1.0)")
    parser.add_argument("--omega", type=float, default=0.5,
                        help="Weight for fitness component (default: 0.5)")
    parser.add_argument("--beta", type=float, default=0.5,
                        help="Weight for sparsity component (default: 0.5)")
    parser.add_argument("--tau", type=float, default=1.0,
                        help="Softmax temperature for sparsity scores (default: 1.0)")
    parser.add_argument("--epsilon", type=float, default=1e-10,
                        help="Threshold for considering parameters as zero (default: 1e-10)")
    parser.add_argument("--pruning_sparsity", type=float, default=0.0,
                        help="Target sparsity for Wanda pruning (0.0 = disabled, default: 0.0)")
    parser.add_argument("--pruning_method", type=str, default="wanda",
                        choices=["wanda", "magnitude"],
                        help="Pruning method to use (default: wanda)")
    parser.add_argument("--log_sparsity_stats", action="store_true",
                        help="Log detailed sparsity statistics during evolution")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Output directory for results")
    
    # 🚀 加速参数
    parser.add_argument("--eval_subset_size", type=int, default=None,
                        help="Number of samples to evaluate per iteration (None=all data, 30=sample 30 points per iter for speedup)")
    
    # 🎯 BFCL多任务评估参数
    parser.add_argument("--use_bfcl_eval", action="store_true",
                        help="Enable BFCL function calling evaluation")
    parser.add_argument("--bfcl_data_path", type=str, 
                        default="bfcl/data/bfcl_test_200.json",
                        help="Path to BFCL test dataset")
    parser.add_argument("--gsm8k_weight", type=float, default=0.5,
                        help="Weight for GSM8K task in multi-task learning")
    parser.add_argument("--bfcl_weight", type=float, default=0.5,
                        help="Weight for BFCL task in multi-task learning")
    
    # 🔄 动态稀疏度调度参数（Cosine Annealing with Warm Restarts）
    parser.add_argument("--use_dynamic_sparsity", action="store_true",
                        help="Enable dynamic sparsity scheduling (overrides --pruning_sparsity)")
    parser.add_argument("--sparsity_min", type=float, default=0.1,
                        help="Minimum sparsity value for dynamic scheduling (default: 0.1)")
    parser.add_argument("--sparsity_max", type=float, default=0.6,
                        help="Maximum sparsity value for dynamic scheduling (default: 0.6)")
    parser.add_argument("--sparsity_t0", type=int, default=100,
                        help="Number of iterations in the first cycle (default: 100)")
    parser.add_argument("--sparsity_t_mult", type=int, default=2,
                        help="Cycle length multiplier after each restart (default: 2, i.e., doubling; 1=fixed)")

    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # Print configuration
    print("\n" + "=" * 70)
    print("🔬 Natural Niches with Sparsity-Aware Selection")
    print("=" * 70)
    print(f"Population size: {args.pop_size}")
    print(f"Total forward passes: {args.total_forward_passes}")
    print(f"Runs: {args.runs}")
    if args.eval_subset_size:
        print(f"🚀 Eval subset size: {args.eval_subset_size} (加速模式)")
    if args.use_bfcl_eval:
        print(f"🎯 BFCL Evaluation ENABLED")
        print(f"  GSM8K weight: {args.gsm8k_weight}")
        print(f"  BFCL weight: {args.bfcl_weight}")
        print(f"  BFCL data: {args.bfcl_data_path}")
    print(f"\nSparsity-Aware Parameters:")
    print(f"  ω (omega): {args.omega} - Fitness weight")
    print(f"  β (beta): {args.beta} - Sparsity weight")
    print(f"  τ (tau): {args.tau} - Softmax temperature")
    print(f"  α (alpha): {args.alpha} - Fitness normalization")
    print(f"  ε (epsilon): {args.epsilon} - Zero threshold")
    print(f"\nPruning Parameters:")
    if args.use_dynamic_sparsity:
        print(f"  🔄 Dynamic Sparsity ENABLED")
        print(f"  Sparsity range: [{args.sparsity_min:.2f}, {args.sparsity_max:.2f}]")
        print(f"  First cycle: {args.sparsity_t0} iterations")
        print(f"  Cycle multiplier: {args.sparsity_t_mult}x")
        print(f"  Method: {args.pruning_method}")
    elif args.pruning_sparsity > 0:
        print(f"  🔪 Pruning ENABLED")
        print(f"  Target sparsity: {args.pruning_sparsity}")
        print(f"  Method: {args.pruning_method}")
    else:
        print(f"  Pruning DISABLED (set --pruning_sparsity > 0 or --use_dynamic_sparsity to enable)")
    print(f"\nExperimental Setup:")
    print(f"  Crossover: {'Disabled' if args.no_crossover else 'Enabled'}")
    print(f"  Splitpoint: {'Disabled' if args.no_splitpoint else 'Enabled'}")
    print(f"  Matchmaker: {'Disabled' if args.no_matchmaker else 'Enabled'}")
    print(f"  Archive backend: {args.archive_backend}")
    print(f"  Distributed: {args.distributed}")
    print("=" * 70 + "\n")
    
    # Override model paths if using debug models
    if args.debug_models:
        args.model1_path = "models/MathBERT"
        args.model2_path = "models/BERTOverflow"
        print("⚠️  Using debug models (BERTOverflow)")

    # Run evolution
    results = run_natural_niches_sparsity_aware(
        runs=args.runs,
        pop_size=args.pop_size,
        total_forward_passes=args.total_forward_passes,
        store_train_results=args.store_train_results,
        no_matchmaker=args.no_matchmaker,
        no_crossover=args.no_crossover,
        no_splitpoint=args.no_splitpoint,
        alpha=args.alpha,
        omega=args.omega,
        beta=args.beta,
        tau=args.tau,
        epsilon=args.epsilon,
        pruning_sparsity=args.pruning_sparsity,
        pruning_method=args.pruning_method,
        use_pre_trained=args.use_pre_trained,
        model1_path=args.model1_path,
        model2_path=args.model2_path,
        distributed=args.distributed,
        archive_backend=args.archive_backend,
        log_sparsity_stats=args.log_sparsity_stats,
        eval_subset_size=args.eval_subset_size,  # 🚀 加速参数
        use_bfcl_eval=args.use_bfcl_eval,  # 🎯 BFCL评估
        bfcl_data_path=args.bfcl_data_path,
        gsm8k_weight=args.gsm8k_weight,
        bfcl_weight=args.bfcl_weight,
        # 🔄 动态稀疏度参数
        use_dynamic_sparsity=args.use_dynamic_sparsity,
        sparsity_min=args.sparsity_min,
        sparsity_max=args.sparsity_max,
        sparsity_t0=args.sparsity_t0,
        sparsity_t_mult=args.sparsity_t_mult,
    )

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate filename based on configuration
    filename_parts = ["sparsity_aware"]
    if args.use_pre_trained:
        filename_parts.append("pre_trained")
    if args.no_crossover:
        filename_parts.append("no_crossover")
    if args.no_splitpoint and not args.no_crossover:
        filename_parts.append("no_splitpoint")
    if args.no_matchmaker and not args.no_crossover:
        filename_parts.append("no_matchmaker")
    
    # Add parameter info
    filename_parts.append(f"w{args.omega:.2f}_b{args.beta:.2f}_t{args.tau:.2f}")
    
    # Add pruning info if enabled
    if args.pruning_sparsity > 0:
        filename_parts.append(f"prune_{args.pruning_method}_{args.pruning_sparsity:.2f}")
    
    base_filename = "_".join(filename_parts)
    
    # Save as pickle (complete data for analysis)
    pkl_path = os.path.join(args.output_dir, base_filename + ".pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\n✅ Results (pickle) saved to: {pkl_path}")
    
    # Save as JSON (human-readable logs)
    import json
    json_path = os.path.join(args.output_dir, base_filename + ".json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✅ Results (JSON) saved to: {json_path}")
    
    print("=" * 70)


if __name__ == "__main__":
    main()

