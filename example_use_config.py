"""
配置系统使用示例

展示如何使用新的配置系统来启动实验。
"""

from config.presets import load_preset, print_config, PRESETS
from natural_niches_sparsity_aware_fn import run_natural_niches_sparsity_aware


def main():
    print("🎯 SparseFusion 配置系统示例")
    print(f"\n可用预设: {', '.join(PRESETS.keys())}")
    
    # 示例1: 使用预设配置
    print("\n" + "=" * 70)
    print("示例1: 使用预设配置 (multi_task)")
    print("=" * 70)
    
    config = load_preset("multi_task")
    print_config(config)
    
    # 示例2: 修改预设配置
    print("\n" + "=" * 70)
    print("示例2: 修改预设配置")
    print("=" * 70)
    
    config = load_preset("default")
    
    # 修改训练参数
    config["training"].pop_size = 24
    config["training"].total_forward_passes = 20000
    
    # 修改评估参数
    config["evaluation"].batch_size = 24
    config["evaluation"].eval_subset_size = 20
    
    # 启用动态稀疏度
    config["sparsity"].use_dynamic_sparsity = True
    config["sparsity"].sparsity_min = 0.2
    config["sparsity"].sparsity_max = 0.7
    
    # 启用多任务
    config["task"].use_mbpp_eval = True
    config["task"].gsm8k_weight = 0.6
    config["task"].mbpp_weight = 0.4
    
    print_config(config)
    
    # 示例3: 实际运行（注释掉，避免意外运行）
    print("\n" + "=" * 70)
    print("示例3: 运行实验（已注释）")
    print("=" * 70)
    print("""
# 解析配置
model_cfg = config["model"]
training_cfg = config["training"]
eval_cfg = config["evaluation"]
sparsity_cfg = config["sparsity"]
task_cfg = config["task"]

# 运行实验
results = run_natural_niches_sparsity_aware(
    # Training params
    runs=training_cfg.runs,
    pop_size=training_cfg.pop_size,
    total_forward_passes=training_cfg.total_forward_passes,
    no_crossover=training_cfg.no_crossover,
    no_splitpoint=training_cfg.no_splitpoint,
    no_matchmaker=training_cfg.no_matchmaker,
    distributed=training_cfg.distributed,
    archive_backend=training_cfg.archive_backend,
    store_train_results=training_cfg.store_train_results,
    log_sparsity_stats=training_cfg.log_sparsity_stats,
    
    # Model params
    model1_path=model_cfg.model1_path,
    model2_path=model_cfg.model2_path,
    use_pre_trained=model_cfg.use_pre_trained,
    
    # Sparsity params
    alpha=sparsity_cfg.alpha,
    omega=sparsity_cfg.omega,
    beta=sparsity_cfg.beta,
    tau=sparsity_cfg.tau,
    epsilon=sparsity_cfg.epsilon,
    pruning_sparsity=sparsity_cfg.pruning_sparsity,
    pruning_method=sparsity_cfg.pruning_method,
    use_dynamic_sparsity=sparsity_cfg.use_dynamic_sparsity,
    sparsity_min=sparsity_cfg.sparsity_min,
    sparsity_max=sparsity_cfg.sparsity_max,
    sparsity_t0=sparsity_cfg.sparsity_t0,
    sparsity_t_mult=sparsity_cfg.sparsity_t_mult,
    
    # Evaluation params
    eval_subset_size=eval_cfg.eval_subset_size,
    eval_on_test_subset=eval_cfg.eval_on_test_subset,
    eval_subset_size_gsm8k=eval_cfg.eval_subset_size_gsm8k,
    eval_subset_size_mbpp=eval_cfg.eval_subset_size_mbpp,
    test_eval_subset_size=eval_cfg.test_eval_subset_size,
    gsm8k_qwen_chat=eval_cfg.gsm8k_qwen_chat,
    gsm8k_few_shot_k=eval_cfg.gsm8k_few_shot_k,
    gsm8k_few_shot_split=eval_cfg.gsm8k_few_shot_split,
    mbpp_qwen_chat=eval_cfg.mbpp_qwen_chat,
    mbpp_few_shot_k=eval_cfg.mbpp_few_shot_k,
    mbpp_few_shot_split=eval_cfg.mbpp_few_shot_split,
    
    # Task params
    use_bfcl_eval=task_cfg.use_bfcl_eval,
    use_mbpp_eval=task_cfg.use_mbpp_eval,
    use_mult4_eval=task_cfg.use_mult4_eval,
    use_mult5_eval=task_cfg.use_mult5_eval,
    use_bool_eval=task_cfg.use_bool_eval,
    gsm8k_weight=task_cfg.gsm8k_weight,
    bfcl_weight=task_cfg.bfcl_weight,
    mbpp_weight=task_cfg.mbpp_weight,
    mult4_weight=task_cfg.mult4_weight,
    mult5_weight=task_cfg.mult5_weight,
    bool_weight=task_cfg.bool_weight,
    bfcl_data_path=task_cfg.bfcl_data_path,
    mbpp_data_path=task_cfg.mbpp_data_path,
)

print(f"✅ 实验完成！最佳得分: {results['best_fitness']}")
    """)
    
    print("\n✅ 示例完成！查看 config/ 目录了解更多配置选项。")


if __name__ == "__main__":
    main()

