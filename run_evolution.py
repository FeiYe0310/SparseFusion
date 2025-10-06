"""
This script serves as the main entry point for running the Natural Niches
evolutionary algorithm on Large Language Models (LLMs).

It should be run from within the 'natural_niches' directory.

Before running, ensure that:
1. You have two pre-trained Hugging Face models saved locally.
2. The paths to these models are correctly specified in the `model_1_path`
   and `model_2_path` variables below, relative to this directory.

This script should be launched using `torchrun` for multi-GPU execution.
For example, to run on 4 GPUs:
JAX_PLATFORM_NAME=cpu torchrun --nproc_per_node=4 run_evolution.py
"""

import argparse
from pathlib import Path

# Since this script is now inside the 'natural_niches' directory,
# we can import directly from the other files in this directory.
from natural_niches_fn import run_natural_niches
from config import DEFAULT_MODEL_1, DEFAULT_MODEL_2, MODELS_DIR


ALLOWED_MODEL_NAMES = [
    "BERTOverflow",
    "MathBERT",
    "Qwen2.5-Coder-1.5B-Instruct",
    "Qwen2.5-Math-1.5B-Instruct",
    "agentevol-7b",
    "wizardmath_7b",
]
ALLOWED_MODEL_CHOICES = [f"models/{name}" for name in ALLOWED_MODEL_NAMES]
ALLOWED_MODEL_MAP = {
    choice: str((Path(MODELS_DIR) / name).resolve())
    for choice, name in zip(ALLOWED_MODEL_CHOICES, ALLOWED_MODEL_NAMES)
}


def _resolve_model_path(raw_path: str) -> str:
    if raw_path in ALLOWED_MODEL_MAP:
        resolved = Path(ALLOWED_MODEL_MAP[raw_path])
        if not resolved.exists():
            raise ValueError(
                f"Model directory '{resolved}' does not exist."
            )
        return str(resolved)

    candidate = Path(raw_path).resolve()
    for choice, target in ALLOWED_MODEL_MAP.items():
        if candidate == Path(target):
            if not candidate.exists():
                raise ValueError(
                    f"Model directory '{candidate}' does not exist."
                )
            return str(candidate)

    raise ValueError(
        "Model path must match one of the allowed options: "
        + ", ".join(ALLOWED_MODEL_CHOICES)
    )


def _default_choice_from(path: str) -> str:
    try:
        rel = Path(path).resolve().relative_to(Path(MODELS_DIR).resolve().parent)
    except ValueError:
        return ALLOWED_MODEL_CHOICES[0]
    rel_str = str(rel)
    return rel_str if rel_str in ALLOWED_MODEL_CHOICES else ALLOWED_MODEL_CHOICES[0]


def main():
    """
    Main function to configure and run the Natural Niches LLM evolution.
    """
    parser = argparse.ArgumentParser(
        description="Run Natural Niches evolution for LLMs."
    )
    default_model1_choice = _default_choice_from(DEFAULT_MODEL_1)
    default_model2_choice = _default_choice_from(DEFAULT_MODEL_2)
    parser.add_argument(
        "--model1_path",
        type=str,
        default=default_model1_choice,
        choices=ALLOWED_MODEL_CHOICES,
        help=(
            "Path to the first base model under 'models/'. Choices: "
            + ", ".join(ALLOWED_MODEL_CHOICES)
        ),
    )
    parser.add_argument(
        "--model2_path",
        type=str,
        default=default_model2_choice,
        choices=ALLOWED_MODEL_CHOICES,
        help=(
            "Path to the second base model under 'models/'. Choices: "
            + ", ".join(ALLOWED_MODEL_CHOICES)
        ),
    )
    parser.add_argument(
        "--runs", type=int, default=1, help="Number of independent evolutionary runs."
    )
    parser.add_argument(
        "--pop_size",
        type=int,
        default=16,
        help="Number of individuals in the population.",
    )
    parser.add_argument(
        "--total_forward_passes",
        type=int,
        default=100,
        help="Total number of child model evaluations.",
    )
    parser.add_argument(
        "--archive_backend",
        type=str,
        default="gpu",
        choices=["gpu", "cpu"],
        help="Where to store the evolution archive (gpu or cpu).",
    )
    parser.add_argument(
        "--debug_models",
        action="store_true",
        help="Use lightweight BERTOverflow checkpoints for debugging.",
    )
    parser.add_argument(
        "--use_sharded_archive",
        action="store_true",
        help="Run experimental sharded archive implementation.",
    )

    args = parser.parse_args()

    # --- Algorithm Hyperparameters ---
    try:
        model1_path = _resolve_model_path(args.model1_path)
        model2_path = _resolve_model_path(args.model2_path)
    except ValueError as exc:
        parser.error(str(exc))

    config = {
        "runs": args.runs,
        "pop_size": args.pop_size,
        "total_forward_passes": args.total_forward_passes,
        "store_train_results": True,
        "no_matchmaker": False,
        "no_crossover": False,
        "no_splitpoint": False,
        "alpha": 1.0,
        "model1_path": model1_path,
        "model2_path": model2_path,
        "distributed": True,
        "archive_backend": args.archive_backend,
    }

    if args.debug_models:
        config["model1_path"] = _resolve_model_path("models/MathBERT")
        config["model2_path"] = _resolve_model_path("models/BERTOverflow")

    print("--- Starting Natural Niches Evolution for LLMs ---")
    print("Configuration:")
    config_for_print = config.copy()
    config_for_print["use_sharded_archive"] = args.use_sharded_archive
    for key, value in config_for_print.items():
        print(f"  - {key}: {value}")
    print("--------------------------------------------------")

    # --- Run Evolution ---
    if args.use_sharded_archive:
        from natural_niches_sharded import run_natural_niches_sharded

        sharded_kwargs = config.copy()
        sharded_kwargs.pop("archive_backend", None)
        results = run_natural_niches_sharded(**sharded_kwargs)
    else:
        results = run_natural_niches(**config)

    # --- Print Final Results ---
    print("\n--- Evolution Finished ---")
    # The 'results' variable is a list of dictionaries, one for each run.
    for i, run_result in enumerate(results):
        if "test_values" in run_result and run_result["test_values"]:
            best_test_acc = max(run_result["test_values"])
            print(f"Run {i+1}:")
            print(f"  - Best Test Accuracy on GSM8K (50 samples): {best_test_acc:.4f}")
        else:
            print(f"Run {i+1}: No test results were recorded.")


if __name__ == "__main__":
    main()
