import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"
import pickle
import argparse

from natural_niches_fn import run_natural_niches


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("--pop_size", type=int, default=20)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--store_train_results", action="store_true")
    parser.add_argument("--no_crossover", action="store_true")
    parser.add_argument("--no_splitpoint", action="store_true")
    parser.add_argument("--no_matchmaker", action="store_true")
    parser.add_argument("--use_pre_trained", action="store_true")
    parser.add_argument(
        "--method",
        type=str,
        default="natural_niches",
        choices=[
            "natural_niches",
            "natural_niches_sharded",
            "ga",
            "map_elites",
            "cma_es",
            "brute_force",
        ],
    )
    parser.add_argument("--total_forward_passes", type=int, default=50000)
    # add model1_path and model2_path
    parser.add_argument("--model1_path", type=str, default="models/wizardmath_7b")
    parser.add_argument("--model2_path", type=str, default="models/agentevol-7b")
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable multi-node distributed execution via torch.distributed",
    )
    parser.add_argument(
        "--archive_backend",
        type=str,
        default="gpu",
        choices=["gpu", "cpu"],
        help="Where to place the evolutionary archive (gpu or cpu).",
    )
    parser.add_argument(
        "--debug_models",
        action="store_true",
        help="Use lightweight BERTOverflow checkpoints for debugging.",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    args_dict = vars(args)
    method = args_dict.pop("method")
    file_name = method

    if method == "natural_niches":
        if args.no_crossover:
            file_name += "_no_crossover"
        if args.no_splitpoint and not args.no_crossover:
            file_name += "_no_splitpoint"
        if args.no_matchmaker and not args.no_crossover:
            file_name += "_no_matchmaker"
        if args.debug_models:
            args_dict["model1_path"] = "models/MathBERT"
            args_dict["model2_path"] = "models/BERTOverflow"

        if args.debug_models:
            args_dict["model1_path"] = "models/MathBERT"
            args_dict["model2_path"] = "models/BERTOverflow"

        results = run_natural_niches(**args_dict)
    elif method == "ga":
        if not args.no_matchmaker:
            print("WARNING: GA can't use matchmaker. Disabling it.")
            args_dict["no_matchmaker"] = True
        if args.no_crossover:
            file_name += "_no_crossover"
        if args.no_splitpoint and not args.no_crossover:
            file_name += "_no_splitpoint"
        results = run_natural_niches(**args_dict, alpha=0.0)
    elif method == "natural_niches_sharded":
        from natural_niches_sharded import run_natural_niches_sharded

        if args.debug_models:
            args_dict["model1_path"] = "models/BERTOverflow"
            args_dict["model2_path"] = "SparseFusion/models/BERTOverflow"

        results = run_natural_niches_sharded(**args_dict)
    elif method == "brute_force":
        assert args.use_pre_trained, "Brute force requires pre-trained models"
        results = run_brute_force()
    else:
        raise NotImplementedError("Method not implemented")

    if args.use_pre_trained:
        file_name += "_pre_trained"

    with open(f"results/{file_name}.pkl", "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    main()
