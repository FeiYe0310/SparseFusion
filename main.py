import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"
import pickle
import argparse

from natural_niches_fn import run_natural_niches
from map_elites_fn import run_map_elites
from cma_es_fn import run_cma_es
from brute_force_fn import run_brute_force


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
        choices=["natural_niches", "ga", "map_elites", "cma_es", "brute_force"],
    )
    parser.add_argument("--total_forward_passes", type=int, default=50000)
    # add model1_path and model2_path
    parser.add_argument("--model1_path", type=str, default="models/wizardmath_7b")
    parser.add_argument("--model2_path", type=str, default="models/agentevol-7b")

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
