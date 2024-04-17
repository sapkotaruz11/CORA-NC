import argparse
import os

default_model = "SAGE"


def validate_model(value):
    valid_models = ["GCN", "GAT", "SAGE"]
    if value not in valid_models:
        raise argparse.ArgumentTypeError(
            f"Invalid model '{value}'. Must be one of: {', '.join(valid_models)}"
        )
    return value


def kfold_validation(value):
    ivalue = int(value)
    if ivalue < 2:
        raise argparse.ArgumentTypeError(
            "%s is not a valid option. Cross validation folds should be greater than or equal to 2"
            % value
        )
    return ivalue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Framework for Node Classification task on CORA dataset."
    )

    parser.add_argument(
        "--model",
        type=validate_model,
        default=default_model,
        help=f"Specify the model to use (default: {default_model})",
    )

    parser.add_argument(
        "--k_folds",
        type=kfold_validation,
        default=10,
        help="Number of runs to execute (default: 10). Must be equal or greater than 2",
    )

    args = parser.parse_args()
    print(
        f"Running Node Classification task on  Cora Citation Dataset with {args.model} model and {args.k_folds} fold data split."
    )
    from src.runner import run

    run(model_name=args.model, k_folds=args.k_folds)
