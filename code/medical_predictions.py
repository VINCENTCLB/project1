from helpers import *
from implementations import *
from visualizations import *
import data_preprocessing as data_processing
from model_experiments import LAMBDAS, SEED, make_test_predictions, run_experiments
from pathlib import Path
import argparse
import json
import time


def parse_args():
    """Builds and returns the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Run the medical prediction pipeline with specified parameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data directory argument with default value.
    parser.add_argument("--data-path", type=Path, default=Path(__file__).resolve().parents[1] / "data/dataset")

    # Subsample to test the pipeline with a smaller dataset.
    parser.add_argument("--subsample", action="store_true", help="Subsample the dataset for testing purposes.")

    # Compare evaluates standardize and all on development data.
    parser.add_argument("--preprocessing", type=str, default="standardize", choices=["standardize", "best_features", "all", "none", "compare"], help="Preprocessing option; compare tests standardize and all.")

    # Polynomial degree argument with default value.
    parser.add_argument("-d", "--degree", type=int, default=1, help="Polynomial feature expansion degree.")

    # Retained for compatibility with the original CLI; ridge has a closed-form solution.
    parser.add_argument("-lr", type=float, default=0.01, help="Reserved for gradient-based models; unused by ridge regression.")

    # Output a submission CSV file for the predictions for the full test set.
    parser.add_argument("--submission", action="store_true", help="Refit on all labeled data and create a local prediction CSV.")
    parser.add_argument("--lambdas", type=float, nargs="+", default=list(LAMBDAS), help="Ridge regularization values to test.")
    parser.add_argument("--seed", type=int, default=SEED, help="Seed for the stratified data split.")
    parser.add_argument("--plots", action="store_true", help="Save a development F1 plot.")
    parser.add_argument("--show-statistics", action="store_true", help="Display the original training/test statistics plots after model selection.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data/dataset/integrated_results", help="Local directory for metrics, plots, and predictions.")
    return parser.parse_args()


def run_pipeline(data_path: str, subsample: bool, degree: int, lr: float,
                 submission: bool, preprocessing: str, lambdas=LAMBDAS,
                 seed=SEED, plots=False, show_statistics=False, output_dir=None):
    """Core application logic separated from CLI parsing.

    The lr argument is retained for compatibility; the ridge solver does not use it.
    """
    if subsample and submission:
        raise ValueError("Disable --subsample before generating a complete submission")

    time_start_loading = time.time()
    print("Loading CSVs with the course helper...")
    X_tr, X_te, Y_tr, train_ids, test_ids = load_csv_data(data_path, sub_sample=subsample)
    time_end_loading = time.time()
    print(f"Time taken to load data: {time_end_loading - time_start_loading:.2f} seconds")
    print(f"Features: train={X_tr.shape}, test={X_te.shape}")

    # Kora model training: split RAW data before fitting preprocessing.
    # run_experiments calls data_processing.pipeline on training/development
    # portions, then evaluates the selected configuration on the holdout.
    results = run_experiments(
        X_tr, Y_tr, preprocessing=preprocessing, degree=degree,
        lambdas=lambdas, seed=seed,
    )
    results["subsample"] = subsample

    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[1] / "data/dataset/integrated_results"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "metrics.json"
    with report_path.open("w") as file:
        json.dump(results, file, indent=2, allow_nan=False)
        file.write("\n")
    print(f"Experiment metrics: {report_path}")

    if plots:
        plot_validation_results(results["trials"], output_dir / "validation_f1.png")

    # Keep the original statistics plots, but make them optional. This full-data
    # preprocessing happens AFTER holdout evaluation and does not select settings.
    if show_statistics:
        time_start_preprocessing = time.time()
        x_train, x_test = data_processing.pipeline(
            X_tr, Y_tr, X_te,
            method=results["selected"]["preprocessing"], degree=degree,
        )
        time_end_preprocessing = time.time()
        print(f"Time taken for statistics preprocessing: {time_end_preprocessing - time_start_preprocessing:.2f} seconds")

        mean, std, min_val, max_val, median_val = data_processing.statistics(x_train)
        mean_test, std_test, min_val_test, max_val_test, median_val_test = data_processing.statistics(x_test)
        print("Training set statistics for the first 10 features:")
        plot_statistics(mean[:10], std[:10], min_val[:10], max_val[:10], median_val[:10], [f"Feature {i}" for i in range(x_train.shape[1])][:10])
        print("Test set statistics for the first 10 features:")
        plot_statistics(mean_test[:10], std_test[:10], min_val_test[:10], max_val_test[:10], median_val_test[:10], [f"Feature {i}" for i in range(x_test.shape[1])][:10])
        del x_train, x_test

    if submission:
        print("Refitting the selected configuration on all labeled rows...")
        predictions = make_test_predictions(X_tr, Y_tr, X_te, results)
        if len(predictions) != len(test_ids):
            raise ValueError("Prediction count does not match test IDs")
        submission_path = output_dir / "submission.csv"
        create_csv_submission(test_ids, predictions, str(submission_path))
        print(f"Submission: {submission_path} ({len(predictions)} rows)")

    print(f"Total runtime: {time.time() - time_start_loading:.2f} seconds")
    return results


def main():
    """Program entry point."""
    args = parse_args()
    run_pipeline(
        data_path=args.data_path, subsample=args.subsample, degree=args.degree,
        lr=args.lr, submission=args.submission, preprocessing=args.preprocessing,
        lambdas=args.lambdas, seed=args.seed, plots=args.plots,
        show_statistics=args.show_statistics, output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
