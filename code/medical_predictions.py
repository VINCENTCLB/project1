from helpers import *
from implementations import *
from visualizations import *
import data_preprocessing as data_processing
from pathlib import Path
import argparse
import sys
import time

def parse_args():
    """Builds and returns the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Run data processing pipeline with specified parameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # data directory argument with default value
    parser.add_argument("--data-path", type=Path, default=Path(__file__).resolve().parents[1] / "data/dataset")
    
    # subsample to test the pipeline with a smaller dataset
    parser.add_argument("--subsample", action="store_true", help="Subsample the dataset for testing purposes.")
    
    # preprocessing option argument with default value
    parser.add_argument("--preprocessing", type=str, default="standardize", choices=["standardize", "best_features", "all" ,"none"], help="Preprocessing option for the dataset.")
    
    # polynomial degree argument with default value
    parser.add_argument("-d", "--degree", type=int, default=1, help="Polynomial feature expansion degree.")

    # learning rate argument with default value
    parser.add_argument("-lr", type=float, default=0.01  , help="Learning rate for the model.")
    
    # output a submission CSV file for the predictions for the test set    
    parser.add_argument("--submission", action="store_true", help="Optional local CSV output for the predictions for the test set.")
    return parser.parse_args()


def run_pipeline(data_path: str, subsample: bool, degree: int, lr: float, submission: bool, preprocessing: str):
    """Core application logic separated from CLI parsing."""
    time_start_loading = time.time()
    X_tr, X_te, Y_tr, train_ids, test_ids = load_csv_data(data_path, sub_sample=subsample)
    time_end_loading = time.time()
    print(f"Time taken to load data: {time_end_loading - time_start_loading:.2f} seconds")
    
    # TODO is we use validation technic we should apply the pipeline on each fold, not on the whole training set. For now we just apply it on the whole training set.
    time_start_preprocessing = time.time()
    x_train, x_test = data_processing.pipeline(X_tr, Y_tr, X_te, method=preprocessing, degree=degree)
    time_end_preprocessing = time.time()
    print(f"Time taken for preprocessing: {time_end_preprocessing - time_start_preprocessing:.2f} seconds")
    
    mean, std, min_val, max_val, median_val = data_processing.statistics(x_train)
    mean_test, std_test, min_val_test, max_val_test, median_val_test = data_processing.statistics(x_test)
    print("Training set statistics for the first 10 features:")
    plot_statistics(mean[:10], std[:10], min_val[:10], max_val[:10], median_val[:10], [f"Feature {i}" for i in range(x_train.shape[1])][:10])
    print("Test set statistics for the first 10 features:")
    plot_statistics(mean_test[:10], std_test[:10], min_val_test[:10], max_val_test[:10], median_val_test[:10], [f"Feature {i}" for i in range(x_test.shape[1])][:10])
    
    # Kora model training
    



def main():
    """Program entry point."""
    args = parse_args()
    run_pipeline(data_path=args.data_path, subsample=args.subsample, degree=args.degree, lr=args.lr, submission=args.submission, preprocessing=args.preprocessing)


if __name__ == "__main__":
    main()
