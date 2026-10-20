import argparse
import csv
from pathlib import Path

import numpy as np

from helpers import create_csv_submission


SEED = 433
LAMBDAS = (0.0001, 0.001, 0.01, 0.1, 1.0)


def read_features(path, expected_ids=None):
    """Read a feature CSV with missing entries as NaN using bounded memory."""
    with path.open(newline="") as file:
        reader = csv.reader(file)
        header = next(reader)
        names = header[1:]
        rows = sum(1 for _ in reader)

    features = np.empty((rows, len(names)), dtype=np.float32)
    ids = np.empty(rows, dtype=np.int64)
    with path.open(newline="") as file:
        reader = csv.reader(file)
        next(reader)
        for row_index, row in enumerate(reader):
            if len(row) != len(names) + 1:
                raise ValueError(f"Unexpected number of columns at row {row_index + 2}")
            ids[row_index] = int(row[0])
            features[row_index] = np.fromiter(
                (float(value) if value else np.nan for value in row[1:]),
                dtype=np.float32,
                count=len(names),
            )
    if expected_ids is not None and not np.array_equal(ids, expected_ids):
        raise ValueError("Feature and label IDs do not match")
    return features, ids, names


def read_labels(path):
    """Read labels and IDs without assuming the CSV row order is correct."""
    with path.open(newline="") as file:
        reader = csv.reader(file)
        next(reader)
        pairs = [(int(row[0]), int(row[1])) for row in reader]
    ids = np.fromiter((pair[0] for pair in pairs), dtype=np.int64)
    labels = np.fromiter((pair[1] for pair in pairs), dtype=np.int8)
    if not np.all(np.isin(labels, (-1, 1))):
        raise ValueError("Labels must be -1 or +1")
    return ids, labels


def stratified_split(labels, seed=SEED):
    """Return disjoint 60/20/20 train, development, and holdout indices."""
    rng = np.random.default_rng(seed)
    partitions = [[], [], []]
    for label in (-1, 1):
        indices = rng.permutation(np.flatnonzero(labels == label))
        train_end = int(0.6 * len(indices))
        dev_end = int(0.8 * len(indices))
        for part, selected in zip(
            partitions,
            (indices[:train_end], indices[train_end:dev_end], indices[dev_end:]),
        ):
            part.append(selected)
    return tuple(rng.permutation(np.concatenate(part)) for part in partitions)


def fit_preprocessor(features, add_missing_indicators):
    """Estimate imputation and scaling parameters on training rows only."""
    missing = np.isnan(features)
    observed = (~missing).sum(axis=0)
    totals = np.nansum(features, axis=0, dtype=np.float64)
    means = (totals / np.maximum(observed, 1)).astype(np.float32)
    filled = np.array(features, copy=True)
    np.copyto(filled, np.broadcast_to(means, filled.shape), where=missing)
    scales = np.std(filled, axis=0, dtype=np.float64).astype(np.float32)
    scales[scales < 1e-6] = 1.0
    missing_rate = missing.mean(axis=0)
    indicator_columns = np.flatnonzero(
        (missing_rate >= 0.01) & (missing_rate <= 0.99)
    ) if add_missing_indicators else np.array([], dtype=int)
    return means, scales, indicator_columns


def transform(features, preprocessor):
    """Impute, standardize, optionally add missingness flags, and add bias."""
    means, scales, indicator_columns = preprocessor
    missing = np.isnan(features)
    result = np.array(features, copy=True)
    np.copyto(result, np.broadcast_to(means, result.shape), where=missing)
    result -= means
    result /= scales
    if len(indicator_columns):
        result = np.column_stack(
            (result, missing[:, indicator_columns].astype(np.float32))
        )
    return np.column_stack((np.ones(len(result), dtype=np.float32), result))


def ridge_weights(gram, target_product, lambda_):
    """Solve ridge normal equations for half-MSE plus lambda * ||w||^2.

    The intercept is not penalized. This is an experiment-local baseline;
    the team's public ridge_regression implementation can replace it later.
    """
    penalty = np.eye(len(target_product), dtype=np.float64) * (2 * lambda_)
    penalty[0, 0] = 0.0
    return np.linalg.solve(gram + penalty, target_product).astype(np.float32)


def metrics(labels, predictions):
    """Return accuracy, positive-class precision, recall, and F1."""
    positive = labels == 1
    predicted_positive = predictions == 1
    true_positive = np.count_nonzero(positive & predicted_positive)
    false_positive = np.count_nonzero(~positive & predicted_positive)
    false_negative = np.count_nonzero(positive & ~predicted_positive)
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return (float(np.mean(labels == predictions)), precision, recall, f1)


def choose_threshold(labels, scores):
    """Choose a development-set threshold for positive-class F1."""
    thresholds = np.quantile(scores, np.linspace(0.65, 0.99, 35))
    return max(
        ((metrics(labels, np.where(scores >= threshold, 1, -1))[3], threshold)
         for threshold in thresholds),
        key=lambda pair: pair[0],
    )


def run(data_dir, submission_path=None):
    label_ids, labels = read_labels(data_dir / "y_train.csv")
    features, _, names = read_features(data_dir / "x_train.csv", label_ids)
    train_idx, dev_idx, holdout_idx = stratified_split(labels)
    print(f"Rows: {len(labels)}; features: {len(names)}; positive rate: {np.mean(labels == 1):.4f}")
    print(f"Split: train={len(train_idx)}, development={len(dev_idx)}, holdout={len(holdout_idx)}")

    best = None
    for add_indicators in (False, True):
        preprocessor = fit_preprocessor(features[train_idx], add_indicators)
        x_train = transform(features[train_idx], preprocessor)
        x_dev = transform(features[dev_idx], preprocessor)
        n_train = len(train_idx)
        gram = (x_train.T @ x_train).astype(np.float64) / n_train
        target_product = (x_train.T @ labels[train_idx].astype(np.float32)).astype(np.float64) / n_train
        for lambda_ in LAMBDAS:
            weights = ridge_weights(gram, target_product, lambda_)
            f1, threshold = choose_threshold(labels[dev_idx], x_dev @ weights)
            candidate = (f1, lambda_, float(threshold), add_indicators)
            print(f"indicators={add_indicators} lambda={lambda_:g} dev_f1={f1:.4f} threshold={threshold:.4f}")
            if best is None or candidate[0] > best[0]:
                best = candidate
        del x_train, x_dev

    _, lambda_, threshold, add_indicators = best
    preprocessor = fit_preprocessor(features[train_idx], add_indicators)
    x_train = transform(features[train_idx], preprocessor)
    gram = (x_train.T @ x_train).astype(np.float64) / len(train_idx)
    target_product = (x_train.T @ labels[train_idx].astype(np.float32)).astype(np.float64) / len(train_idx)
    weights = ridge_weights(gram, target_product, lambda_)
    del x_train
    holdout_scores = transform(features[holdout_idx], preprocessor) @ weights
    holdout_predictions = np.where(holdout_scores >= threshold, 1, -1)
    baseline = metrics(labels[holdout_idx], np.full(len(holdout_idx), -1))
    result = metrics(labels[holdout_idx], holdout_predictions)
    print(f"Selected: indicators={add_indicators}, lambda={lambda_:g}, threshold={threshold:.4f}")
    print(f"Majority baseline holdout: accuracy={baseline[0]:.4f}, F1={baseline[3]:.4f}")
    print(f"Model holdout: accuracy={result[0]:.4f}, precision={result[1]:.4f}, recall={result[2]:.4f}, F1={result[3]:.4f}")

    if submission_path is not None:
        final_preprocessor = fit_preprocessor(features, add_indicators)
        x_full = transform(features, final_preprocessor)
        gram = (x_full.T @ x_full).astype(np.float64) / len(labels)
        target_product = (x_full.T @ labels.astype(np.float32)).astype(np.float64) / len(labels)
        final_weights = ridge_weights(gram, target_product, lambda_)
        del x_full, features
        x_test, test_ids, test_names = read_features(data_dir / "x_test.csv")
        if names != test_names:
            raise ValueError("Train and test feature columns differ")
        test_scores = transform(x_test, final_preprocessor) @ final_weights
        predictions = np.where(test_scores >= threshold, 1, -1)
        submission_path.parent.mkdir(parents=True, exist_ok=True)
        create_csv_submission(test_ids, predictions, str(submission_path))
        print(f"Submission: {submission_path} ({len(predictions)} rows)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data/dataset")
    parser.add_argument("--submission", type=Path, help="Optional local CSV output; do not commit this file")
    args = parser.parse_args()
    run(args.data_dir, args.submission)


if __name__ == "__main__":
    main()
