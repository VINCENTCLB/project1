import numpy as np
import data_preprocessing as data_processing
from implementations import ridge_regression


SEED = 433
LAMBDAS = (0.0001, 0.001, 0.01, 0.1, 1.0)


def stratified_split(labels, seed=SEED):
    """Return disjoint 60/20/20 training, development, and holdout indices."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.all(np.isin(labels, (-1, 1))):
        raise ValueError("Labels must be a one-dimensional array of -1 and +1")
    rng = np.random.default_rng(seed)
    partitions = [[], [], []]
    for label in (-1, 1):
        indices = rng.permutation(np.flatnonzero(labels == label))
        if len(indices) < 5:
            raise ValueError("At least five examples of each class are required")
        train_end = int(0.6 * len(indices))
        dev_end = int(0.8 * len(indices))
        for part, selected in zip(
            partitions,
            (indices[:train_end], indices[train_end:dev_end], indices[dev_end:]),
        ):
            part.append(selected)
    return tuple(rng.permutation(np.concatenate(part)) for part in partitions)


def metrics(labels, predictions):
    """Return positive-class metrics and confusion-matrix counts."""
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    if labels.shape != predictions.shape or labels.ndim != 1 or not len(labels):
        raise ValueError("Labels and predictions must be nonempty matching vectors")
    if not np.all(np.isin(labels, (-1, 1))) or not np.all(np.isin(predictions, (-1, 1))):
        raise ValueError("Labels and predictions must contain only -1 and +1")
    positive, predicted_positive = labels == 1, predictions == 1
    tp = int(np.count_nonzero(positive & predicted_positive))
    fp = int(np.count_nonzero(~positive & predicted_positive))
    fn = int(np.count_nonzero(positive & ~predicted_positive))
    tn = int(np.count_nonzero(~positive & ~predicted_positive))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "accuracy": float(np.mean(labels == predictions)),
        "precision": precision,
        "recall": recall,
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def choose_threshold(labels, scores):
    """Choose a threshold using development labels only, maximizing F1."""
    thresholds = np.unique(np.append(np.quantile(scores, np.linspace(0.65, 0.99, 35)), 0.0))
    return max(
        ((metrics(labels, np.where(scores >= threshold, 1, -1))["f1"], float(threshold))
         for threshold in thresholds),
        key=lambda pair: pair[0],
    )


def run_experiments(features, labels, preprocessing="compare", degree=1,
                    lambdas=LAMBDAS, seed=SEED):
    """Select settings on development data, then evaluate the holdout once.

    Raw rows are split BEFORE preprocessing. For each method, pipeline() fits
    its imputer, feature selection, clipping, and scaling on training rows only.
    Holdout labels are never used to select preprocessing, lambda, or threshold.
    """
    if features.ndim != 2 or len(features) != len(labels):
        raise ValueError("Feature rows and labels must match")
    if not lambdas or any(not np.isfinite(value) or value <= 0 for value in lambdas):
        raise ValueError("Ridge lambdas must be finite and strictly positive")
    if degree < 1:
        raise ValueError("Polynomial degree must be at least one")
    methods = ("standardize", "all") if preprocessing == "compare" else (preprocessing,)
    train_idx, dev_idx, holdout_idx = stratified_split(labels, seed)
    train_raw, dev_raw = features[train_idx], features[dev_idx]
    y_train, y_dev = labels[train_idx], labels[dev_idx]
    print(f"Split: train={len(train_idx)}, development={len(dev_idx)}, holdout={len(holdout_idx)}")

    trials = []
    best = None
    for method in methods:
        print(f"Preprocessing training and development rows: {method}")
        x_train, x_dev = data_processing.pipeline(
            train_raw, y_train, dev_raw, method=method, degree=degree
        )
        for lambda_ in lambdas:
            weights, _ = ridge_regression(y_train, x_train, lambda_)
            scores = x_dev @ weights
            f1, threshold = choose_threshold(y_dev, scores)
            dev_metrics = metrics(y_dev, np.where(scores >= threshold, 1, -1))
            trial = {
                "preprocessing": method, "lambda": float(lambda_),
                "threshold": threshold, "development": dev_metrics,
            }
            trials.append(trial)
            print(f"method={method} lambda={lambda_:g} dev_f1={f1:.4f} threshold={threshold:.6f}")
            if best is None or f1 > best["development"]["f1"]:
                best = trial
        del x_train, x_dev

    # Refit the selected training-only transformation for the unseen holdout.
    x_train, x_holdout = data_processing.pipeline(
        train_raw, y_train, features[holdout_idx],
        method=best["preprocessing"], degree=degree,
    )
    weights, _ = ridge_regression(y_train, x_train, best["lambda"])
    predicted = np.where(x_holdout @ weights >= best["threshold"], 1, -1)
    holdout = metrics(labels[holdout_idx], predicted)
    baseline = metrics(labels[holdout_idx], np.full(len(holdout_idx), -1))
    print(f"Selected: {best['preprocessing']}, lambda={best['lambda']:g}, threshold={best['threshold']:.6f}")
    print(f"Holdout: accuracy={holdout['accuracy']:.4f}, precision={holdout['precision']:.4f}, recall={holdout['recall']:.4f}, F1={holdout['f1']:.4f}")
    print(f"Majority baseline: accuracy={baseline['accuracy']:.4f}, F1={baseline['f1']:.4f}")
    return {
        "seed": seed, "degree": degree, "model": "ridge_regression",
        "rows": int(len(labels)), "raw_features": int(features.shape[1]),
        "positive_rate": float(np.mean(labels == 1)),
        "split_sizes": {"train": len(train_idx), "development": len(dev_idx), "holdout": len(holdout_idx)},
        "selected": best, "trials": trials,
        "holdout": holdout, "majority_baseline": baseline,
    }


def make_test_predictions(features, labels, test_features, results):
    """Refit the selected configuration on all labeled rows and predict test IDs.
    All preprocessing parameters are now estimated on the full labeled set.
    The development-selected lambda and threshold stay fixed.
    """
    selected = results["selected"]
    x_train, x_test = data_processing.pipeline(
        features, labels, test_features,
        method=selected["preprocessing"], degree=results["degree"],
    )
    weights, _ = ridge_regression(labels, x_train, selected["lambda"])
    return np.where(x_test @ weights >= selected["threshold"], 1, -1)
