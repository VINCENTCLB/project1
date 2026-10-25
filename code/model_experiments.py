import numpy as np
import data_preprocessing as data_processing
import project_root  # noqa: F401  (puts the repository root on sys.path)
from implementations import ridge_regression


SEED = 433
# Cross-validation on the 789-feature "clean" design matrix shows F1 falling
# monotonically as lambda grows (0.4379 at 1e-4 against 0.3989 at 10), so the
# grid is weighted toward small values. 1.0 is retained as evidence that
# regularisation does not help at this sample size.
LAMBDAS = (0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0)
# Cross-validation puts the useful range at 3-7 (mean dev F1 0.4373-0.4384,
# a flat plateau) and rejects heavier weights outright: w=20 scores -0.0044
# against w=10 (t=-3.25) and w=30 scores -0.0097 (t=-13.67). w=10 lost to w=7
# in all five folds, so it is excluded; w=1 is kept as the unweighted baseline.
POS_WEIGHTS = (1.0, 3.0, 5.0, 7.0)
COMPARED_METHODS = ("standardize", "all", "clean")
# Selecting on one 20% development split is noisy enough to matter here: the
# dev-F1 differences across the useful part of the grid are around 0.001 while
# the fold-to-fold spread is 0.003, so the single split can prefer a setting
# that loses on every fold of a cross-validated comparison. Five folds is the
# default when cross-validation is requested.
DEFAULT_FOLDS = 5


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


def stratified_folds(labels, indices, folds, seed=SEED):
    """Split indices into stratified folds, preserving the class ratio.

    Both classes are permuted and dealt out separately, so each fold holds
    roughly the same 8.8% positive rate as the whole sample. With 5 folds the
    rarest class still contributes about 5,800 positives per fold, which is
    what keeps the F1 estimate from being dominated by fold composition.
    """
    if folds < 2:
        raise ValueError("Cross-validation needs at least two folds")
    labels, indices = np.asarray(labels), np.asarray(indices)
    rng = np.random.default_rng(seed)
    buckets = [[] for _ in range(folds)]
    for label in (-1, 1):
        part = rng.permutation(indices[labels[indices] == label])
        if len(part) < folds:
            raise ValueError("Each class needs at least one row per fold")
        for position, chunk in enumerate(np.array_split(part, folds)):
            buckets[position].append(chunk)
    return [np.concatenate(bucket) for bucket in buckets]


def metrics(labels, predictions):
    """Return positive-class metrics and confusion-matrix counts."""
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    if labels.shape != predictions.shape or labels.ndim != 1 or not len(labels):
        raise ValueError("Labels and predictions must be nonempty matching vectors")
    if not np.all(np.isin(labels, (-1, 1))) or not np.all(
        np.isin(predictions, (-1, 1))
    ):
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
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def choose_threshold(labels, scores):
    """Choose a threshold using development labels only, maximizing F1."""
    thresholds = np.unique(
        np.append(np.quantile(scores, np.linspace(0.5, 0.995, 100)), 0.0)
    )
    return max(
        (
            (
                metrics(labels, np.where(scores >= threshold, 1, -1))["f1"],
                float(threshold),
            )
            for threshold in thresholds
        ),
        key=lambda pair: pair[0],
    )


def flag_rate(scores, threshold):
    """Fraction of rows a threshold would flag as positive."""
    return float(np.mean(np.asarray(scores) >= threshold))


def threshold_at_flag_rate(scores, rate):
    """Score cutoff that flags the given fraction of rows.

    Transferring a decision boundary as a flag rate rather than as a raw score
    is scale invariant. That matters for the final refit: training on all
    labelled rows instead of 60% of them shifts the score distribution, so a
    numeric cutoff chosen on development scores no longer flags the intended
    fraction, whereas a rate does.
    """
    rate = min(max(float(rate), 0.0), 1.0)
    return float(np.quantile(np.asarray(scores), 1.0 - rate))


def weighted_ridge(y, tx, lambda_, pos_weight=1.0, block_rows=None):
    """Ridge regression with the positive class upweighted by pos_weight.

    Minimising sum_i s_i (y_i - x_i^T w)^2 / 2N with an L2 penalty gives the
    weighted normal equations (X^T S X + 2 N lambda I) w = X^T S y, with
    S = diag(s). Scaling every row by sqrt(s_i) makes the unweighted solver
    produce exactly that solution, so the course implementation is reused
    unchanged rather than duplicated here.

    Without this, 91% of the rows are negatives and so 91% of the gradient
    signal that sets the direction of w comes from classifying negatives, which
    biases the fit against the rare positive class.

    block_rows switches to accumulating the same normal equations in row blocks.
    That avoids holding a second full copy of the design matrix, which matters
    for the final refit: at 328,135 rows and 789 columns the scaled copy alone
    is 2 GB. The result is identical up to floating-point summation order, which
    the test suite asserts.
    """
    if block_rows is None:
        if pos_weight == 1.0:
            return ridge_regression(y, tx, lambda_)
        root = np.sqrt(np.where(y == 1, pos_weight, 1.0))
        return ridge_regression(root * y, root[:, None] * tx, lambda_)

    n_samples, n_features = tx.shape
    scale = np.where(y == 1, float(pos_weight), 1.0)
    gram = np.zeros((n_features, n_features))
    rhs = np.zeros(n_features)
    for start in range(0, n_samples, block_rows):
        block = tx[start : start + block_rows]
        block_scale = scale[start : start + block_rows]
        gram += block.T @ (block_scale[:, None] * block)
        rhs += block.T @ (block_scale * y[start : start + block_rows])
    gram += 2 * n_samples * lambda_ * np.eye(n_features)
    weights = np.linalg.solve(gram, rhs)
    residual = y - tx @ weights
    return weights, float(np.mean(scale * residual**2) / 2)


def _evaluate_fold(
    features,
    labels,
    train_idx,
    score_idx,
    method,
    lambdas,
    pos_weights,
    degree,
    feature_names,
    clean_options,
    seed,
):
    """Score every (lambda, weight) on one fold, with a nested threshold.

    The threshold is chosen on an inner split of the fold's training rows and
    transferred to the scored rows as a flag rate, so no label of the scored
    rows informs any decision. Returns {(lambda, weight): (f1, flag_rate)}.
    """
    inner_train, inner_val = train_test_indices(labels, train_idx, seed)
    rest = np.concatenate([inner_val, score_idx])
    x_train, x_rest = data_processing.pipeline(
        features[inner_train],
        labels[inner_train],
        features[rest],
        method=method,
        degree=degree,
        feature_names=feature_names,
        clean_options=clean_options,
    )
    x_val, x_score = x_rest[: len(inner_val)], x_rest[len(inner_val) :]
    scored = {}
    for lambda_ in lambdas:
        for pos_weight in pos_weights:
            weights, _ = weighted_ridge(
                labels[inner_train], x_train, lambda_, pos_weight
            )
            val_scores = x_val @ weights
            _, threshold = choose_threshold(labels[inner_val], val_scores)
            rate = flag_rate(val_scores, threshold)
            scores = x_score @ weights
            cutoff = threshold_at_flag_rate(scores, rate)
            predicted = np.where(scores >= cutoff, 1, -1)
            scored[(lambda_, pos_weight)] = (
                metrics(labels[score_idx], predicted)["f1"],
                rate,
            )
    return scored


def train_test_indices(labels, indices, seed=SEED, fraction=0.75):
    """Stratified split of indices into a larger and a smaller part."""
    labels, indices = np.asarray(labels), np.asarray(indices)
    rng = np.random.default_rng(seed)
    larger, smaller = [], []
    for label in (-1, 1):
        part = rng.permutation(indices[labels[indices] == label])
        cut = int(fraction * len(part))
        larger.append(part[:cut])
        smaller.append(part[cut:])
    return np.concatenate(larger), np.concatenate(smaller)


def select_by_cross_validation(
    features,
    labels,
    pool,
    methods,
    folds=DEFAULT_FOLDS,
    degree=1,
    lambdas=LAMBDAS,
    pos_weights=POS_WEIGHTS,
    feature_names=None,
    clean_options=None,
    seed=SEED,
):
    """Rank every (method, lambda, weight) by mean F1 over stratified folds.

    Returns (best_trial, trials). Each trial records the per-fold F1 values so
    a paired comparison between two settings is possible after the fact, which
    an aggregate mean alone does not support.
    """
    fold_indices = stratified_folds(labels, pool, folds, seed)
    trials = []
    best = None
    for method in methods:
        print(f"Cross-validating {method} over {folds} folds")
        collected = {}
        width = None
        for position in range(folds):
            score_idx = fold_indices[position]
            train_idx = np.concatenate(
                [fold_indices[other] for other in range(folds) if other != position]
            )
            scored = _evaluate_fold(
                features,
                labels,
                train_idx,
                score_idx,
                method,
                lambdas,
                pos_weights,
                degree,
                feature_names,
                clean_options,
                seed,
            )
            for key, value in scored.items():
                collected.setdefault(key, []).append(value)
            print(f"  fold {position + 1}/{folds} done")
        for (lambda_, pos_weight), values in collected.items():
            f1_values = [value[0] for value in values]
            rates = [value[1] for value in values]
            trial = {
                "preprocessing": method,
                "lambda": float(lambda_),
                "pos_weight": float(pos_weight),
                "threshold": 0.0,
                "flag_rate": float(np.mean(rates)),
                "folds": folds,
                "fold_f1": [float(value) for value in f1_values],
                "development": {
                    "f1": float(np.mean(f1_values)),
                    "f1_std": float(np.std(f1_values)),
                },
            }
            trials.append(trial)
            if best is None or trial["development"]["f1"] > best["development"]["f1"]:
                best = trial
        for trial in sorted(
            (t for t in trials if t["preprocessing"] == method),
            key=lambda t: -t["development"]["f1"],
        )[:3]:
            print(
                f"  method={method} lambda={trial['lambda']:g} "
                f"pos_weight={trial['pos_weight']:g} "
                f"cv_f1={trial['development']['f1']:.4f} "
                f"+-{trial['development']['f1_std']:.4f}"
            )
    return best, trials


def run_experiments(
    features,
    labels,
    preprocessing="compare",
    degree=1,
    lambdas=LAMBDAS,
    seed=SEED,
    pos_weights=POS_WEIGHTS,
    feature_names=None,
    clean_options=None,
    folds=1,
):
    """Select settings on development data, then evaluate the holdout once.

    Raw rows are split BEFORE preprocessing. For each method, pipeline() fits
    its imputer, feature selection, clipping, and scaling on training rows only.
    Holdout labels are never used to select preprocessing, lambda, the class
    weight, or the threshold.
    """
    if features.ndim != 2 or len(features) != len(labels):
        raise ValueError("Feature rows and labels must match")
    if not lambdas or any(not np.isfinite(value) or value < 0 for value in lambdas):
        raise ValueError("Ridge lambdas must be finite and non-negative")
    if not pos_weights or any(not np.isfinite(w) or w <= 0 for w in pos_weights):
        raise ValueError("Class weights must be finite and strictly positive")
    if degree < 1:
        raise ValueError("Polynomial degree must be at least one")
    if folds != 1 and folds < 2:
        raise ValueError("folds must be 1 for a single split, or at least 2")
    methods = COMPARED_METHODS if preprocessing == "compare" else (preprocessing,)
    train_idx, dev_idx, holdout_idx = stratified_split(labels, seed)
    if folds != 1:
        return _run_cross_validated(
            features,
            labels,
            methods,
            train_idx,
            dev_idx,
            holdout_idx,
            folds,
            degree,
            lambdas,
            pos_weights,
            feature_names,
            clean_options,
            seed,
        )
    train_raw, dev_raw = features[train_idx], features[dev_idx]
    y_train, y_dev = labels[train_idx], labels[dev_idx]
    print(
        f"Split: train={len(train_idx)}, development={len(dev_idx)}, holdout={len(holdout_idx)}"
    )

    trials = []
    best = None
    for method in methods:
        print(f"Preprocessing training and development rows: {method}")
        x_train, x_dev = data_processing.pipeline(
            train_raw,
            y_train,
            dev_raw,
            method=method,
            degree=degree,
            feature_names=feature_names,
            clean_options=clean_options,
        )
        for lambda_ in lambdas:
            for pos_weight in pos_weights:
                weights, _ = weighted_ridge(y_train, x_train, lambda_, pos_weight)
                scores = x_dev @ weights
                f1, threshold = choose_threshold(y_dev, scores)
                dev_metrics = metrics(y_dev, np.where(scores >= threshold, 1, -1))
                trial = {
                    "preprocessing": method,
                    "lambda": float(lambda_),
                    "pos_weight": float(pos_weight),
                    "features": int(x_train.shape[1]),
                    "threshold": threshold,
                    "flag_rate": flag_rate(scores, threshold),
                    "development": dev_metrics,
                }
                trials.append(trial)
                print(
                    f"method={method} lambda={lambda_:g} pos_weight={pos_weight:g} "
                    f"dev_f1={f1:.4f} threshold={threshold:.6f}"
                )
                if best is None or f1 > best["development"]["f1"]:
                    best = trial
        del x_train, x_dev

    # Refit the selected training-only transformation for the unseen holdout.
    x_train, x_holdout = data_processing.pipeline(
        train_raw,
        y_train,
        features[holdout_idx],
        method=best["preprocessing"],
        degree=degree,
        feature_names=feature_names,
        clean_options=clean_options,
    )
    weights, _ = weighted_ridge(y_train, x_train, best["lambda"], best["pos_weight"])
    predicted = np.where(x_holdout @ weights >= best["threshold"], 1, -1)
    holdout = metrics(labels[holdout_idx], predicted)
    baseline = metrics(labels[holdout_idx], np.full(len(holdout_idx), -1))
    print(
        f"Selected: {best['preprocessing']}, lambda={best['lambda']:g}, "
        f"pos_weight={best['pos_weight']:g}, threshold={best['threshold']:.6f}"
    )
    print(
        f"Holdout: accuracy={holdout['accuracy']:.4f}, precision={holdout['precision']:.4f}, "
        f"recall={holdout['recall']:.4f}, F1={holdout['f1']:.4f}"
    )
    print(
        f"Majority baseline: accuracy={baseline['accuracy']:.4f}, F1={baseline['f1']:.4f}"
    )
    return {
        "seed": seed,
        "degree": degree,
        "model": "ridge_regression",
        "rows": int(len(labels)),
        "raw_features": int(features.shape[1]),
        "positive_rate": float(np.mean(labels == 1)),
        "split_sizes": {
            "train": len(train_idx),
            "development": len(dev_idx),
            "holdout": len(holdout_idx),
        },
        "clean_options": clean_options or {},
        "selected": best,
        "trials": trials,
        "holdout": holdout,
        "majority_baseline": baseline,
    }


def _run_cross_validated(
    features,
    labels,
    methods,
    train_idx,
    dev_idx,
    holdout_idx,
    folds,
    degree,
    lambdas,
    pos_weights,
    feature_names,
    clean_options,
    seed,
):
    """Select by cross-validation over train+development, then score holdout.

    Selection uses the 80% of rows outside the holdout, so the chosen setting
    is backed by five estimates instead of one, and the final model is fitted
    on all of them rather than on 60%. The holdout is still touched once, and
    its labels inform nothing: the decision boundary transfers as the flag rate
    the folds selected, which needs only the holdout score distribution.
    """
    pool = np.concatenate([train_idx, dev_idx])
    best, trials = select_by_cross_validation(
        features,
        labels,
        pool,
        methods,
        folds=folds,
        degree=degree,
        lambdas=lambdas,
        pos_weights=pos_weights,
        feature_names=feature_names,
        clean_options=clean_options,
        seed=seed,
    )
    x_pool, x_holdout = data_processing.pipeline(
        features[pool],
        labels[pool],
        features[holdout_idx],
        method=best["preprocessing"],
        degree=degree,
        feature_names=feature_names,
        clean_options=clean_options,
    )
    weights, _ = weighted_ridge(
        labels[pool], x_pool, best["lambda"], best["pos_weight"], block_rows=20000
    )
    scores = x_holdout @ weights
    threshold = threshold_at_flag_rate(scores, best["flag_rate"])
    best = {**best, "threshold": float(threshold)}
    predicted = np.where(scores >= threshold, 1, -1)
    holdout = metrics(labels[holdout_idx], predicted)
    baseline = metrics(labels[holdout_idx], np.full(len(holdout_idx), -1))
    print(
        f"Selected by {folds}-fold CV: {best['preprocessing']}, "
        f"lambda={best['lambda']:g}, pos_weight={best['pos_weight']:g}, "
        f"cv_f1={best['development']['f1']:.4f}, flag_rate={best['flag_rate']:.4f}"
    )
    print(
        f"Holdout: accuracy={holdout['accuracy']:.4f}, precision={holdout['precision']:.4f}, "
        f"recall={holdout['recall']:.4f}, F1={holdout['f1']:.4f}"
    )
    print(
        f"Majority baseline: accuracy={baseline['accuracy']:.4f}, F1={baseline['f1']:.4f}"
    )
    return {
        "seed": seed,
        "degree": degree,
        "model": "ridge_regression",
        "selection": f"{folds}-fold cross-validation",
        "rows": int(len(labels)),
        "raw_features": int(features.shape[1]),
        "positive_rate": float(np.mean(labels == 1)),
        "split_sizes": {
            "selection_pool": int(len(pool)),
            "holdout": int(len(holdout_idx)),
        },
        "clean_options": clean_options or {},
        "selected": best,
        "trials": trials,
        "holdout": holdout,
        "majority_baseline": baseline,
    }


def make_test_predictions(
    features,
    labels,
    test_features,
    results,
    feature_names=None,
    clean_options=None,
    threshold_mode="flag_rate",
    block_rows=20000,
):
    """Refit the selected configuration on all labeled rows and predict test IDs.
    All preprocessing parameters are now estimated on the full labeled set.
    The development-selected lambda and class weight stay fixed.

    threshold_mode controls how the decision boundary transfers to the refit
    model. "flag_rate" reproduces the fraction of rows the selected threshold
    flagged on development scores, which is invariant to the score rescaling
    that refitting on more rows causes. "score" reuses the raw cutoff.
    """
    if threshold_mode not in ("flag_rate", "score"):
        raise ValueError("threshold_mode must be 'flag_rate' or 'score'")
    selected = results["selected"]
    x_train, x_test = data_processing.pipeline(
        features,
        labels,
        test_features,
        method=selected["preprocessing"],
        degree=results["degree"],
        feature_names=feature_names,
        clean_options=clean_options
        if clean_options is not None
        else results.get("clean_options"),
    )
    weights, _ = weighted_ridge(
        labels,
        x_train,
        selected["lambda"],
        selected.get("pos_weight", 1.0),
        block_rows=block_rows,
    )
    scores = x_test @ weights
    rate = selected.get("flag_rate")
    if threshold_mode == "flag_rate" and rate is not None:
        threshold = threshold_at_flag_rate(scores, rate)
    else:
        threshold = selected["threshold"]
    return np.where(scores >= threshold, 1, -1)
