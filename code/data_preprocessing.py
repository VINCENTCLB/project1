import numpy as np
from visualizations import plot_correlation_matrix, plot_target_distribution, plot_statistics



def statistics(data):
    """Per-feature statistics (ignoring NaNs). Each returned array has shape (d,)."""
    return (np.nanmean(data, axis=0), np.nanstd(data, axis=0), np.nanmin(data, axis=0),
            np.nanmax(data, axis=0), np.nanmedian(data, axis=0))
    
def fit_imputer(x_train):
    """Median of each feature, computed on the training set only."""
    fill = np.nanmedian(x_train, axis=0)
    return np.nan_to_num(fill, nan=0.0)  # features that are entirely NaN -> 0

def impute(x, fill):
    x = x.copy()
    rows, cols = np.where(np.isnan(x))
    x[rows, cols] = fill[cols]
    return x   

def standardize(x, mean=None, std=None):
    """Standardize each feature. Pass the TRAIN mean/std when transforming test data.

    >>> standardize(np.array([[1, 2], [3, 4], [5, 6]]))
    array([[-1.22474487, -1.22474487],
           [ 0.        ,  0.        ],
           [ 1.22474487,  1.22474487]])
    """
    mean = np.mean(x, axis=0) if mean is None else mean
    std = np.std(x, axis=0) if std is None else std
    std = np.where(std == 0, 1.0, std)  # avoid division by zero for constant features
    return (x - mean) / std


def build_poly_features(x, degree):
    """(N, d) -> (N, d*degree): [x, x^2, ..., x^degree] for every feature, no bias."""
    assert degree >= 1
    return np.hstack([x ** p for p in range(1, degree + 1)])

def add_bias(x):
    return np.hstack([np.ones((x.shape[0], 1)), x])


def fit_clip_bounds(x_train, bounds=(2.5, 97.5)):
    """Percentile bounds computed on train. Degenerate bounds (e.g. rare binary
    flags where lower == upper) are disabled so the feature is left untouched."""
    lower, upper = np.percentile(x_train, bounds, axis=0)
    degenerate = lower >= upper
    return np.where(degenerate, -np.inf, lower), np.where(degenerate, np.inf, upper)


def correlation_matrix(x_train):
    """Feature-feature correlation (d, d). Constant features get correlation 0."""
    return np.nan_to_num(np.corrcoef(x_train, rowvar=False), nan=0.0)

def target_correlation(x, y):
    """Pearson correlation between each feature and the target, shape (d,)."""
    y = np.asarray(y).ravel()
    xc, yc = x - x.mean(axis=0), y - y.mean()
    denom = np.sqrt((xc ** 2).sum(axis=0) * (yc ** 2).sum())
    return np.divide(xc.T @ yc, denom, out=np.zeros(x.shape[1]), where=denom > 0)

def best_features(corr_matrix, target_corr, threshold=0.5):
    """Greedy selection: go through features by decreasing |corr with target| and keep
    a feature only if it is not highly correlated (>= threshold) with one already kept."""
    kept = []
    for i in np.argsort(-np.abs(target_corr)):
        if all(abs(corr_matrix[i, j]) < threshold for j in kept):
            kept.append(i)
    return sorted(kept)


def target_distribution(y_train):
    """Dictionary {label: count} of the target variable."""
    unique, counts = np.unique(y_train, return_counts=True)
    return dict(zip(unique, counts))


def explore(x_train, y_train):
    x = impute(x_train, fit_imputer(x_train))
    names = [f"Feature {i}" for i in range(x.shape[1])]
    plot_statistics(*statistics(x_train), names)
    plot_correlation_matrix(correlation_matrix(x), feature_names=names)
    plot_target_distribution(target_distribution(y_train))



def pipeline(x_train, y_train, x_test, method="all", degree=1,
             bounds=(2.5, 97.5), corr_threshold=0.5):
    """Preprocess train and test data. Everything is fitted on x_train ONLY.

    Args:
        x_train: (N, d) training features (may contain NaN).
        y_train: (N,) training labels.
        x_test:  (M, d) test features (may contain NaN).
        method: "none", "standardize", "best_features" or "all".
        degree: polynomial degree (>= 1).

    Returns:
        x_train_processed, x_test_processed (with a bias column). Rows are never
        removed, so y_train is unchanged.
    """
    if method not in ("none", "standardize", "best_features", "all"):
        raise ValueError(f"Unknown method: {method}")
    assert degree >= 1

    # fill missing values with train medians
    fill = fit_imputer(x_train)
    x_tr, x_te = impute(x_train, fill), impute(x_test, fill)

    # selection of best features based on correlation with target and among themselves
    if method in ("best_features", "all"):
        selected = best_features(correlation_matrix(x_tr),
                                 target_correlation(x_tr, y_train), corr_threshold)
        x_tr, x_te = x_tr[:, selected], x_te[:, selected]

    # clipping outliers using train bounds, on both sets
    if method == "all":
        lower, upper = fit_clip_bounds(x_tr, bounds)
        x_tr, x_te = np.clip(x_tr, lower, upper), np.clip(x_te, lower, upper)

    # normalize with train statistics 
    if method in ("standardize", "all"):
        mean, std = x_tr.mean(axis=0), x_tr.std(axis=0)
        x_tr, x_te = standardize(x_tr, mean, std), standardize(x_te, mean, std)
        
    # polynomial expansion
    if degree > 1:
        x_tr, x_te = build_poly_features(x_tr, degree), build_poly_features(x_te, degree)
 

    return add_bias(x_tr), add_bias(x_te)