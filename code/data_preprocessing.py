import numpy as np
from visualizations import (
    plot_correlation_matrix,
    plot_target_distribution,
    plot_statistics,
)


def statistics(data):
    """Per-feature statistics (ignoring NaNs). Each returned array has shape (d,)."""
    return (
        np.nanmean(data, axis=0),
        np.nanstd(data, axis=0),
        np.nanmin(data, axis=0),
        np.nanmax(data, axis=0),
        np.nanmedian(data, axis=0),
    )


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
    return np.hstack([x**p for p in range(1, degree + 1)])


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
    denom = np.sqrt((xc**2).sum(axis=0) * (yc**2).sum())
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


# ----------------------------------------------------------------------------
# BRFSS-specific cleaning
#
# The columns are telephone-survey response codes, not measurements, so the raw
# values need decoding before a linear model can use them. Everything below is
# fitted on training rows only and then applied unchanged to dev/holdout/test.
# ----------------------------------------------------------------------------

# Survey administration fields: they describe how the interview happened, not
# the respondent. SEQNO and _PSU are record identifiers and the _*WT* fields are
# post-stratification weights, so any correlation they show is spurious.
ADMIN_COLUMNS = frozenset(
    {
        "IDATE",
        "IMONTH",
        "IDAY",
        "IYEAR",
        "FMONTH",
        "SEQNO",
        "_PSU",
        "CTELENUM",
        "CTELNUM1",
        "DISPCODE",
        "CELLFON3",
        "CELLFON2",
        "PVTRESD1",
        "PVTRESD2",
        "CCLGHOUS",
        "COLGHOUS",
        "STATERES",
        "LANDLINE",
        "CSTATE",
        "LADULT",
        "CADULT",
        "_STSTR",
        "_STRWT",
        "_RAWRAKE",
        "_WT2RAKE",
        "_LLCPWT",
        "_LLCPWT2",
        "_CLLCPWT",
        "_DUALUSE",
        "_DUALCOR",
        "QSTVER",
        "QSTLANG",
    }
)

# BRFSS non-answer codes, grouped by magnitude. Each entry is
# (codes meaning "don't know"/"refused", code meaning "none").
SENTINEL_TIERS = (((777, 999), 888), ((77, 99), 88), ((7, 9), None))

# Columns whose numeric codes carry no order, so that the numeric form the
# pipeline always keeps is meaningless for them and only the one-hot block
# carries usable information.
#
# Ranking candidates by absolute target correlation cannot find these: _STATE is
# a FIPS code, and its correlation with the label is 0.0057 purely because the
# numbering is arbitrary, not because CHD prevalence is flat across states. The
# same ranking spends slots on binary columns, where dropping the first level
# leaves a single indicator that is an affine function of the numeric column and
# so adds nothing at all.
NOMINAL_COLUMNS = frozenset(
    {
        "_STATE",
        "EMPLOY1",
        "MARITAL",
        "RENTHOM1",
        "MSCODE",
        "_RACE",
        "_RACEGR3",
        "_PRACE1",
        "_MRACE1",
        "_IMPRACE",
        "BPHIGH4",
        "DIABETE3",
        "_SMOKER3",
        "SXORIENT",
        "TRNSGNDR",
        "CRGVREL1",
        "CRGVPRB1",
    }
)

# Defaults chosen by paired 5-fold cross-validation inside the training pool,
# with nested selection of the class weight and threshold. See IMPROVEMENT_PLAN.md
# for the comparisons; the rejected alternatives were a wider one-hot block,
# more interaction pairs, dropping the clipping stage, and bagging.
CLEAN_OPTIONS = {
    "drop_admin": True,
    "sentinels": True,
    "indicators": True,
    "composite": True,
    "one_hot_top_k": 15,
    "max_levels": 12,
    # Both of these are off by default because paired 5-fold CV rejected them.
    # Excluding binary columns from the ranked one-hot selection measured
    # -0.0006 (t=-1.05) and -0.0003 (t=-0.97) at the two plateau settings, and
    # the curated nominal list -0.0010 to +0.0000. The slots the binaries
    # occupied were indeed carrying no information, but the 3-level columns
    # that replace them carry no more, and ridge already shrinks a collinear
    # indicator harmlessly rather than being damaged by it. Kept available and
    # tested, since the reasoning is sound and only the effect is absent.
    "min_levels": 2,
    "nominal": False,
    "nominal_max_levels": 60,
    "interactions_top_k": 20,
    "clip": True,
    "bounds": (2.5, 97.5),
}


def fit_sentinel_rules(x_train):
    """Decide, per column and per tier, which coded values are non-answers.

    A tier applies only when every remaining legitimate value lies strictly
    below that tier's codes. That guard is what keeps the rule from destroying
    real data: PHYSHLTH uses 7 and 9 as genuine day counts while also using
    77/88/99 as codes, and _STATE contains 9 as a FIPS code.

    Returns:
        dict mapping column index -> (codes to NaN, codes to set to zero).
    """
    rules = {}
    for col in range(x_train.shape[1]):
        values = x_train[:, col]
        values = values[~np.isnan(values)]
        if not len(values) or not np.allclose(values, np.round(values)):
            continue
        present = set(int(v) for v in np.unique(values))
        to_nan, to_zero = [], []
        for nan_codes, zero_code in SENTINEL_TIERS:
            tier = set(nan_codes) | ({zero_code} if zero_code else set())
            coded = present & tier
            legitimate = present - tier - set(to_nan) - set(to_zero)
            if coded and legitimate and max(legitimate) < min(coded):
                to_nan += [c for c in nan_codes if c in present]
                if zero_code and zero_code in present:
                    to_zero.append(zero_code)
        if to_nan or to_zero:
            rules[col] = (to_nan, to_zero)
    return rules


def apply_sentinel_rules(x, rules):
    """Map non-answer codes to NaN and 'none' codes to zero."""
    x = x.copy()
    for col, (to_nan, to_zero) in rules.items():
        values = x[:, col]
        if to_nan:
            values[np.isin(values, to_nan)] = np.nan
        for code in to_zero:
            values[values == code] = 0.0
        x[:, col] = values
    return x


def fit_dropped_columns(x_train, feature_names=None):
    """Indices of columns to discard: administrative fields and constants.

    Constant columns carry no information and are what makes np.corrcoef emit
    'invalid value encountered in divide'.
    """
    dropped = set()
    if feature_names is not None:
        names = np.asarray(feature_names)
        dropped |= set(
            int(j) for j in np.flatnonzero(np.isin(names, list(ADMIN_COLUMNS)))
        )
    for col in range(x_train.shape[1]):
        values = x_train[:, col]
        if len(np.unique(values[~np.isnan(values)])) <= 1:
            dropped.add(col)
    return sorted(dropped)


def fit_composite_columns(x_train):
    """Columns whose values are banded unit codes rather than quantities.

    BRFSS encodes several frequency questions as '1xx' / '2xx' / '3xx', where
    the leading digit selects the unit (per week vs per month, say) and the last
    two digits carry the count. Detection requires every legitimate value to sit
    in the 101-399 code space and at least two different bands to be present,
    which rules out ordinary numeric columns that merely happen to be large.
    """
    composite = []
    for col in range(x_train.shape[1]):
        values = x_train[:, col]
        values = values[~np.isnan(values)]
        if not len(values) or not np.allclose(values, np.round(values)):
            continue
        if values.min() < 101 or values.max() > 399:
            continue
        if len(np.unique(values // 100)) >= 2:
            composite.append(col)
    return composite


def split_composite(x, composite, bands=None):
    """Replace each banded code by its count, and return the band labels.

    The unit that each band stands for differs between columns, so instead of
    guessing a multiplier we hand the model the count and the band separately
    and let it learn a coefficient per band.

    Returns:
        (x with counts in place of codes, band label array of shape (N, len(composite))).
    """
    x = x.copy()
    labels = np.zeros((x.shape[0], len(composite)))
    for position, col in enumerate(composite):
        values = x[:, col]
        labels[:, position] = np.where(np.isnan(values), 0.0, values // 100)
        x[:, col] = values % 100
    return x, labels


def fit_nominal_levels(x_train, names, max_levels=60):
    """Levels for the hand-identified nominal columns, ignoring correlation.

    These are selected by meaning rather than by rank, so a nominal column is
    encoded whether or not its arbitrary numbering happens to correlate with the
    label, and whether or not it fits under the cap that bounds the automatic
    selection.
    """
    if names is None:
        return {}
    levels = {}
    for col, name in enumerate(names):
        if name in NOMINAL_COLUMNS:
            values = np.unique(x_train[:, col])
            values = values[~np.isnan(values)]
            if 2 <= len(values) <= max_levels:
                levels[col] = values
    return levels


def fit_categorical_levels(
    x_train, y_train, max_levels=12, top_k=30, skip=(), min_levels=2
):
    """Choose which low-cardinality columns to one-hot encode, and their levels.

    Every candidate is also kept in its numeric form by the caller, so an
    ordinal scale keeps its ordering while a nominal code gains one free
    parameter per level. Candidates are ranked by absolute target correlation
    and truncated at top_k to bound the width of the design matrix.

    min_levels excludes columns with too few distinct values to be worth a
    block. At the default of 3 that means binary columns: one_hot_blocks drops
    the first level, so a 2-level column contributes a single indicator that is
    an affine function of the numeric column already present, which is exactly
    collinear with it and carries no information. Nine of the fifteen slots went
    that way before this guard existed.
    """
    if top_k <= 0:
        return {}
    candidates = []
    for col in range(x_train.shape[1]):
        if col in set(skip):
            continue
        levels = np.unique(x_train[:, col])
        if min_levels <= len(levels) <= max_levels:
            candidates.append((col, levels))
    if not candidates:
        return {}
    columns = [col for col, _ in candidates]
    correlations = np.abs(target_correlation(x_train[:, columns], y_train))
    order = np.argsort(-correlations)[:top_k]
    return {candidates[i][0]: candidates[i][1] for i in order}


def one_hot_blocks(x, levels):
    """Indicator columns for the recorded levels, dropping each column's first
    level to stay linearly independent of the bias term. Values unseen during
    fitting become all-zero rows, which is the correct extrapolation."""
    if not levels:
        return None
    blocks = [
        (x[:, [col]] == np.asarray(values)[None, 1:]).astype(float)
        for col, values in sorted(levels.items())
        if len(values) > 1
    ]
    return np.hstack(blocks) if blocks else None


def fit_interaction_pairs(x_train, y_train, top_k=0):
    """Index pairs among the top_k features by absolute target correlation."""
    if top_k < 2:
        return []
    strongest = np.argsort(-np.abs(target_correlation(x_train, y_train)))[:top_k]
    return [
        (int(a), int(b)) for i, a in enumerate(strongest) for b in strongest[i + 1 :]
    ]


def interaction_block(x, pairs):
    """Pairwise products, which a purely additive model cannot represent."""
    if not pairs:
        return None
    return np.hstack([x[:, [a]] * x[:, [b]] for a, b in pairs])


def _clean_transform(x, state, y=None, fit=False):
    """Apply every clean-pipeline stage, fitting the parameters when fit=True."""
    options = state["options"]

    names = state.get("feature_names")
    if options["drop_admin"]:
        if fit:
            dropped = set(fit_dropped_columns(x, names))
            state["keep"] = [j for j in range(x.shape[1]) if j not in dropped]
        x = x[:, state["keep"]]
        if names is not None:
            names = np.asarray(names)[state["keep"]]

    if options["sentinels"]:
        if fit:
            state["sentinel_rules"] = fit_sentinel_rules(x)
        x = apply_sentinel_rules(x, state["sentinel_rules"])

    bands = None
    if options["composite"]:
        if fit:
            state["composite"] = fit_composite_columns(x)
        x, band_labels = split_composite(x, state["composite"])
        if state["composite"]:
            if fit:
                state["band_levels"] = {
                    position: np.unique(band_labels[:, position])
                    for position in range(band_labels.shape[1])
                }
            bands = one_hot_blocks(band_labels, state["band_levels"])

    indicators = None
    if options["indicators"]:
        missing = np.isnan(x).astype(float)
        if fit:
            state["indicator_keep"] = missing.std(axis=0) > 0
        indicators = missing[:, state["indicator_keep"]]

    if fit:
        state["fill"] = fit_imputer(x)
    x = impute(x, state["fill"])

    one_hot = None
    if options["one_hot_top_k"] or options["nominal"]:
        if fit:
            forced = (
                fit_nominal_levels(x, names, options["nominal_max_levels"])
                if options["nominal"]
                else {}
            )
            ranked = fit_categorical_levels(
                x,
                y,
                options["max_levels"],
                options["one_hot_top_k"],
                skip=tuple(forced),
                min_levels=options["min_levels"],
            )
            state["levels"] = {**forced, **ranked}
        one_hot = one_hot_blocks(x, state["levels"])

    if options["clip"]:
        if fit:
            state["clip"] = fit_clip_bounds(x, options["bounds"])
        x = np.clip(x, *state["clip"])

    if fit:
        state["scale"] = (x.mean(axis=0), x.std(axis=0))
    x = standardize(x, *state["scale"])

    interactions = None
    if options["interactions_top_k"]:
        if fit:
            state["pairs"] = fit_interaction_pairs(x, y, options["interactions_top_k"])
        interactions = interaction_block(x, state["pairs"])

    blocks = [x] + [
        b for b in (bands, one_hot, indicators, interactions) if b is not None
    ]
    return add_bias(np.hstack(blocks) if len(blocks) > 1 else blocks[0])


def clean_pipeline(x_train, y_train, x_test, feature_names=None, **options):
    """Decode the BRFSS codes, then impute, scale and expand the features.

    Fitted entirely on x_train: dropped columns, sentinel rules, composite
    bands, imputation medians, one-hot levels, clip bounds, scaling statistics
    and interaction pairs. Rows are never removed, so y_train is unchanged.

    Args:
        x_train: (N, d) training features, possibly containing NaN.
        y_train: (N,) training labels in {-1, +1}.
        x_test:  (M, d) features to transform with the fitted parameters.
        feature_names: (d,) column names; required to drop administrative
            columns by name, optional otherwise.
        **options: overrides for CLEAN_OPTIONS.

    Returns:
        x_train_processed, x_test_processed, both with a leading bias column.
    """
    unknown = set(options) - set(CLEAN_OPTIONS)
    if unknown:
        raise ValueError(f"Unknown clean options: {sorted(unknown)}")
    state = {
        "options": {**CLEAN_OPTIONS, **options},
        "feature_names": feature_names,
    }
    return (
        _clean_transform(x_train, state, y=y_train, fit=True),
        _clean_transform(x_test, state),
    )


def pipeline(
    x_train,
    y_train,
    x_test,
    method="all",
    degree=1,
    bounds=(2.5, 97.5),
    corr_threshold=0.5,
    feature_names=None,
    clean_options=None,
):
    """Preprocess train and test data. Everything is fitted on x_train ONLY.

    Args:
        x_train: (N, d) training features (may contain NaN).
        y_train: (N,) training labels.
        x_test:  (M, d) test features (may contain NaN).
        method: "none", "standardize", "best_features", "all" or "clean".
        degree: polynomial degree (>= 1).
        feature_names: (d,) column names, used by "clean".
        clean_options: dict of CLEAN_OPTIONS overrides, used by "clean".

    Returns:
        x_train_processed, x_test_processed (with a bias column). Rows are never
        removed, so y_train is unchanged.
    """
    if method not in ("none", "standardize", "best_features", "all", "clean"):
        raise ValueError(f"Unknown method: {method}")
    assert degree >= 1

    if method == "clean":
        return clean_pipeline(
            x_train,
            y_train,
            x_test,
            feature_names=feature_names,
            **(clean_options or {}),
        )

    # fill missing values with train medians
    fill = fit_imputer(x_train)
    x_tr, x_te = impute(x_train, fill), impute(x_test, fill)

    # selection of best features based on correlation with target and among themselves
    if method in ("best_features", "all"):
        selected = best_features(
            correlation_matrix(x_tr), target_correlation(x_tr, y_train), corr_threshold
        )
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
        x_tr, x_te = build_poly_features(x_tr, degree), build_poly_features(
            x_te, degree
        )

    return add_bias(x_tr), add_bias(x_te)
