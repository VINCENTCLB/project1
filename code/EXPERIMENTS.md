# Predicting coronary heart disease from BRFSS 2015 responses

NumPy and the Python standard library only; matplotlib is used for plots.
`implementations.py` holds the six required course functions and is not
modified by any of this work — the class-weighted and blocked solvers live in
`code/model_experiments.py` and call `ridge_regression` unchanged.

## The task and why accuracy is the wrong target

328,135 labelled rows, 321 feature columns, 8.83% positive. Predicting `-1`
everywhere scores 0.9117 accuracy and 0.0000 F1, so accuracy alone cannot
distinguish a useful model from a constant. Every number below is
positive-class F1 unless stated otherwise.

## Protocol

- Raw rows are split by class at seed 433 into 60% training, 20% development,
  20% holdout. **The split happens before any preprocessing**, and every fitted
  quantity — dropped columns, sentinel rules, composite bands, imputation
  medians, one-hot levels, clip bounds, scaling statistics, interaction pairs —
  is estimated on training rows only.
- Settings are chosen on development F1 (`--folds 1`, the default) or by
  stratified cross-validation over the non-holdout rows (`--folds 5`).
- The holdout is scored once, at the end. Its labels inform nothing: the
  decision boundary transfers as a *flag rate*, which needs only the holdout
  score distribution.
- For the submission, everything is refitted on all 328,135 labelled rows. That
  model is not the one the holdout measured, so the holdout figure is an
  estimate of it, not a measurement of it.

### Why the threshold transfers as a flag rate, not a score

Refitting on 328,135 rows instead of 196,881 shifts the score distribution, so
a numeric cutoff chosen on development scores no longer flags the intended
fraction. Reproducing the *fraction* is scale invariant.
`threshold_at_flag_rate` does this. Measured on the actual refit path, over 5
folds: flag rate 0.1350 selected, 0.1350 realised, and F1 indistinguishable
between the two transfer modes — so this is insurance against a failure mode
that did not materialise, kept because it costs nothing.

## Preprocessing: what each stage is for

| Stage | What it does | Why |
| --- | --- | --- |
| `drop_admin` | Removes 31 administrative columns by name (`IDATE`, `SEQNO`, `_PSU`, `_LLCPWT`, `QSTVER`, …) | Survey bookkeeping, not physiology; 6 are constant |
| `sentinels` | Decodes BRFSS non-answers (7/9, 77/99, 777/999 → missing; 88/888 → zero) to `NaN` | Untouched, "refused to answer" enters the model as the number 9 |
| `indicators` | Appends a binary column per feature with varying missingness | Non-response is informative here, not missing at random |
| `composite` | Splits banded codes (1xx/2xx) into a count and a one-hot band label | `201` means "2 times per week", not the quantity 201 |
| one-hot | Encodes nominal columns (see below) | A numeric code with no order is meaningless as a number |
| `clip` | Clips to training 2.5/97.5 percentiles | Bounds leverage of extreme responses |
| `interactions` | Products of the 20 strongest features | Lets the linear model express simple non-additivity |

The sentinel rule is tiered and conditional: a tier fires only when every
remaining legitimate value lies strictly below that tier's codes. This is what
keeps it from destroying real data — `PHYSHLTH` uses 7 and 9 as genuine day
counts while also using 77/88/99 as codes, and `_STATE` contains 9 as a FIPS
code. Traced: `PHYSHLTH` {1..30, 77, 88, 99} fires at tier 2 and declines tier
3; `GENHLTH` {1-5, 7, 9} fires at tier 3; `_STATE` declines every tier.

### Class weighting

91% of rows are negative, so 91% of the gradient that sets the direction of `w`
comes from classifying negatives. Scaling each row by `sqrt(s_i)` makes the
unmodified course solver produce the weighted solution exactly, so
`implementations.py` is reused rather than duplicated. This changes the
*ranking*, which is what threshold tuning alone cannot do.

## Measured ablation

Each stage added to the one above it, development F1:

| Config | Added | Dev F1 |
| --- | --- | ---: |
| E0 | standardize + median impute | 0.4089 |
| E1 | + drop admin columns | 0.4079 |
| E2 | + sentinel decoding | 0.4188 |
| E3 | + missingness indicators | 0.4225 |
| E4 | + composite split, one-hot, interactions, clip | 0.4318 |

Sentinel decoding is the single largest win (+0.0109). Dropping admin columns
slightly *hurt* on this split and was kept anyway, because 6 of the 31 are
constant and the rest cannot carry physiological signal.

## Cross-validated verdicts

Paired 5-fold CV over the non-holdout rows, holdout sealed throughout. A
positive `t` with 5/5 fold wins is treated as real; anything under |t|=2 is not.

| Question | Result | Verdict |
| --- | --- | --- |
| Winning config vs. runner-up | +0.003, t=+12.4, 5/5 | keep |
| `clip` | t=+2.79 | keep |
| Sentinel decoding | large, consistent | keep |
| `interactions_top_k` 30 vs 20 | −0.0006, t=−0.93, 1/5 | reject |
| `one_hot_top_k` 40 vs 15 | −0.0001, t=−0.39, 2/5 | reject |
| λ ≥ 0.1 | monotonically worse | reject |
| Bagging 10 models | t=+1.22 | reject |
| Binary columns excluded from one-hot | −0.0006, t=−1.05, 1/5 | reject |
| Curated nominal encoding | −0.0010 to +0.0000 | reject |
| Logistic loss instead of squared | −0.0072 | reject |
| Flag-rate vs score transfer | neutral | keep (free insurance) |

### λ and the class weight are a flat plateau

An earlier conclusion that "λ does nothing" was **wrong**, and it cost real
score: λ had been pinned at 0.01 throughout an earlier coordinate search, which
handicapped every configuration by about 0.005. At the 789-feature width,
λ=1e-4 beats λ=1e-2 by +0.0049 (t=6.25, 5/5).

Re-tuning λ and the class weight jointly, 18 combinations, 5-fold CV:

| | w=2 | w=3 | w=5 | w=7 | w=10 | w=20 | w=30 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| λ=1e-5 | 0.4382 | 0.4373 | 0.4373 | 0.4376 | — | — | — |
| λ=1e-4 | 0.4377 | **0.4384** | 0.4382 | 0.4374 | 0.4356 | 0.4312 | 0.4259 |
| λ=1e-3 | 0.4371 | 0.4373 | 0.4376 | **0.4382** | — | — | — |

Everything from w=2 to w=7 and λ=1e-5 to 1e-3 spans 0.4369–0.4384 against a
fold standard deviation of 0.0034: it is one flat plateau, and the top five
combinations are within 0.0004 of each other. Only the edges are rejectable —
w=20 at −0.0044 (t=−3.25) and w=30 at −0.0097 (t=−13.67) against w=10.

Two consequences:

1. `POS_WEIGHTS` excludes 10, which lost to 7 on all five folds, and keeps 1 as
   the unweighted baseline. Weights above 7 are rejected outright.
2. **Selecting on one 20% development split is the weak link.** The useful
   differences are ~0.001 while the single-split noise is ~0.003, so the
   selector can and does prefer settings that lose on every CV fold. Hence
   `--folds 5`.

## Logistic loss does not help here

The structural objection to ridge on this task is that squared loss on ±1
targets charges `(1−3)² = 4` for a confidently-correct positive scored at 3.0,
where logistic charges 0.049. Tested with the course
`reg_logistic_regression`, full-batch GD, same 789-feature matrix and split:

| Model | Dev F1 |
| --- | ---: |
| Ridge, λ=1e-3, w=7 | **0.4389** |
| Ridge, λ=1e-4, w=5 | 0.4379 |
| Logistic, γ=0.2, 4000 iters, λ=0 | 0.4317 |
| Logistic, γ=0.2, 4000 iters, λ=1e-5 | 0.4317 |

Learning rates were probed first (γ=0.05/0.2/1/4/16 → dev F1
0.4230/0.4302/0.3600/0.1299/0.2210; γ≥1 overflows `exp` and diverges). At
γ=0.2 the run is converged for practical purposes: dev F1 sat between 0.4311
and 0.4317 from iteration 1,750 to 4,000 while train cross-entropy fell only
from 0.26029 to 0.25943.

Class weighting makes logistic worse, not better: at γ=0.2 and 1,500
iterations, w=1 scores 0.4308, w=3 scores 0.4131 and w=5 scores 0.4178. The
thresholds selected for the weighted runs sit far out in the tails (−6.39 and
+8.95, against −0.38 unweighted), so those two are also less converged than the
unweighted run and their exact values should not be leaned on. The direction is
nevertheless opposite to ridge, where class weighting is worth about 0.01, and
it is what the theory predicts: logistic already models the conditional
probability, so reweighting the classes mostly shifts a prior that threshold
tuning has already absorbed, while distorting the fit.

**Logistic is 0.0072 worse, and the gap is stable.** The honest caveat: train
cross-entropy was still decreasing slowly, so this compares an exactly-solved
ridge (normal equations) against a first-order approximation of logistic. A
second-order optimiser might close some of the gap, but it would have to find
0.007 from a dev F1 that has been flat for 2,250 iterations, and the course
constraints provide only gradient descent. The theoretical objection to squared
loss is real; its effect on ranking quality at this sample size is not.

## Categorical encoding: nominal by meaning, not by correlation

`fit_categorical_levels` ranked candidates by absolute target correlation and
capped them at 12 levels. Both rules are blind to meaning, and the survey of
what they actually selected showed two concrete defects:

1. **Nine of the fifteen slots went to binary columns** (`_RFHLTH`, `_RFHYPE5`,
   `DIFFWALK`, `CVDSTRK3`, `USEEQUIP`, `TOLDHI2`, `_RFCHOL`, `QLACTLM2`,
   `CHCCOPD1`). `one_hot_blocks` drops the first level, so a 2-level column
   contributes exactly one indicator, which is an affine function of the numeric
   column already in the matrix — no information, and exactly collinear with a
   column already present.
2. **Correlation ranking cannot find a nominal code.** `_STATE` has 53 FIPS
   levels and a target correlation of 0.0057 — not because CHD prevalence is
   flat across states, but because the numbering is arbitrary. It was excluded
   twice over: by rank and by the 12-level cap. `NOMINAL_COLUMNS` names 17 such
   columns and encodes them by meaning regardless of either rule.

Both defects are real. Neither fix helps. Paired 5-fold CV, each change
isolated, at both plateau settings:

| Variant | d | λ=1e-3, w=7 | λ=1e-4, w=5 |
| --- | ---: | ---: | ---: |
| A previous | 789 | 0.4382 | 0.4382 |
| B binaries excluded | 812 | 0.4376 | 0.4379 |
| C curated nominal only | 916 | 0.4382 | 0.4373 |
| D both | 943 | 0.4373 | 0.4386 |

Paired against A: B measures −0.0006 (t=−1.05, 1/5) and −0.0003 (t=−0.97,
1/5); C +0.0000 (t=+0.03) and −0.0010 (t=−1.40); D −0.0009 (t=−0.84) and
+0.0004 (t=+0.49). D holds the single highest cell in the table, 0.4386, and
also the lowest of the four at the other setting — the sign flips between two
settings that are themselves indistinguishable, which is what noise looks like.
**All three are rejected.**

Why the reasoning was right and the effect is still absent:

- The nine binary slots genuinely carried no information, but the 3-level
  columns promoted in their place carry no more. Freeing a wasted budget only
  helps if something valuable is waiting for it.
- Ridge is not damaged by the collinear indicator a binary column produces; it
  shrinks it. The redundancy was inert, not harmful, so removing it recovers
  nothing.
- `_STATE` is genuinely nominal, and 52 one-hot columns for it still add
  nothing measurable — whatever geographic variation in CHD prevalence exists
  appears already carried by the demographic and health covariates.

`min_levels` and `nominal` therefore default to off, with the options and their
tests kept in place and this rejection recorded beside them.

## Sealed-holdout result

Touched once, at the single-split selection `clean`, λ=1e-3, w=7, flag rate
0.1300:

| Model | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Always predict -1 | 0.9117 | 0.0000 | 0.0000 | 0.0000 |
| Selected ridge | 0.8743 | 0.3591 | 0.5393 | **0.4311** |

Confusion counts: TP 3,125 · FP 5,578 · FN 2,670 · TN 54,254. The model trades
accuracy (0.8743 against the baseline's 0.9117) for recall, which is the right
trade when the positive class is the point.

Competition scores on the held-back test set: **0.419** at the first submission
(λ=0.01, before sentinel decoding and class weighting were tuned), **0.448**
after. The 0.448 model and the one above differ only in plateau position.

## Honest limitations

- The AIcrowd score (0.448) exceeds the local holdout estimate (0.4311). The
  submission model is refitted on 67% more rows, which plausibly accounts for
  it, but this is an explanation after the fact, not a prediction that was
  tested.
- `ridge_regression` penalises the intercept along with every other weight,
  because the course formula specifies it. An unpenalised intercept would be
  marginally better specified; at λ=1e-3 the effect is negligible, and changing
  it would mean either editing a graded function or wrapping it.
- `reg_logistic_regression` is implemented, tested and measured but unused, for
  the reasons above.
- The 17 nominal columns are hand-named from BRFSS codebook knowledge. The A/B
  above tested that judgment and found no effect, so the curated encoding is
  available but off.
- Three independent sweeps now agree that λ and the class weight are exhausted,
  and feature width has been rejected at two different λ values. The remaining
  untested directions are a second-order optimiser for logistic regression
  (outside the course constraints) and per-column ordinal encoding beyond the
  nominal/numeric split tried here.

## Reproduce

From the repository root, with the dataset in `data/dataset/`:

```sh
python -m unittest discover -s code -p 'test_*.py'   # 26 tests
python run.py                                        # ~180 s with warm cache
```

`run.py` writes `metrics.json`, `validation_f1.png` and `submission.csv` to
`data/dataset/integrated_results/`. For the cross-validated selection instead
of the single split:

```sh
python code/medical_predictions.py --folds 5 --submission
```

Raw data and submissions stay local; `data/dataset/` is git-ignored.
