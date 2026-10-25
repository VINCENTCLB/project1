# CS-433 Project 1 — predicting coronary heart disease (MICHD)

Binary classification on the BRFSS 2015 telephone survey: predict whether a
respondent has been told they have coronary heart disease or have had a heart
attack, from 321 survey features. Implemented with NumPy and the Python
standard library only; matplotlib is used for figures.

The positive class is rare (8.83% of 328,135 labelled rows), so the competition
and this repository score **positive-class F1**, not accuracy. Always predicting
the negative class scores 0.9117 accuracy and 0.0 F1, which is why accuracy
alone is misleading here.

## Repository layout

```
implementations.py        The six required course functions (graded)
run.py                    Reproduce the submission end to end
code/
  helpers.py              Course-provided CSV loading and submission writing
  data_cache.py           .npy cache around the CSV parser (253s -> 0.8s)
  data_preprocessing.py   Imputation, scaling, BRFSS code decoding, encoding
  model_experiments.py    Splits, metrics, threshold and hyperparameter search
  medical_predictions.py  Command-line interface
  visualizations.py       Figures
  test_my_methods.py      Tests for the gradient-descent implementations
  test_model_experiments.py  Tests for the pipeline and split isolation
  EXPERIMENTS.md          Experiment log and results
  IMPROVEMENT_PLAN.md     Prioritised plan with measured ablations
grading_tests/            Course-provided public tests
latex-example-paper/      IEEE LaTeX template for the report
```

## Setup

Requires Python 3 with NumPy and matplotlib.

```sh
pip install numpy matplotlib
```

Download `x_train.csv`, `y_train.csv` and `x_test.csv` from the competition
platform and place them, with their original names, in:

```
data/dataset/
```

That directory is gitignored: the raw data and generated submissions are never
committed.

## Reproduce

```sh
python run.py
```

This writes three files to `data/dataset/integrated_results/`:

| File | Contents |
| --- | --- |
| `submission.csv` | Predictions for `x_test.csv` in `Id,Prediction` format |
| `metrics.json` | Every trial, the selected configuration and holdout metrics |
| `validation_f1.png` | Development F1 against ridge lambda, per method |

The first run parses the CSVs (about four minutes) and caches the parsed arrays
as `.npy` beside the data; later runs load in under a second. The cache is keyed
on the size and modification time of the three source files, so replacing one
invalidates it automatically.

For more control, call the CLI directly:

```sh
python code/medical_predictions.py --help
python code/medical_predictions.py --preprocessing clean --submission --plots
python code/medical_predictions.py --subsample --preprocessing standardize
```

Useful flags: `--preprocessing {standardize,best_features,all,none,clean,compare}`,
`--lambdas`, `--pos-weights`, `--one-hot-top-k`, `--interactions-top-k`,
`--clip`, `--degree`, `--subsample`, `--no-cache`.

## Method

1. **Split before preprocessing.** Raw rows are split 60/20/20 into training,
   development and holdout, stratified by class, with a fixed seed. Every
   transformation is fitted on the training rows only.
2. **Decode the survey codes.** BRFSS stores "don't know" and "refused" as
   numbers (`7/9`, `77/99`, `777/999`) and "none" as `88/888`. Left alone, a
   refusal enters the model as the quantity 99. A tiered rule maps them to NaN
   (or zero), applying a tier only when every legitimate value in the column
   lies below that tier's codes, so genuine values such as 7 days in `PHYSHLTH`
   or FIPS code 9 in `_STATE` survive.
3. **Impute and encode.** Missing values are filled with training medians.
   A missingness indicator block is appended, because 44.8% of cells are missing
   and which questions were asked is itself informative. Low-cardinality columns
   are additionally one-hot encoded, so ordinal scales keep their ordering while
   nominal codes gain one parameter per level.
4. **Fit ridge regression** in closed form, with the positive class upweighted
   to counteract the 91/9 imbalance.
5. **Select on development data.** Preprocessing method, lambda, class weight
   and decision threshold are all chosen to maximise development F1. The
   holdout is scored exactly once, and compared against a majority baseline.
6. **Refit on all labelled rows** with the selected settings to predict the
   test set.

See `code/EXPERIMENTS.md` for results and `code/IMPROVEMENT_PLAN.md` for the
measured ablation study behind these choices.

## Tests

```sh
python -m unittest discover -s code -p "test_*.py" -v
```

The course's public tests check the signatures and numerics of the six required
functions in `implementations.py`; run them from the course repository:

```sh
pytest --github_link <GITHUB-REPO-URL> grading_tests/
```
