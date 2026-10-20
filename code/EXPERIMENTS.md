# Medical-dataset model experiment

This experiment uses only Python's standard library and NumPy. It is separate
from the six required functions in `implementations.py`; it does not replace
any teammate's implementation.

## Method

- Read the original `x_train.csv`, `y_train.csv`, and `x_test.csv` from the
  ignored `data/dataset/` directory. Verify that feature and label IDs align.
- Split the labeled data by class, using seed 433: 60% training, 20%
  development, and 20% untouched holdout.
- Fit mean imputation and standardization **only on training rows**. Compare
  standardized features with and without missing-value indicator features.
- Fit ridge regression with an unpenalized intercept and five values of
  `lambda` (`0.0001`, `0.001`, `0.01`, `0.1`, `1`). On the development set,
  choose a prediction threshold for the positive-class F1 score.
- Select the feature variant, `lambda`, and threshold using development F1.
  Evaluate the selected configuration once on the untouched holdout.
- For a competition submission, refit preprocessing and weights on all labeled
  rows with the selected settings, then predict `x_test.csv` and write a local
  CSV in the official `Id,Prediction` format. This final refit is not identical
  to the 60%-training model used for the holdout estimate.

## Results on the provided dataset

The labeled data contain 328,135 rows, 321 feature columns, and 8.83% positive
labels. Each development and holdout split contains 65,627 rows.

| Model | Holdout accuracy | Positive precision | Positive recall | Positive F1 |
| --- | ---: | ---: | ---: | ---: |
| Always predict -1 | 0.9117 | 0.0000 | 0.0000 | 0.0000 |
| Standardized ridge | 0.8683 | 0.3367 | 0.5061 | 0.4044 |

The selected settings were no missingness indicators, `lambda=0.01`, and
threshold `-0.5574`. Its development F1 was `0.4083`. The majority baseline
illustrates why accuracy alone is misleading for this imbalanced dataset.
These are local validation results, not an AIcrowd test score.

## Reproduce

Run from the repository root, after installing NumPy:

```sh
python3 -m unittest discover -s code -p 'test_*.py' -v
python3 code/model_experiments.py --data-dir data/dataset \
  --submission data/dataset/submission_kora_ridge.csv
```

Keep the raw datasets and generated submission CSV local. They are ignored by
Git and should not be uploaded to the public repository. The CSV can be
uploaded separately to the competition platform after team review.

Limitations: ridge regression treats many coded categorical variables as
ordinal. The threshold was selected for positive-class F1; if the competition
uses a different metric, retune it using local validation. Further work could
investigate categorical encoding and compare against the team's logistic
regression implementation after that code is reviewed and merged.
