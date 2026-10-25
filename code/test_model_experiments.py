"""Tests for split isolation and integration with the team's pipeline."""
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import data_preprocessing
from medical_predictions import run_pipeline
from model_experiments import (
    COMPARED_METHODS,
    POS_WEIGHTS,
    choose_threshold,
    make_test_predictions,
    metrics,
    run_experiments,
    stratified_folds,
    stratified_split,
    weighted_ridge,
)


class TestModelExperiments(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.features = rng.normal(size=(100, 4))
        self.features[::7, 1] = np.nan
        self.labels = np.where(self.features[:, 0] > 0.4, 1, -1)

    def test_split_is_reproducible_disjoint_and_stratified(self):
        labels = np.array([-1] * 80 + [1] * 20)
        first, second = stratified_split(labels), stratified_split(labels)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(sorted(np.concatenate(first).tolist()), list(range(100)))
        self.assertEqual([np.mean(labels[part] == 1) for part in first], [0.2] * 3)

    def test_positive_class_metrics(self):
        result = metrics(np.array([1, 1, -1, -1]), np.array([1, -1, 1, -1]))
        for name in ("accuracy", "precision", "recall", "f1"):
            self.assertEqual(result[name], 0.5)

    def test_threshold_separates_simple_scores(self):
        labels = np.array([-1, -1, 1, 1])
        f1, threshold = choose_threshold(labels, np.array([-3.0, -2.0, 2.0, 3.0]))
        self.assertEqual(f1, 1.0)
        self.assertGreater(threshold, -2.0)
        self.assertLessEqual(threshold, 2.0)

    def test_preprocessing_is_fitted_only_on_training_rows(self):
        original = self.features.copy()
        train_idx, _, _ = stratified_split(self.labels)
        with patch(
            "model_experiments.data_processing.pipeline",
            wraps=data_preprocessing.pipeline,
        ) as pipeline:
            with contextlib.redirect_stdout(io.StringIO()):
                result = run_experiments(self.features, self.labels, lambdas=(0.01,))
            # three compared methods plus one refit for the holdout
            self.assertEqual(pipeline.call_count, 4)
            for call in pipeline.call_args_list:
                np.testing.assert_array_equal(call.args[0], self.features[train_idx])
                np.testing.assert_array_equal(call.args[1], self.labels[train_idx])
        np.testing.assert_array_equal(self.features, original)
        self.assertEqual(
            len(result["trials"]), len(COMPARED_METHODS) * len(POS_WEIGHTS)
        )
        self.assertEqual(
            sum(result["holdout"][key] for key in ("tp", "fp", "fn", "tn")),
            result["split_sizes"]["holdout"],
        )

    def test_final_refit_uses_all_labels_and_returns_valid_predictions(self):
        settings = {
            "selected": {
                "preprocessing": "standardize",
                "lambda": 0.01,
                "threshold": 0.0,
            },
            "degree": 1,
        }
        with patch(
            "model_experiments.data_processing.pipeline",
            wraps=data_preprocessing.pipeline,
        ) as pipeline:
            predicted = make_test_predictions(
                self.features, self.labels, self.features[:9], settings
            )
            self.assertEqual(len(pipeline.call_args.args[0]), len(self.labels))
        self.assertEqual(predicted.shape, (9,))
        self.assertTrue(np.all(np.isin(predicted, (-1, 1))))

    def test_weighted_ridge_matches_explicit_weighted_normal_equations(self):
        rng = np.random.default_rng(3)
        tx = np.hstack([np.ones((40, 1)), rng.normal(size=(40, 3))])
        y = np.where(rng.normal(size=40) > 0.8, 1, -1)
        lambda_, pos_weight = 0.05, 7.0
        weights, _ = weighted_ridge(y, tx, lambda_, pos_weight)
        scale = np.where(y == 1, pos_weight, 1.0)
        lhs = tx.T @ (scale[:, None] * tx) + 2 * len(y) * lambda_ * np.eye(tx.shape[1])
        expected = np.linalg.solve(lhs, tx.T @ (scale * y))
        np.testing.assert_allclose(weights, expected, rtol=1e-10, atol=1e-12)

    def test_blocked_weighted_ridge_matches_the_single_pass_solver(self):
        rng = np.random.default_rng(5)
        tx = np.hstack([np.ones((97, 1)), rng.normal(size=(97, 4))])
        y = np.where(rng.normal(size=97) > 0.7, 1, -1)
        for pos_weight in (1.0, 6.0):
            single = weighted_ridge(y, tx, 0.02, pos_weight)
            blocked = weighted_ridge(y, tx, 0.02, pos_weight, block_rows=10)
            np.testing.assert_allclose(blocked[0], single[0], rtol=1e-9, atol=1e-11)

    def test_weighted_ridge_is_unweighted_at_unit_weight(self):
        rng = np.random.default_rng(4)
        tx = rng.normal(size=(30, 3))
        y = np.where(rng.normal(size=30) > 0, 1, -1)
        np.testing.assert_allclose(
            weighted_ridge(y, tx, 0.1, 1.0)[0], weighted_ridge(y, tx, 0.1)[0]
        )

    def test_clean_pipeline_decodes_sentinels_and_keeps_real_values(self):
        # GENHLTH-like (1-5 scale, 7/9 are codes) and _STATE-like (9 is a real
        # FIPS code, so the tier must decline).
        x_train = np.array(
            [[1.0, 1.0], [2.0, 9.0], [3.0, 53.0], [5.0, 72.0], [7.0, 6.0], [9.0, 4.0]]
        )
        y_train = np.array([-1, -1, 1, 1, -1, 1])
        rules = data_preprocessing.fit_sentinel_rules(x_train)
        self.assertEqual(rules[0], ([7, 9], []))
        self.assertNotIn(1, rules)
        mapped = data_preprocessing.apply_sentinel_rules(x_train, rules)
        self.assertEqual(int(np.count_nonzero(np.isnan(mapped[:, 0]))), 2)
        self.assertEqual(int(np.count_nonzero(np.isnan(mapped[:, 1]))), 0)

    def test_clean_pipeline_fits_only_on_training_rows(self):
        x_train = self.features[:60]
        x_test = self.features[60:]
        train_out, test_out = data_preprocessing.clean_pipeline(
            x_train, self.labels[:60], x_test
        )
        self.assertEqual(train_out.shape[1], test_out.shape[1])
        self.assertTrue(np.all(train_out[:, 0] == 1.0))
        self.assertFalse(np.any(np.isnan(train_out)))
        self.assertFalse(np.any(np.isnan(test_out)))

    def test_main_pipeline_loads_csv_and_writes_outputs(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            data_dir = Path(temporary_dir) / "dataset"
            data_dir.mkdir()
            for filename, values, ids in (
                ("x_train.csv", self.features, range(100)),
                ("x_test.csv", self.features[:9], range(100, 109)),
            ):
                with (data_dir / filename).open("w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(["Id", "a", "b", "c", "d"])
                    for row_id, row in zip(ids, values):
                        writer.writerow(
                            [row_id]
                            + ["" if np.isnan(value) else value for value in row]
                        )
            with (data_dir / "y_train.csv").open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Id", "_MICHD"])
                writer.writerows(enumerate(self.labels))
            output_dir = Path(temporary_dir) / "results"
            with patch("medical_predictions.plot_statistics") as plot_statistics:
                with contextlib.redirect_stdout(io.StringIO()):
                    run_pipeline(
                        data_path=data_dir,
                        subsample=False,
                        degree=1,
                        lr=0.01,
                        submission=True,
                        preprocessing="compare",
                        lambdas=(0.01,),
                        output_dir=output_dir,
                        plots=True,
                        show_statistics=True,
                    )
                self.assertEqual(plot_statistics.call_count, 2)
            with (output_dir / "submission.csv").open(newline="") as file:
                predictions = list(csv.DictReader(file))
            self.assertEqual(
                [int(row["Id"]) for row in predictions], list(range(100, 109))
            )
            self.assertTrue(
                all(row["Prediction"] in ("-1", "1") for row in predictions)
            )
            with (output_dir / "metrics.json").open() as file:
                self.assertEqual(json.load(file)["rows"], 100)
            self.assertGreater((output_dir / "validation_f1.png").stat().st_size, 0)

    def test_stratified_folds_are_disjoint_and_keep_the_class_ratio(self):
        labels = np.where(np.arange(200) % 10 == 0, 1, -1)
        indices = np.arange(200)
        folds = stratified_folds(labels, indices, 5, seed=1)
        self.assertEqual(len(folds), 5)
        combined = np.concatenate(folds)
        self.assertEqual(sorted(combined.tolist()), list(range(200)))
        self.assertEqual(len(set(combined.tolist())), 200)
        for fold in folds:
            self.assertEqual(int(np.count_nonzero(labels[fold] == 1)), 4)

    def test_stratified_folds_rejects_a_single_fold(self):
        labels = np.where(np.arange(50) % 5 == 0, 1, -1)
        with self.assertRaises(ValueError):
            stratified_folds(labels, np.arange(50), 1)

    def test_cross_validated_selection_records_one_f1_per_fold(self):
        results = run_experiments(
            self.features,
            self.labels,
            preprocessing="standardize",
            lambdas=(0.1,),
            pos_weights=(1.0, 3.0),
            folds=3,
        )
        self.assertEqual(results["selection"], "3-fold cross-validation")
        self.assertEqual(len(results["trials"]), 2)
        for trial in results["trials"]:
            self.assertEqual(len(trial["fold_f1"]), 3)
            self.assertAlmostEqual(
                trial["development"]["f1"], float(np.mean(trial["fold_f1"]))
            )
        self.assertIn("f1", results["holdout"])

    def test_run_experiments_rejects_a_nonsense_fold_count(self):
        with self.assertRaises(ValueError):
            run_experiments(
                self.features,
                self.labels,
                preprocessing="standardize",
                lambdas=(0.1,),
                folds=0,
            )

    def test_binary_columns_are_excluded_from_ranked_one_hot(self):
        # A binary column one-hot encodes to a single indicator that is an
        # affine function of the numeric column, so min_levels=3 should skip it
        # while still encoding the three-level column beside it.
        rng = np.random.default_rng(3)
        binary = rng.integers(0, 2, size=400).astype(float)
        ternary = rng.integers(0, 3, size=400).astype(float)
        x = np.column_stack([binary, ternary])
        y = np.where(binary + ternary > 1.5, 1, -1)
        permissive = data_preprocessing.fit_categorical_levels(
            x, y, max_levels=12, top_k=5, min_levels=2
        )
        guarded = data_preprocessing.fit_categorical_levels(
            x, y, max_levels=12, top_k=5, min_levels=3
        )
        self.assertEqual(sorted(permissive), [0, 1])
        self.assertEqual(sorted(guarded), [1])

    def test_nominal_columns_are_encoded_whatever_their_correlation(self):
        # _STATE is a FIPS code: its numeric correlation with the label is
        # meaningless, so ranked selection would never reach it, and its 53
        # levels exceed the cap that bounds the ranked candidates.
        rng = np.random.default_rng(5)
        state = rng.integers(1, 54, size=600).astype(float)
        other = rng.normal(size=600)
        x = np.column_stack([state, other])
        names = np.array(["_STATE", "GENHLTH"])
        levels = data_preprocessing.fit_nominal_levels(x, names, max_levels=60)
        self.assertIn(0, levels)
        self.assertNotIn(1, levels)
        self.assertEqual(len(levels[0]), len(np.unique(state)))
        self.assertEqual(data_preprocessing.fit_nominal_levels(x, None), {})


if __name__ == "__main__":
    unittest.main()
