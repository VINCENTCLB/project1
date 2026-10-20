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
    choose_threshold, make_test_predictions, metrics, run_experiments,
    stratified_split,
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
        with patch("model_experiments.data_processing.pipeline", wraps=data_preprocessing.pipeline) as pipeline:
            with contextlib.redirect_stdout(io.StringIO()):
                result = run_experiments(self.features, self.labels, lambdas=(0.01,))
            self.assertEqual(pipeline.call_count, 3)
            for call in pipeline.call_args_list:
                np.testing.assert_array_equal(call.args[0], self.features[train_idx])
                np.testing.assert_array_equal(call.args[1], self.labels[train_idx])
        np.testing.assert_array_equal(self.features, original)
        self.assertEqual(len(result["trials"]), 2)
        self.assertEqual(sum(result["holdout"][key] for key in ("tp", "fp", "fn", "tn")), result["split_sizes"]["holdout"])

    def test_final_refit_uses_all_labels_and_returns_valid_predictions(self):
        settings = {"selected": {"preprocessing": "standardize", "lambda": 0.01, "threshold": 0.0}, "degree": 1}
        with patch("model_experiments.data_processing.pipeline", wraps=data_preprocessing.pipeline) as pipeline:
            predicted = make_test_predictions(self.features, self.labels, self.features[:9], settings)
            self.assertEqual(len(pipeline.call_args.args[0]), len(self.labels))
        self.assertEqual(predicted.shape, (9,))
        self.assertTrue(np.all(np.isin(predicted, (-1, 1))))

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
                        writer.writerow([row_id] + ["" if np.isnan(value) else value for value in row])
            with (data_dir / "y_train.csv").open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Id", "_MICHD"])
                writer.writerows(enumerate(self.labels))
            output_dir = Path(temporary_dir) / "results"
            with patch("medical_predictions.plot_statistics") as plot_statistics:
                with contextlib.redirect_stdout(io.StringIO()):
                    run_pipeline(
                        data_path=data_dir, subsample=False, degree=1, lr=0.01,
                        submission=True, preprocessing="compare", lambdas=(0.01,),
                        output_dir=output_dir, plots=True, show_statistics=True,
                    )
                self.assertEqual(plot_statistics.call_count, 2)
            with (output_dir / "submission.csv").open(newline="") as file:
                predictions = list(csv.DictReader(file))
            self.assertEqual([int(row["Id"]) for row in predictions], list(range(100, 109)))
            self.assertTrue(all(row["Prediction"] in ("-1", "1") for row in predictions))
            with (output_dir / "metrics.json").open() as file:
                self.assertEqual(json.load(file)["rows"], 100)
            self.assertGreater((output_dir / "validation_f1.png").stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
