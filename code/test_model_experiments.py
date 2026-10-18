"""Small tests for the reproducible experiment pipeline."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from model_experiments import (
    fit_preprocessor,
    metrics,
    read_features,
    ridge_weights,
    stratified_split,
    transform,
)


class TestModelExperiments(unittest.TestCase):
    def test_split_is_reproducible_disjoint_and_stratified(self):
        labels = np.array([-1] * 80 + [1] * 20)
        first = stratified_split(labels)
        second = stratified_split(labels)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(sorted(np.concatenate(first).tolist()), list(range(100)))
        self.assertEqual([np.mean(labels[part] == 1) for part in first], [0.2] * 3)

    def test_preprocessor_uses_training_statistics(self):
        train = np.array([[1.0, np.nan], [3.0, 10.0]], dtype=np.float32)
        preprocessor = fit_preprocessor(train, False)
        holdout = np.array([[np.nan, 1000.0]], dtype=np.float32)
        result = transform(holdout, preprocessor)
        self.assertEqual(result[0, 0], 1.0)
        self.assertEqual(result[0, 1], 0.0)
        self.assertAlmostEqual(result[0, 2], 990.0)
        np.testing.assert_array_equal(np.isnan(train), [[False, True], [False, False]])

    def test_ridge_does_not_penalize_intercept(self):
        gram = np.eye(2)
        target_product = np.array([2.0, 2.0])
        weights = ridge_weights(gram, target_product, 0.5)
        np.testing.assert_allclose(weights, [2.0, 1.0])

    def test_positive_class_metrics(self):
        labels = np.array([1, 1, -1, -1])
        predicted = np.array([1, -1, 1, -1])
        np.testing.assert_allclose(metrics(labels, predicted), [0.5, 0.5, 0.5, 0.5])

    def test_reader_checks_ids(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "features.csv"
            with path.open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerows([["Id", "a", "b"], [3, 2, ""], [4, 5, 6]])
            features, ids, names = read_features(path, np.array([3, 4]))
            self.assertEqual(names, ["a", "b"])
            np.testing.assert_array_equal(ids, [3, 4])
            self.assertTrue(np.isnan(features[0, 1]))
            with self.assertRaises(ValueError):
                read_features(path, np.array([4, 3]))


if __name__ == "__main__":
    unittest.main()
