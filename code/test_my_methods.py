"""Run with: python3 -m unittest -v test_my_methods.py.

Only NumPy and the standard library are needed. The first three cases use
the numerical examples in the course's public tests
This is a focused check of two methods, not the complete grading suite.
"""

import unittest

import numpy as np

from implementations import mean_squared_error_gd, mean_squared_error_sgd


class TestMyMethods(unittest.TestCase):
    def setUp(self):
        self.y = np.array([0.1, 0.3, 0.5])
        self.tx = np.array([[2.3, 3.2], [1.0, 0.1], [1.4, 2.3]])
        self.initial_w = np.array([0.5, 1.0])
        # Keep tests from changing the random state of other code.
        self.random_state = np.random.get_state()
        self.addCleanup(np.random.set_state, self.random_state)

    def test_public_gd_zero_updates(self):
        initial_w = np.array([0.413044, 0.875757])
        w, loss = mean_squared_error_gd(self.y, self.tx, initial_w, 0, 0.1)
        np.testing.assert_allclose(w, initial_w, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(loss, 2.959836, rtol=1e-4, atol=1e-8)
        self.assertEqual(w.shape, (2,))
        self.assertEqual(loss.ndim, 0)

    def test_public_gd_two_updates(self):
        w, loss = mean_squared_error_gd(
            self.y, self.tx, self.initial_w, 2, 0.1
        )
        np.testing.assert_allclose(
            w, [-0.050586, 0.203718], rtol=1e-4, atol=1e-8
        )
        np.testing.assert_allclose(loss, 0.051534, rtol=1e-4, atol=1e-8)
        self.assertEqual(w.shape, (2,))
        self.assertEqual(loss.ndim, 0)

    def test_public_sgd_two_updates_one_sample(self):
        w, loss = mean_squared_error_sgd(
            self.y[:1], self.tx[:1], self.initial_w, 2, 0.1
        )
        np.testing.assert_allclose(
            w, [0.063058, 0.39208], rtol=1e-4, atol=1e-8
        )
        np.testing.assert_allclose(loss, 0.844595, rtol=1e-4, atol=1e-8)
        self.assertEqual(w.shape, (2,))
        self.assertEqual(loss.ndim, 0)

    def test_gradient_matches_finite_difference(self):
        # Check the direction and scale against numerical differentiation.
        epsilon = 1e-6
        numerical_gradient = np.zeros(2)
        for j in range(2):
            step = np.zeros(2)
            step[j] = epsilon
            loss_plus = np.mean(
                (self.y - self.tx @ (self.initial_w + step)) ** 2
            ) / 2
            loss_minus = np.mean(
                (self.y - self.tx @ (self.initial_w - step)) ** 2
            ) / 2
            numerical_gradient[j] = (loss_plus - loss_minus) / (2 * epsilon)
        w, _ = mean_squared_error_gd(
            self.y, self.tx, self.initial_w, 1, 0.1
        )
        inferred_gradient = (self.initial_w - w) / 0.1
        np.testing.assert_allclose(inferred_gradient, numerical_gradient, rtol=1e-7)

    def test_sgd_one_update_and_full_data_loss(self):
        # A one-row update must not be divided by the full dataset size.
        np.random.seed(42)
        index = np.random.randint(len(self.y))
        residual = self.y[index] - self.tx[index] @ self.initial_w
        expected_w = self.initial_w + 0.1 * self.tx[index] * residual
        np.random.seed(42)
        w, loss = mean_squared_error_sgd(
            self.y, self.tx, self.initial_w, 1, 0.1
        )
        np.testing.assert_allclose(w, expected_w)
        expected_loss = np.mean((self.y - self.tx @ expected_w) ** 2) / 2
        np.testing.assert_allclose(loss, expected_loss)

    def test_no_input_mutation(self):
        for method in (mean_squared_error_gd, mean_squared_error_sgd):
            y, tx, initial_w = (
                self.y.copy(), self.tx.copy(), self.initial_w.copy()
            )
            method(y, tx, initial_w, 3, 0.1)
            np.testing.assert_array_equal(y, self.y)
            np.testing.assert_array_equal(tx, self.tx)
            np.testing.assert_array_equal(initial_w, self.initial_w)

    def test_sgd_zero_updates(self):
        w, loss = mean_squared_error_sgd(
            self.y, self.tx, self.initial_w, 0, 0.1
        )
        np.testing.assert_array_equal(w, self.initial_w)
        np.testing.assert_allclose(
            loss, np.mean((self.y - self.tx @ self.initial_w) ** 2) / 2
        )

    def test_sgd_reproducible_with_seed(self):
        np.random.seed(42)
        first = mean_squared_error_sgd(
            self.y, self.tx, self.initial_w, 20, 0.01
        )
        np.random.seed(42)
        second = mean_squared_error_sgd(
            self.y, self.tx, self.initial_w, 20, 0.01
        )
        np.testing.assert_array_equal(first[0], second[0])
        self.assertEqual(first[1], second[1])

    def test_both_converge_on_exact_line(self):
        x = np.linspace(-1, 1, 21)
        tx = np.column_stack((np.ones(len(x)), x))
        y = 1 + 2 * x
        np.random.seed(42)
        for method in (mean_squared_error_gd, mean_squared_error_sgd):
            w, loss = method(y, tx, np.zeros(2), 2000, 0.1)
            np.testing.assert_allclose(w, [1, 2], atol=1e-6)
            self.assertLess(loss, 1e-12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
