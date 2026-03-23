import unittest

import numpy as np
import pandas as pd

from pgmpy.ci_tests import KCI


class TestKCI(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(seed=42)

        # Independent continuous data
        self.df_ind = pd.DataFrame(
            rng.standard_normal(size=(200, 3)),
            columns=["X", "Y", "Z"],
        )

        # Conditionally independent: X <- Z -> Y (linear)
        Z = rng.normal(size=500)
        X = 3 * Z + rng.normal(loc=0, scale=0.5, size=500)
        Y = 2 * Z + rng.normal(loc=0, scale=0.5, size=500)
        self.df_cind = pd.DataFrame({"X": X, "Y": Y, "Z": Z})

        # Dependent: X -> Y (direct edge, no confounder)
        X_dep = rng.normal(size=300)
        Y_dep = 2 * X_dep + rng.normal(loc=0, scale=0.3, size=300)
        self.df_dep = pd.DataFrame({"X": X_dep, "Y": Y_dep})

        # Non-linear dependency: Y = X^2 + noise
        X_nl = rng.normal(size=300)
        Y_nl = X_nl**2 + rng.normal(loc=0, scale=0.3, size=300)
        self.df_nonlinear = pd.DataFrame({"X": X_nl, "Y": Y_nl})

        # Conditionally independent with multiple Z variables
        Z1 = rng.normal(size=300)
        Z2 = rng.normal(size=300)
        X_mul = Z1 + Z2 + rng.normal(loc=0, scale=0.3, size=300)
        Y_mul = Z1 - Z2 + rng.normal(loc=0, scale=0.3, size=300)
        self.df_cind_mul = pd.DataFrame({"X": X_mul, "Y": Y_mul, "Z1": Z1, "Z2": Z2})

    def test_unconditional_independent(self):
        test = KCI(data=self.df_ind)
        result = test("X", "Y", [], significance_level=0.05)
        self.assertTrue(result)
        self.assertTrue(test.p_value_ > 0.05)

    def test_unconditional_dependent(self):
        test = KCI(data=self.df_dep)
        result = test("X", "Y", [], significance_level=0.05)
        self.assertFalse(result)
        self.assertTrue(test.p_value_ < 0.05)

    def test_nonlinear_dependency(self):
        # KCI should detect non-linear dependence (Y = X^2 + noise)
        test = KCI(data=self.df_nonlinear)
        result = test("X", "Y", [], significance_level=0.05)
        self.assertFalse(result)
        self.assertTrue(test.p_value_ < 0.05)

    def test_conditional_independent(self):
        test = KCI(data=self.df_cind)
        result = test("X", "Y", ["Z"], significance_level=0.05)
        self.assertTrue(result)
        self.assertTrue(test.p_value_ > 0.05)

    def test_conditional_independent_multiple_z(self):
        test = KCI(data=self.df_cind_mul)
        result = test("X", "Y", ["Z1", "Z2"], significance_level=0.05)
        self.assertTrue(result)
        self.assertTrue(test.p_value_ > 0.05)

    def test_conditional_dependent(self):
        # X and Y are marginally independent but dependent given Z (v-structure)
        rng = np.random.default_rng(seed=42)
        X = rng.normal(size=300)
        Y = rng.normal(size=300)
        Z = 2 * X + 2 * Y + rng.normal(loc=0, scale=0.3, size=300)
        df_vstruct = pd.DataFrame({"X": X, "Y": Y, "Z": Z})

        test = KCI(data=df_vstruct)
        result = test("X", "Y", ["Z"], significance_level=0.05)
        self.assertFalse(result)
        self.assertTrue(test.p_value_ < 0.05)

    def test_statistic_and_pvalue_set(self):
        test = KCI(data=self.df_ind)
        test("X", "Y", [], significance_level=0.05)
        self.assertTrue(hasattr(test, "statistic_"))
        self.assertTrue(hasattr(test, "p_value_"))
        self.assertIsInstance(test.statistic_, float)
        self.assertIsInstance(test.p_value_, float)

    def test_is_independent_method(self):
        test = KCI(data=self.df_ind)
        result = test.is_independent("X", "Y", [], significance_level=0.05)
        self.assertIn(result, (True, False))

    def test_invalid_inputs(self):
        test = KCI(data=self.df_ind)
        with self.assertRaises(ValueError):
            test("X", "X", [], significance_level=0.05)
        with self.assertRaises(ValueError):
            test("X", "Y", "Z", significance_level=0.05)

    def test_get_ci_test_by_name(self):
        from pgmpy.ci_tests import get_ci_test

        test = get_ci_test(test="kci", data=self.df_ind)
        self.assertIsInstance(test, KCI)
