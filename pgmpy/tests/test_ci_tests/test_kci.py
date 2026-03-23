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

    def test_unconditional(self):
        # Independent X, Y -> should be independent
        test = KCI(data=self.df_ind)
        self.assertTrue(test("X", "Y", [], significance_level=0.05))
        self.assertGreater(test.p_value_, 0.05)

        # Linear dependence X -> Y -> should be dependent
        test = KCI(data=self.df_dep)
        self.assertFalse(test("X", "Y", [], significance_level=0.05))
        self.assertLess(test.p_value_, 0.05)

        # Non-linear dependence Y = X^2 + noise -> should be dependent
        test = KCI(data=self.df_nonlinear)
        self.assertFalse(test("X", "Y", [], significance_level=0.05))
        self.assertLess(test.p_value_, 0.05)

    def test_conditional(self):
        # X <- Z -> Y: independent given Z (single conditioning variable)
        test = KCI(data=self.df_cind)
        self.assertTrue(test("X", "Y", ["Z"], significance_level=0.05))
        self.assertGreater(test.p_value_, 0.05)

        # X <- (Z1,Z2) -> Y: independent given multiple Z
        test = KCI(data=self.df_cind_mul)
        self.assertTrue(test("X", "Y", ["Z1", "Z2"], significance_level=0.05))
        self.assertGreater(test.p_value_, 0.05)

        # V-structure X -> Z <- Y: dependent given Z
        rng = np.random.default_rng(seed=42)
        X = rng.normal(size=300)
        Y = rng.normal(size=300)
        Z = 2 * X + 2 * Y + rng.normal(loc=0, scale=0.3, size=300)
        df_vstruct = pd.DataFrame({"X": X, "Y": Y, "Z": Z})

        test = KCI(data=df_vstruct)
        self.assertFalse(test("X", "Y", ["Z"], significance_level=0.05))
        self.assertLess(test.p_value_, 0.05)

    def test_api_and_attributes(self):
        test = KCI(data=self.df_ind)
        test("X", "Y", [], significance_level=0.05)

        # statistic_ and p_value_ are set as floats
        self.assertIsInstance(test.statistic_, float)
        self.assertIsInstance(test.p_value_, float)

        # is_independent() returns a boolean value
        result = test.is_independent("X", "Y", [], significance_level=0.05)
        self.assertIn(result, (True, False))

        # get_ci_test factory lookup
        from pgmpy.ci_tests import get_ci_test

        test = get_ci_test(test="kci", data=self.df_ind)
        self.assertIsInstance(test, KCI)

    def test_invalid_inputs(self):
        test = KCI(data=self.df_ind)
        with self.assertRaises(ValueError):
            test("X", "X", [], significance_level=0.05)
        with self.assertRaises(ValueError):
            test("X", "Y", "Z", significance_level=0.05)
