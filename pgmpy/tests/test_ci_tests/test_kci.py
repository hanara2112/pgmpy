import unittest

import numpy as np
import pandas as pd
from sklearn.gaussian_process.kernels import RBF

from pgmpy.ci_tests import KCI, get_ci_test


class TestKCI(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(seed=42)

        self.df_ind = pd.DataFrame(rng.standard_normal((200, 3)), columns=["X", "Y", "Z"])

        Z = rng.normal(size=500)
        X = 3 * Z + rng.normal(loc=0, scale=0.5, size=500)
        Y = 2 * Z + rng.normal(loc=0, scale=0.5, size=500)
        self.df_cind = pd.DataFrame({"X": X, "Y": Y, "Z": Z})

        X = rng.normal(size=300)
        Y = 2 * X + rng.normal(loc=0, scale=0.3, size=300)
        self.df_dep = pd.DataFrame({"X": X, "Y": Y})

        X = rng.normal(size=300)
        Y = X**2 + rng.normal(loc=0, scale=0.3, size=300)
        self.df_nonlinear = pd.DataFrame({"X": X, "Y": Y})

        X = rng.normal(size=300)
        Y = rng.normal(size=300)
        Z = 2 * X + 2 * Y + rng.normal(loc=0, scale=0.3, size=300)
        self.df_vstruct = pd.DataFrame({"X": X, "Y": Y, "Z": Z})

    def test_unconditional(self):
        # Independent variables
        test = KCI(data=self.df_ind)
        self.assertTrue(test("X", "Y", [], significance_level=0.05))
        self.assertTrue(test.p_value_ > 0.05)

        # Linear dependence
        test = KCI(data=self.df_dep)
        self.assertFalse(test("X", "Y", [], significance_level=0.05))
        self.assertTrue(test.p_value_ < 0.05)

        # Nonlinear dependence (KCI's advantage over Pearsonr)
        test = KCI(data=self.df_nonlinear)
        self.assertFalse(test("X", "Y", [], significance_level=0.05))
        self.assertTrue(test.p_value_ < 0.05)

    def test_conditional(self):
        # Conditionally independent: X _|_ Y | Z
        test = KCI(data=self.df_cind)
        self.assertTrue(test("X", "Y", ["Z"], significance_level=0.05))
        self.assertTrue(test.p_value_ > 0.05)

        # V-structure: conditioning on collider creates dependence
        test = KCI(data=self.df_vstruct)
        self.assertFalse(test("X", "Y", ["Z"], significance_level=0.05))
        self.assertTrue(test.p_value_ < 0.05)

    def test_custom_kernel(self):
        test = KCI(
            data=self.df_dep,
            kernel_X=RBF(length_scale=0.5),
            kernel_Y=RBF(length_scale=0.5),
        )
        self.assertFalse(test("X", "Y", [], significance_level=0.05))
        self.assertTrue(test.p_value_ < 0.05)

    def test_get_ci_test_factory(self):
        test = get_ci_test(test="kci", data=self.df_ind)
        self.assertIsInstance(test, KCI)
