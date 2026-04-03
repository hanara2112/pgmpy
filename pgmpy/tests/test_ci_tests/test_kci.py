import numpy as np
import pandas as pd
import pytest
from sklearn.gaussian_process.kernels import RBF

from pgmpy.ci_tests import HSIC, KCI, get_ci_test


@pytest.fixture
def kci_data():
    """Generate test dataframes for KCI tests."""
    rng = np.random.default_rng(seed=42)

    df_ind = pd.DataFrame(rng.standard_normal((200, 3)), columns=["X", "Y", "Z"])

    Z = rng.normal(size=500)
    X = 3 * Z + rng.normal(loc=0, scale=0.5, size=500)
    Y = 2 * Z + rng.normal(loc=0, scale=0.5, size=500)
    df_cind = pd.DataFrame({"X": X, "Y": Y, "Z": Z})

    X = rng.normal(size=300)
    Y = 2 * X + rng.normal(loc=0, scale=0.3, size=300)
    df_dep = pd.DataFrame({"X": X, "Y": Y})

    X = rng.normal(size=300)
    Y = X**2 + rng.normal(loc=0, scale=0.3, size=300)
    df_nonlinear = pd.DataFrame({"X": X, "Y": Y})

    X = rng.normal(size=300)
    Y = rng.normal(size=300)
    Z = 2 * X + 2 * Y + rng.normal(loc=0, scale=0.3, size=300)
    df_vstruct = pd.DataFrame({"X": X, "Y": Y, "Z": Z})

    return {
        "ind": df_ind,
        "cind": df_cind,
        "dep": df_dep,
        "nonlinear": df_nonlinear,
        "vstruct": df_vstruct,
    }


class TestKCI:
    def test_unconditional_fallback(self, kci_data):
        # KCI with empty Z should delegate to HSIC
        test_kci = KCI(data=kci_data["ind"])
        test_hsic = HSIC(data=kci_data["ind"])

        stat_kci, p_kci = test_kci.run_test("X", "Y", [])
        stat_hsic, p_hsic = test_hsic.run_test("X", "Y", [])

        assert stat_kci == stat_hsic
        assert p_kci == p_hsic

        # get_ci_test factory returns a KCI instance
        test = get_ci_test(test="kci", data=kci_data["ind"])
        assert isinstance(test, KCI)

    def test_conditional(self, kci_data):
        # Conditionally independent: X _|_ Y | Z
        test = KCI(data=kci_data["cind"])
        assert test("X", "Y", ["Z"], significance_level=0.05)
        assert test.p_value_ > 0.05

        # V-structure: conditioning on collider creates dependence
        test = KCI(data=kci_data["vstruct"])
        assert not test("X", "Y", ["Z"], significance_level=0.05)
        assert test.p_value_ < 0.05

        # Custom kernels in conditional test
        test = KCI(
            data=kci_data["vstruct"],
            kernel_X=RBF(length_scale=0.5),
            kernel_Y=RBF(length_scale=0.5),
            kernel_Z=RBF(length_scale=0.5),
        )
        assert not test("X", "Y", ["Z"], significance_level=0.05)

    def test_small_sample(self):
        # Exercises the n < 200 bandwidth branches
        rng = np.random.default_rng(seed=42)
        Z = rng.normal(size=100)
        X = 2 * Z + rng.normal(scale=0.5, size=100)
        Y = 3 * Z + rng.normal(scale=0.5, size=100)
        df = pd.DataFrame({"X": X, "Y": Y, "Z": Z})

        test = KCI(data=df)
        assert not test("X", "Y", [], significance_level=0.05)
        assert test("X", "Y", ["Z"], significance_level=0.05)

    def test_large_sample_and_median_bandwidth(self):
        # Exercises the n >= 1200 bandwidth branch
        rng = np.random.default_rng(seed=42)
        Z = rng.normal(size=1250)
        X = 2 * Z + rng.normal(scale=0.5, size=1250)
        Y = 3 * Z + rng.normal(scale=0.5, size=1250)
        df = pd.DataFrame({"X": X, "Y": Y, "Z": Z})

        test = KCI(data=df)
        assert test("X", "Y", ["Z"], significance_level=0.05)

        # Exercises the bandwidth="median" branch in conditional test
        test_median = KCI(data=df, bandwidth="median")
        assert test_median("X", "Y", ["Z"], significance_level=0.05)


class TestKCICompareCausalLearn:
    """Compare pgmpy KCI against causal-learn (v0.1.4.5, numpy 2.4.3).

    Reproduction code for reference values::

        import numpy as np
        from causallearn.utils.KCI.KCI import KCI_UInd, KCI_CInd
    """

    def test_conditional(self):
        # Cond. independent: causal-learn stat=2.9587, p=0.5219
        rng = np.random.default_rng(seed=7)
        Z = rng.normal(size=500)
        X = 2 * Z + rng.normal(scale=0.5, size=500)
        Y = 3 * Z + rng.normal(scale=0.5, size=500)
        df = pd.DataFrame({"X": X, "Y": Y, "Z": Z})
        test = KCI(data=df)
        test.run_test("X", "Y", ["Z"])
        assert test.statistic_ == pytest.approx(2.9587, abs=0.01)
        assert test.p_value_ == pytest.approx(0.5219, abs=0.01)

        # V-structure: causal-learn stat=1084.5896, p=0.0
        rng = np.random.default_rng(seed=42)
        X = rng.normal(size=300)
        Y = rng.normal(size=300)
        Z = 2 * X + 2 * Y + rng.normal(scale=0.3, size=300)
        df = pd.DataFrame({"X": X, "Y": Y, "Z": Z})
        test = KCI(data=df)
        test.run_test("X", "Y", ["Z"])
        assert test.statistic_ == pytest.approx(1084.5896, abs=0.01)
        assert test.p_value_ == pytest.approx(0.0, abs=0.001)
