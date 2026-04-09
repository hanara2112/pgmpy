import numpy as np
import pandas as pd
import pytest
from sklearn.gaussian_process.kernels import RBF

from pgmpy.ci_tests import HSIC, get_ci_test


@pytest.fixture
def hsic_data():
    """Generate test dataframes for HSIC tests."""
    rng = np.random.default_rng(seed=42)

    # Draw 3 columns so rng state matches KCI fixture (X, Y independent Z consumed)
    df_ind = pd.DataFrame(rng.standard_normal((200, 3)), columns=["X", "Y", "Z"])

    X = rng.normal(size=300)
    Y = 2 * X + rng.normal(loc=0, scale=0.3, size=300)
    df_dep = pd.DataFrame({"X": X, "Y": Y})

    X = rng.normal(size=300)
    Y = X**2 + rng.normal(loc=0, scale=0.3, size=300)
    df_nonlinear = pd.DataFrame({"X": X, "Y": Y})

    return {
        "ind": df_ind,
        "dep": df_dep,
        "nonlinear": df_nonlinear,
    }


class TestHSIC:
    def test_unconditional(self, hsic_data):
        # Independent variables — Gamma approximation (default)
        test = HSIC(data=hsic_data["ind"])
        assert test("X", "Y", [], significance_level=0.05)
        assert test.p_value_ > 0.05

        # Linear dependence detected
        test = HSIC(data=hsic_data["dep"])
        assert not test("X", "Y", [], significance_level=0.05)
        assert test.p_value_ < 0.05

        # Nonlinear dependence detected (HSIC's advantage over Pearsonr)
        test = HSIC(data=hsic_data["nonlinear"])
        assert not test("X", "Y", [], significance_level=0.05)
        assert test.p_value_ < 0.05

        # Custom kernel still detects dependence
        test = HSIC(data=hsic_data["dep"], kernel_X=RBF(length_scale=0.5), kernel_Y=RBF(length_scale=0.5))
        assert not test("X", "Y", [], significance_level=0.05)

        # get_ci_test factory returns an HSIC instance
        test = get_ci_test(test="hsic", data=hsic_data["ind"])
        assert isinstance(test, HSIC)

    def test_bandwidth_median(self, hsic_data):
        # Median heuristic should detect independence and dependence as well
        test = HSIC(data=hsic_data["ind"], bandwidth="median")
        assert test("X", "Y", [], significance_level=0.05)

        test = HSIC(data=hsic_data["dep"], bandwidth="median")
        assert not test("X", "Y", [], significance_level=0.05)

    def test_null_dist_permutation(self, hsic_data):
        # Permutation test should agree with Gamma for clear-cut cases
        test = HSIC(data=hsic_data["ind"], null_dist="permutation", n_permutations=200)
        assert test("X", "Y", [], significance_level=0.05)

        test = HSIC(data=hsic_data["dep"], null_dist="permutation", n_permutations=200)
        assert not test("X", "Y", [], significance_level=0.05)

    def test_permutation_random_state_reproducible(self, hsic_data):
        test_1 = HSIC(data=hsic_data["ind"], null_dist="permutation", n_permutations=50, random_state=7)
        stat_1, p_value_1 = test_1.run_test("X", "Y", [])

        test_2 = HSIC(data=hsic_data["ind"], null_dist="permutation", n_permutations=50, random_state=7)
        stat_2, p_value_2 = test_2.run_test("X", "Y", [])

        assert stat_1 == stat_2
        assert p_value_1 == p_value_2

    def test_invalid_configuration_raises_error(self, hsic_data):
        with pytest.raises(ValueError, match="bandwidth must be"):
            HSIC(data=hsic_data["ind"], bandwidth="invalid")

        with pytest.raises(ValueError, match="null_dist must be"):
            HSIC(data=hsic_data["ind"], null_dist="invalid")

        with pytest.raises(ValueError, match="n_permutations must be"):
            HSIC(data=hsic_data["ind"], n_permutations=0)

    def test_conditional_raises_error(self, hsic_data):
        # HSIC does not support a conditioning set
        df = hsic_data["ind"].copy()
        test = HSIC(data=df)
        with pytest.raises(ValueError, match="marginal independence test"):
            test("X", "Y", ["Z"])


class TestHSICCompareCausalLearn:
    """Compare pgmpy HSIC (Gamma path) against causal-learn (v0.1.4.5, numpy 2.4.3).

    Reproduction code for reference values::

        import numpy as np
        from causallearn.utils.KCI.KCI import KCI_UInd

        rng = np.random.default_rng(seed=10)
        data = rng.standard_normal((300, 2))
        KCI_UInd().compute_pvalue(data[:, 0:1], data[:, 1:2])   # stat=143.38, p=0.36

        rng = np.random.default_rng(seed=42)
        X = rng.normal(size=300); Y = 2*X + rng.normal(scale=0.3, size=300)
        KCI_UInd().compute_pvalue(X[:, None], Y[:, None])        # stat=8287.12, p=0.0
    """

    def test_unconditional(self):
        # Independent: causal-learn stat=143.3768, p=0.3597
        rng = np.random.default_rng(seed=10)
        df = pd.DataFrame(rng.standard_normal((300, 2)), columns=["X", "Y"])
        test = HSIC(data=df)
        test.run_test("X", "Y", [])
        assert test.statistic_ == pytest.approx(143.3768, abs=0.01)
        assert test.p_value_ == pytest.approx(0.3597, abs=0.01)

        # Dependent: causal-learn stat=8287.1236, p=0.0
        rng = np.random.default_rng(seed=42)
        X = rng.normal(size=300)
        Y = 2 * X + rng.normal(scale=0.3, size=300)
        df = pd.DataFrame(np.column_stack([X, Y]), columns=["X", "Y"])
        test = HSIC(data=df)
        test.run_test("X", "Y", [])
        assert test.statistic_ == pytest.approx(8287.1236, abs=0.01)
        assert test.p_value_ == pytest.approx(0.0, abs=0.001)
