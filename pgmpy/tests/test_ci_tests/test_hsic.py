import numpy as np
import pandas as pd
import pytest
from sklearn.gaussian_process.kernels import RBF

from pgmpy.ci_tests import HSIC


@pytest.fixture
def hsic_data():
    rng = np.random.default_rng(seed=42)

    df_ind = pd.DataFrame(rng.standard_normal((200, 3)), columns=["X", "Y", "Z"])

    X = rng.normal(size=300)
    Y = 2 * X + rng.normal(scale=0.3, size=300)
    df_dep = pd.DataFrame({"X": X, "Y": Y})

    X = rng.normal(size=300)
    Y = X**2 + rng.normal(scale=0.3, size=300)
    df_nonlinear = pd.DataFrame({"X": X, "Y": Y})

    return df_ind, df_dep, df_nonlinear


class TestHSIC:
    def test_independent(self, hsic_data):
        df_ind, _, _ = hsic_data
        test = HSIC(data=df_ind)
        assert test("X", "Y", [], significance_level=0.05)
        assert test.p_value_ > 0.05

    def test_linear_dependence(self, hsic_data):
        _, df_dep, _ = hsic_data
        test = HSIC(data=df_dep)
        assert not test("X", "Y", [], significance_level=0.05)
        assert test.p_value_ < 0.05

    def test_nonlinear_dependence(self, hsic_data):
        _, _, df_nonlinear = hsic_data
        test = HSIC(data=df_nonlinear)
        assert not test("X", "Y", [], significance_level=0.05)
        assert test.p_value_ < 0.05

    def test_median_bandwidth(self, hsic_data):
        df_ind, df_dep, _ = hsic_data
        assert HSIC(data=df_ind, bandwidth="median")("X", "Y", [], significance_level=0.05)
        assert not HSIC(data=df_dep, bandwidth="median")("X", "Y", [], significance_level=0.05)

    def test_custom_kernel(self, hsic_data):
        _, df_dep, _ = hsic_data
        test = HSIC(data=df_dep, kernel_X=RBF(0.5), kernel_Y=RBF(0.5))
        assert not test("X", "Y", [], significance_level=0.05)

    def test_permutation(self, hsic_data):
        df_ind, df_dep, _ = hsic_data
        assert HSIC(data=df_ind, null_dist="permutation", n_permutations=200, random_state=0)(
            "X", "Y", [], significance_level=0.05
        )
        assert not HSIC(data=df_dep, null_dist="permutation", n_permutations=200, random_state=0)(
            "X", "Y", [], significance_level=0.05
        )

    def test_small_n(self):
        df = pd.DataFrame({"X": [1.0, 2.0, 3.0, 4.0, 5.0], "Y": [5.0, 4.0, 3.0, 2.0, 1.0]})
        _, p = HSIC(data=df).run_test("X", "Y", [])
        assert p == 1.0

    def test_conditioning_not_supported(self, hsic_data):
        df_ind, _, _ = hsic_data
        with pytest.raises(ValueError, match="does not support conditioning"):
            HSIC(data=df_ind)("X", "Y", ["Z"])

    def test_invalid_params(self, hsic_data):
        df_ind, _, _ = hsic_data
        with pytest.raises(ValueError, match="bandwidth"):
            HSIC(data=df_ind, bandwidth="bad")
        with pytest.raises(ValueError, match="null_dist"):
            HSIC(data=df_ind, null_dist="bad")
        with pytest.raises(ValueError, match="n_permutations"):
            HSIC(data=df_ind, n_permutations=0)


class TestHSICCompareCausalLearn:
    """Cross-validate against causal-learn v0.1.4.5 (numpy 2.4.3).

    Reference values reproduced with::

        import numpy as np
        from causallearn.utils.KCI.KCI import KCI_UInd

        rng = np.random.default_rng(seed=10)
        data = rng.standard_normal((300, 2))
        KCI_UInd().compute_pvalue(data[:, 0:1], data[:, 1:2])   # stat=143.38, p=0.36

        rng = np.random.default_rng(seed=42)
        X = rng.normal(size=300); Y = 2*X + rng.normal(scale=0.3, size=300)
        KCI_UInd().compute_pvalue(X[:, None], Y[:, None])        # stat=8287.12, p=0.0
    """

    def test_independent(self):
        rng = np.random.default_rng(seed=10)
        df = pd.DataFrame(rng.standard_normal((300, 2)), columns=["X", "Y"])
        test = HSIC(data=df)
        test.run_test("X", "Y", [])

        assert test.statistic_ == pytest.approx(143.3768, abs=0.01)
        assert test.p_value_ == pytest.approx(0.3597, abs=0.01)

    def test_dependent(self):
        rng = np.random.default_rng(seed=42)
        X = rng.normal(size=300)
        Y = 2 * X + rng.normal(scale=0.3, size=300)
        df = pd.DataFrame({"X": X, "Y": Y})
        test = HSIC(data=df)
        test.run_test("X", "Y", [])

        assert test.statistic_ == pytest.approx(8287.1236, abs=0.01)
        assert test.p_value_ == pytest.approx(0.0, abs=0.001)
