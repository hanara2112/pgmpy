import numpy as np
import pandas as pd
import pytest
from sklearn.gaussian_process.kernels import RBF

from pgmpy.ci_tests import HSIC, KCI, get_ci_test


@pytest.fixture
def kci_data():
    rng = np.random.default_rng(seed=42)

    df_ind = pd.DataFrame(rng.standard_normal((200, 3)), columns=["X", "Y", "Z"])

    Z = rng.normal(size=500)
    X = 3 * Z + rng.normal(scale=0.5, size=500)
    Y = 2 * Z + rng.normal(scale=0.5, size=500)
    df_cind = pd.DataFrame({"X": X, "Y": Y, "Z": Z})

    X = rng.normal(size=300)
    Y = rng.normal(size=300)
    Z = 2 * X + 2 * Y + rng.normal(scale=0.3, size=300)
    df_vstruct = pd.DataFrame({"X": X, "Y": Y, "Z": Z})

    return df_ind, df_cind, df_vstruct


class TestKCI:
    def test_unconditional_fallback(self, kci_data):
        df_ind, _, _ = kci_data
        stat_kci, p_kci = KCI(data=df_ind).run_test("X", "Y", [])
        stat_hsic, p_hsic = HSIC(data=df_ind).run_test("X", "Y", [])

        assert stat_kci == stat_hsic
        assert p_kci == p_hsic
        assert isinstance(get_ci_test(test="kci", data=df_ind), KCI)

    def test_conditional(self, kci_data):
        _, df_cind, df_vstruct = kci_data

        # Conditionally independent: X _|_ Y | Z
        test = KCI(data=df_cind)
        assert test("X", "Y", ["Z"], significance_level=0.05)
        assert test.statistic_ == pytest.approx(2.8415, abs=0.01)
        assert test.p_value_ == pytest.approx(0.4719, abs=0.01)

        # V-structure: conditioning on collider induces dependence
        test = KCI(data=df_vstruct)
        assert not test("X", "Y", ["Z"], significance_level=0.05)
        assert test.statistic_ == pytest.approx(1009.6465, abs=0.01)
        assert test.p_value_ == pytest.approx(0.0, abs=0.001)

    def test_custom_kernels(self, kci_data):
        _, _, df_vstruct = kci_data
        # Single kernel shared by X, Y, Z.
        assert not KCI(data=df_vstruct, kernel=RBF(0.5))("X", "Y", ["Z"], significance_level=0.05)
        # Tuple: separate kernels for X, Y, Z.
        assert not KCI(data=df_vstruct, kernel=(RBF(0.5), RBF(0.5), RBF(0.5)))("X", "Y", ["Z"], significance_level=0.05)

    def test_sample_size_and_kci_bandwidth(self):
        rng = np.random.default_rng(seed=42)
        Z = rng.normal(size=100)
        X = 2 * Z + rng.normal(scale=0.5, size=100)
        Y = 3 * Z + rng.normal(scale=0.5, size=100)
        df_small = pd.DataFrame({"X": X, "Y": Y, "Z": Z})
        test = KCI(data=df_small)
        assert not test("X", "Y", [], significance_level=0.05)
        assert test("X", "Y", ["Z"], significance_level=0.05)

        rng = np.random.default_rng(seed=42)
        Z = rng.normal(size=1250)
        X = 2 * Z + rng.normal(scale=0.5, size=1250)
        Y = 3 * Z + rng.normal(scale=0.5, size=1250)
        df_large = pd.DataFrame({"X": X, "Y": Y, "Z": Z})
        assert KCI(data=df_large)("X", "Y", ["Z"], significance_level=0.05)
        assert KCI(data=df_large, bandwidth="median")("X", "Y", ["Z"], significance_level=0.05)


class TestKCICompareCausalLearn:
    """Cross-validate against causal-learn v0.1.4.5 (numpy 2.4.3).

    Reproduction code for reference values::

        import numpy as np
        from causallearn.utils.KCI.KCI import KCI_CInd

        rng = np.random.default_rng(seed=7)
        Z = rng.normal(size=500)
        X = 2*Z + rng.normal(scale=0.5, size=500)
        Y = 3*Z + rng.normal(scale=0.5, size=500)
        KCI_CInd().compute_pvalue(X[:,None], Y[:,None], Z[:,None])  # stat=2.9587, p=0.5219

        rng = np.random.default_rng(seed=42)
        X = rng.normal(size=300); Y = rng.normal(size=300)
        Z = 2*X + 2*Y + rng.normal(scale=0.3, size=300)
        KCI_CInd().compute_pvalue(X[:,None], Y[:,None], Z[:,None])  # stat=1084.59, p=0.0
    """

    def test_matches_causal_learn_kci_cind(self):
        rng = np.random.default_rng(seed=7)
        Z = rng.normal(size=500)
        X = 2 * Z + rng.normal(scale=0.5, size=500)
        Y = 3 * Z + rng.normal(scale=0.5, size=500)
        df = pd.DataFrame({"X": X, "Y": Y, "Z": Z})
        test = KCI(data=df)
        test.run_test("X", "Y", ["Z"])
        assert test.statistic_ == pytest.approx(2.9587, abs=0.01)
        assert test.p_value_ == pytest.approx(0.5219, abs=0.01)

        rng = np.random.default_rng(seed=42)
        X = rng.normal(size=300)
        Y = rng.normal(size=300)
        Z = 2 * X + 2 * Y + rng.normal(scale=0.3, size=300)
        df = pd.DataFrame({"X": X, "Y": Y, "Z": Z})
        test = KCI(data=df)
        test.run_test("X", "Y", ["Z"])
        assert test.statistic_ == pytest.approx(1084.5896, abs=0.01)
        assert test.p_value_ == pytest.approx(0.0, abs=0.001)
