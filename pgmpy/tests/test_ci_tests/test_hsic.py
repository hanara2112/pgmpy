import numpy as np
import pandas as pd
import pytest
from sklearn.gaussian_process.kernels import RBF

from pgmpy.ci_tests import HSIC, get_ci_test


@pytest.fixture
def hsic_data():
    """Generate test dataframes for HSIC tests."""
    rng = np.random.default_rng(seed=42)

    # Draw 3 columns so rng state matches KCI fixture (X, Y independent Z consumed).
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


def _assert_independence_result(test: HSIC, expected_independent: bool) -> None:
    assert test("X", "Y", [], significance_level=0.05) == expected_independent
    assert (test.p_value_ >= 0.05) == expected_independent


def _make_causal_learn_reference_data(case: str) -> pd.DataFrame:
    if case == "independent":
        rng = np.random.default_rng(seed=10)
        return pd.DataFrame(rng.standard_normal((300, 2)), columns=["X", "Y"])

    if case == "dependent":
        rng = np.random.default_rng(seed=42)
        X = rng.normal(size=300)
        Y = 2 * X + rng.normal(scale=0.3, size=300)
        return pd.DataFrame({"X": X, "Y": Y})

    raise ValueError(f"Unknown causal-learn reference case: {case}")


class TestHSIC:
    @pytest.mark.parametrize(
        ("data_key", "kwargs", "expected_independent"),
        [
            pytest.param("ind", {}, True, id="gamma-independent"),
            pytest.param("dep", {}, False, id="gamma-linear"),
            pytest.param("nonlinear", {}, False, id="gamma-nonlinear"),
            pytest.param(
                "dep",
                {"kernel_X": RBF(length_scale=0.5), "kernel_Y": RBF(length_scale=0.5)},
                False,
                id="custom-kernel",
            ),
            pytest.param("ind", {"bandwidth": "median"}, True, id="median-independent"),
            pytest.param("dep", {"bandwidth": "median"}, False, id="median-dependent"),
            pytest.param(
                "ind",
                {"null_dist": "permutation", "n_permutations": 200},
                True,
                id="permutation-independent",
            ),
            pytest.param(
                "dep",
                {"null_dist": "permutation", "n_permutations": 200},
                False,
                id="permutation-dependent",
            ),
        ],
    )
    def test_unconditional_variants(self, hsic_data, data_key, kwargs, expected_independent):
        _assert_independence_result(HSIC(data=hsic_data[data_key], **kwargs), expected_independent)

    def test_get_ci_test_returns_hsic(self, hsic_data):
        assert isinstance(get_ci_test(test="hsic", data=hsic_data["ind"]), HSIC)

    def test_permutation_random_state_reproducible(self, hsic_data):
        kwargs = {"null_dist": "permutation", "n_permutations": 50, "random_state": 7}

        stat_1, p_value_1 = HSIC(data=hsic_data["ind"], **kwargs).run_test("X", "Y", [])
        stat_2, p_value_2 = HSIC(data=hsic_data["ind"], **kwargs).run_test("X", "Y", [])

        assert stat_1 == stat_2
        assert p_value_1 == p_value_2

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            pytest.param({"bandwidth": "invalid"}, "bandwidth must be", id="invalid-bandwidth"),
            pytest.param({"null_dist": "invalid"}, "null_dist must be", id="invalid-null-dist"),
            pytest.param({"n_permutations": 0}, "n_permutations must be", id="invalid-n-permutations"),
        ],
    )
    def test_invalid_configuration_raises_error(self, hsic_data, kwargs, message):
        with pytest.raises(ValueError, match=message):
            HSIC(data=hsic_data["ind"], **kwargs)

    def test_conditional_raises_error(self, hsic_data):
        with pytest.raises(ValueError, match="marginal independence test"):
            HSIC(data=hsic_data["ind"])("X", "Y", ["Z"])


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

    @pytest.mark.parametrize(
        ("case", "expected_stat", "expected_p", "p_abs"),
        [
            pytest.param("independent", 143.3768, 0.3597, 0.01, id="independent"),
            pytest.param("dependent", 8287.1236, 0.0, 0.001, id="dependent"),
        ],
    )
    def test_unconditional(self, case, expected_stat, expected_p, p_abs):
        test = HSIC(data=_make_causal_learn_reference_data(case))
        test.run_test("X", "Y", [])

        assert test.statistic_ == pytest.approx(expected_stat, abs=0.01)
        assert test.p_value_ == pytest.approx(expected_p, abs=p_abs)
