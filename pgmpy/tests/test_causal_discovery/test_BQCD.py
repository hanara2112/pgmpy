import numpy as np
import pandas as pd
import pytest

from pgmpy.causal_discovery import BQCD


class EmpiricalQuantileRegressor:
    """Minimal regressor used to exercise the custom quantile-regressor API."""

    def __init__(self, quantile):
        self.quantile = quantile

    def fit(self, X, y):
        self.prediction_ = float(np.quantile(y, self.quantile))
        return self

    def predict(self, X):
        return np.full(len(X), self.prediction_)


class NonFiniteQuantileRegressor:
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.full(len(X), np.nan)


def _nonlinear_data(n=800, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = x + (0.05 + 2.0 * x**2) * rng.normal(size=n)
    return x, y


@pytest.mark.parametrize(("n", "seed"), [(150, 0), (150, 1), (400, 0), (400, 1)])
def test_fit_recovers_direction(n, seed):
    x, y = _nonlinear_data(n=n, seed=seed)
    data = pd.DataFrame({"X": x, "Y": y})

    est = BQCD(n_quantiles=3, seed=seed).fit(data)

    assert list(est.causal_graph_.edges()) == [("X", "Y")]
    assert est.forward_score_ < est.backward_score_
    assert np.isfinite(est.forward_score_)
    assert np.isfinite(est.backward_score_)
    expected_quantiles = 1 if n < 200 else 3
    assert len(est.quantile_levels_) == expected_quantiles
    assert len(est.forward_estimators_) == expected_quantiles
    normalizer = est.score_components_["marginal_x"] + est.score_components_["marginal_y"]
    assert (
        est.forward_score_
        == (est.score_components_["marginal_x"] + est.score_components_["conditional_y_given_x"]) / normalizer
    )
    assert (
        est.backward_score_
        == (est.score_components_["marginal_y"] + est.score_components_["conditional_x_given_y"]) / normalizer
    )
    assert est.adjacency_matrix_.loc["X", "Y"] == 1
    assert est.adjacency_matrix_.loc["Y", "X"] == 0


def test_fit_recovers_backward_direction():
    x, y = _nonlinear_data()
    data = pd.DataFrame({"Y": y, "X": x})

    est = BQCD(n_quantiles=3, seed=0).fit(data)

    assert list(est.causal_graph_.edges()) == [("X", "Y")]
    assert est.backward_score_ < est.forward_score_


@pytest.mark.parametrize("n_quantiles", [0, -1, 1.5, True])
def test_invalid_n_quantiles(n_quantiles):
    data = pd.DataFrame({"X": [0.0, 1.0], "Y": [1.0, 2.0]})

    with pytest.raises(ValueError, match="positive integer"):
        BQCD(n_quantiles=n_quantiles).fit(data)


def test_numpy_integer_n_quantiles():
    x, y = _nonlinear_data(n=200)
    est = BQCD(n_quantiles=np.int64(2), seed=0).fit(pd.DataFrame({"X": x, "Y": y}))

    assert len(est.quantile_levels_) == 2


def test_constant_column_raises():
    data = pd.DataFrame({"X": [1.0, 1.0, 1.0], "Y": [1.0, 2.0, 3.0]})

    with pytest.raises(ValueError, match="constant"):
        BQCD().fit(data)


def test_custom_quantile_regressor():
    calls = []

    def quantile_regressor(quantile):
        calls.append(float(quantile))
        return EmpiricalQuantileRegressor(quantile)

    x, y = _nonlinear_data(n=200)
    est = BQCD(n_quantiles=2, quantile_regressor=quantile_regressor).fit(pd.DataFrame({"X": x, "Y": y}))

    assert len(calls) == 4
    assert all(isinstance(regressor, EmpiricalQuantileRegressor) for regressor in est.forward_estimators_)
    assert all(isinstance(regressor, EmpiricalQuantileRegressor) for regressor in est.backward_estimators_)


def test_non_finite_quantile_predictions_raise():
    x, y = _nonlinear_data(n=200)
    data = pd.DataFrame({"X": x, "Y": y})

    with pytest.raises(ValueError, match="non-finite predictions"):
        BQCD(quantile_regressor=lambda quantile: NonFiniteQuantileRegressor()).fit(data)


def test_small_sample_uses_single_quantile():
    x, y = _nonlinear_data(n=199)
    est = BQCD(n_quantiles=7, seed=0).fit(pd.DataFrame({"X": x, "Y": y}))

    assert est.quantile_levels_ == pytest.approx([0.5])
    assert est.quadrature_weights_ == pytest.approx([1.0])
    assert len(est.forward_estimators_) == 1
    assert len(est.backward_estimators_) == 1


def test_fixed_seed_is_reproducible():
    x, y = _nonlinear_data(n=200, seed=2)
    data = pd.DataFrame({"X": x, "Y": y})

    first = BQCD(n_quantiles=3, seed=7).fit(data)
    second = BQCD(n_quantiles=3, seed=7).fit(data)

    assert first.forward_score_ == second.forward_score_
    assert first.backward_score_ == second.backward_score_
    assert list(first.causal_graph_.edges()) == list(second.causal_graph_.edges())
