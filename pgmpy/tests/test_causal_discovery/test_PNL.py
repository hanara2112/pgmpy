import importlib

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, clone

from pgmpy.causal_discovery import PNL
from pgmpy.causal_discovery.PNL import NormFlow


class PolynomialDisturbanceEstimator(BaseEstimator):
    """Small deterministic estimator used to test PNL orchestration."""

    def fit(self, X, y=None):
        X = np.asarray(X)
        self.function_ = np.polynomial.Polynomial.fit(X[:, 0], X[:, 1], deg=3)
        return self

    def predict(self, X):
        X = np.asarray(X)
        return X[:, 1] - self.function_(X[:, 0])


class NonFiniteDisturbanceEstimator(BaseEstimator):
    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.full(len(X), np.nan)


class SeededPolynomialDisturbanceEstimator(PolynomialDisturbanceEstimator):
    def __init__(self, seed=None):
        self.seed = seed


def _squared_dependence_score(cause, disturbance):
    return abs(np.corrcoef(np.asarray(cause) ** 2, np.asarray(disturbance) ** 2)[0, 1])


@pytest.fixture
def nonlinear_data():
    rng = np.random.default_rng(0)
    x = rng.uniform(-2.0, 2.0, 500)
    y = x**3 + 0.2 * rng.normal(size=x.size)
    return pd.DataFrame({"X": x, "Y": y})


def test_init():
    assert PNL().get_params() == {
        "estimator": None,
        "scoring_method": "independence",
        "seed": None,
    }


def test_fit_recovers_direction(nonlinear_data):
    est = PNL(
        estimator=PolynomialDisturbanceEstimator(),
        scoring_method=_squared_dependence_score,
    ).fit(nonlinear_data)

    assert list(est.causal_graph_.edges()) == [("X", "Y")]
    assert est.forward_score_ < est.backward_score_
    assert hasattr(est, "forward_estimator_")
    assert hasattr(est, "backward_estimator_")
    assert est.adjacency_matrix_.loc["X", "Y"] == 1
    assert est.score(true_graph=est.causal_graph_, metric="SHD") == 0


def test_non_finite_disturbance_raises():
    data = pd.DataFrame({"X": [0.0, 1.0, 2.0], "Y": [1.0, 2.0, 4.0]})

    with pytest.raises(ValueError, match="non-finite disturbance"):
        PNL(estimator=NonFiniteDisturbanceEstimator()).fit(data)


@pytest.mark.parametrize(
    ("score", "match"),
    [
        (np.nan, "dependence score is non-finite"),
        (0.0, "both directions produced the same score"),
    ],
)
def test_non_finite_or_tied_direction_scores_raise(nonlinear_data, score, match):
    with pytest.raises(ValueError, match=match):
        PNL(
            estimator=PolynomialDisturbanceEstimator(),
            scoring_method=lambda cause, disturbance: score,
        ).fit(nonlinear_data)


def test_clone_preserves_scoring_method_instance():
    from pgmpy.causal_discovery.bivariate_scores import IndependenceScore

    cloned = clone(PNL(scoring_method=IndependenceScore(criterion="p_value")))

    assert isinstance(cloned.scoring_method, IndependenceScore)
    assert cloned.scoring_method.criterion == "p_value"


def test_seed_is_passed_to_default_estimators(monkeypatch, nonlinear_data):
    pnl_module = importlib.import_module("pgmpy.causal_discovery.PNL")
    monkeypatch.setattr(pnl_module, "NormFlow", SeededPolynomialDisturbanceEstimator)

    est = PNL(seed=42, scoring_method=_squared_dependence_score).fit(nonlinear_data)

    assert est.forward_estimator_.seed == 42
    assert est.backward_estimator_.seed == 42


def test_normflow_breaks_unit_symmetry_and_preserves_rng_state():
    torch = pytest.importorskip("torch")
    x = np.linspace(-1.0, 1.0, 32)
    data = np.column_stack((x, x**3))
    torch.manual_seed(123)
    rng_state = torch.random.get_rng_state().clone()

    first = NormFlow(hidden_dim=4, n_components=3, max_iter=3, seed=7).fit(data)
    second = NormFlow(hidden_dim=4, n_components=3, max_iter=3, seed=7).fit(data)

    assert torch.equal(torch.random.get_rng_state(), rng_state)
    assert float(first.post_bias_.detach().std()) > 0
    assert float(first.post_input_weight_.detach().std()) > 0
    assert float(first.post_output_weight_.detach().std()) > 0
    _, derivative = first._post_transform(torch.tensor([[0.0]]))
    assert torch.all(derivative > 0)
    np.testing.assert_allclose(first.predict(data), second.predict(data))
