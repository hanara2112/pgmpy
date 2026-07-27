import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator

from pgmpy.causal_discovery import PNL


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


def _squared_dependence_score(cause, disturbance):
    return abs(np.corrcoef(np.asarray(cause) ** 2, np.asarray(disturbance) ** 2)[0, 1])


def test_init():
    assert PNL().get_params() == {"estimator": None, "score": "independence"}


def test_fit_recovers_direction():
    rng = np.random.default_rng(0)
    x = rng.uniform(-2.0, 2.0, 500)
    y = x**3 + 0.2 * rng.normal(size=x.size)
    data = pd.DataFrame({"X": x, "Y": y})

    est = PNL(estimator=PolynomialDisturbanceEstimator(), score=_squared_dependence_score).fit(data)

    assert list(est.causal_graph_.edges()) == [("X", "Y")]
    assert est.forward_score_ < est.backward_score_
    assert hasattr(est, "forward_estimator_")
    assert hasattr(est, "backward_estimator_")
    assert est.adjacency_matrix_.loc["X", "Y"] == 1


def test_non_finite_disturbance_raises():
    data = pd.DataFrame({"X": [0.0, 1.0, 2.0], "Y": [1.0, 2.0, 4.0]})

    with pytest.raises(ValueError, match="non-finite disturbance"):
        PNL(estimator=NonFiniteDisturbanceEstimator()).fit(data)
