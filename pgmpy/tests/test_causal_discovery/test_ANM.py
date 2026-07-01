import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from pgmpy.causal_discovery import ANM


def _nonlinear_data(n=500, seed=0):
    """Generate additive-noise data with X -> Y, where Y = X ** 3 + noise."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, n)
    y = x**3 + rng.normal(0, 0.5, n)
    return pd.DataFrame({"X": x, "Y": y})


def test_init():
    est = ANM(random_state=0)
    assert est.get_params() == {"regressor": None, "ci_test": None, "random_state": 0}


def test_fit_recovers_direction():
    data = _nonlinear_data()
    est = ANM(random_state=0).fit(data)

    assert list(est.causal_graph_.edges()) == [("X", "Y")]
    assert est.forward_score_ == pytest.approx(0.0003224986489045623, rel=1e-4)
    assert est.backward_score_ == pytest.approx(0.003947076565539923, rel=1e-4)
    assert est.forward_score_ < est.backward_score_
    assert est.adjacency_matrix_.loc["X", "Y"] == 1
    assert est.adjacency_matrix_.loc["Y", "X"] == 0


def test_reproducible():
    data = _nonlinear_data()
    est1 = ANM(random_state=0).fit(data)
    est2 = ANM(random_state=0).fit(data)
    assert est1.forward_score_ == est2.forward_score_
    assert est1.backward_score_ == est2.backward_score_


def test_regressor_and_ci_test_override():
    data = _nonlinear_data()
    est = ANM(regressor=LinearRegression(), ci_test="gcm", random_state=0).fit(data)
    # A linear regressor cannot capture Y = X**3, so gcm finds near-zero residual
    # dependence in both directions (no direction is meaningfully preferred).
    assert est.forward_score_ == pytest.approx(0.0, abs=1e-9)
    assert est.backward_score_ == pytest.approx(0.0, abs=1e-9)


def test_constant_input_raises():
    data = pd.DataFrame({"X": [1.0, 1.0, 1.0], "Y": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="constant"):
        ANM().fit(data)


def test_wrong_number_of_variables_raises():
    data = pd.DataFrame({"X": [0.0, 1.0, 2.0], "Y": [1.0, 2.0, 3.0], "Z": [2.0, 1.0, 0.0]})
    with pytest.raises(ValueError, match="exactly two variables"):
        ANM().fit(data)


def test_non_finite_input_raises():
    data = pd.DataFrame({"X": [0.0, 1.0, np.nan], "Y": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError):
        ANM().fit(data)


def test_non_continuous_input_raises():
    data = pd.DataFrame({"X": list("aabbab"), "Y": list("xyxyxy")})
    with pytest.raises(ValueError, match="continuous"):
        ANM().fit(data)
