import numpy as np
import pandas as pd
import pytest

from pgmpy.causal_discovery import IGCI


def make_data(n=500, seed=0):
    # near-deterministic X -> Y
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    y = x**3 + rng.normal(0, 1e-3, n)
    return pd.DataFrame({"X": x, "Y": y})


def test_init():
    est = IGCI(scoring="slope", ref_measure="uniform", random_state=0)
    assert est.get_params() == {"scoring": "slope", "ref_measure": "uniform", "random_state": 0}


def test_fit_recovers_direction():
    est = IGCI().fit(make_data())
    assert list(est.causal_graph_.edges()) == [("X", "Y")]
    assert est.direction_score_ > 0
    assert est.adjacency_matrix_.loc["X", "Y"] == 1
    assert est.adjacency_matrix_.loc["Y", "X"] == 0


def test_reproducible():
    data = make_data()
    assert IGCI().fit(data).direction_score_ == IGCI().fit(data).direction_score_


def test_entropy_scoring():
    est = IGCI(scoring="entropy", ref_measure="gaussian").fit(make_data())
    assert list(est.causal_graph_.edges()) == [("X", "Y")]


def test_tied_observations():
    rng = np.random.default_rng(1)
    x = rng.integers(0, 5, 200).astype(float)
    y = np.exp(x) + rng.normal(0, 1e-3, 200)
    est = IGCI().fit(pd.DataFrame({"X": x, "Y": y}))
    assert np.isfinite(est.direction_score_)


def test_invalid_args_raise():
    data = make_data(n=50)
    with pytest.raises(ValueError, match="scoring"):
        IGCI(scoring="bogus").fit(data)
    with pytest.raises(ValueError, match="ref_measure"):
        IGCI(ref_measure="bogus").fit(data)


def test_constant_input_raises():
    data = pd.DataFrame({"X": [1.0, 1.0, 1.0], "Y": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="constant"):
        IGCI().fit(data)


def test_wrong_number_of_variables_raises():
    data = pd.DataFrame({"X": [0.0, 1.0, 2.0], "Y": [1.0, 2.0, 3.0], "Z": [2.0, 1.0, 0.0]})
    with pytest.raises(ValueError, match="exactly two variables"):
        IGCI().fit(data)


def test_non_continuous_input_raises():
    data = pd.DataFrame({"X": list("aabbab"), "Y": list("xyxyxy")})
    with pytest.raises(ValueError, match="continuous"):
        IGCI().fit(data)
