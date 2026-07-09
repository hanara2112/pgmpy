import numpy as np
import pandas as pd
import pytest

from pgmpy.causal_discovery import IGCI


def _near_deterministic_data(n=500, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    y = x**3 + rng.normal(0, 1e-3, n)
    return pd.DataFrame({"X": x, "Y": y})


def _reverse_direction_data(n=500, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.uniform(0, 1, n)
    x = y**3 + rng.normal(0, 1e-3, n)
    return pd.DataFrame({"X": x, "Y": y})


def test_init():
    est = IGCI(scoring="slope", ref_measure="uniform")
    assert est.get_params() == {
        "scoring": "slope",
        "ref_measure": "uniform",
        "entropy_method": "auto",
    }


def test_fit_slope():
    est = IGCI(scoring="slope").fit(_near_deterministic_data())
    assert list(est.causal_graph_.edges()) == [("X", "Y")]
    assert est.forward_score_ == pytest.approx(0.19854159536238275, rel=1e-4)
    assert est.backward_score_ == pytest.approx(1.3206465666212361, rel=1e-4)
    assert est.forward_score_ < est.backward_score_
    assert est.adjacency_matrix_.loc["X", "Y"] == 1
    assert est.adjacency_matrix_.loc["Y", "X"] == 0


def test_fit_entropy():
    est = IGCI(scoring="entropy").fit(_near_deterministic_data())
    assert list(est.causal_graph_.edges()) == [("X", "Y")]
    assert est.forward_score_ < est.backward_score_


def test_reverse_direction():
    data = _reverse_direction_data()
    for scoring in ("slope", "entropy"):
        est = IGCI(scoring=scoring).fit(data)
        assert list(est.causal_graph_.edges()) == [("Y", "X")]
        assert est.backward_score_ < est.forward_score_


def test_tied_cause_values():
    rng = np.random.default_rng(1)
    x = np.round(rng.uniform(0, 1, 500), 2)
    y = x**3 + rng.normal(0, 1e-3, 500)
    data = pd.DataFrame({"X": x, "Y": y})
    for scoring in ("slope", "entropy"):
        est = IGCI(scoring=scoring).fit(data)
        assert list(est.causal_graph_.edges()) == [("X", "Y")]


@pytest.mark.parametrize(
    ("estimator", "data", "match"),
    [
        (IGCI(scoring="bogus"), _near_deterministic_data(n=50), "scoring"),
        (IGCI(ref_measure="bogus"), _near_deterministic_data(n=50), "ref_measure"),
        (IGCI(scoring="entropy", entropy_method="bogus"), _near_deterministic_data(n=50), "entropy_method"),
        (IGCI(), pd.DataFrame({"X": [1.0, 1.0, 1.0], "Y": [1.0, 2.0, 3.0]}), "constant"),
        (IGCI(), pd.DataFrame({"X": [0.0, 1.0, 2.0], "Y": [1.0, 2.0, 3.0], "Z": [2.0, 1.0, 0.0]}), "exactly two"),
        (IGCI(), pd.DataFrame({"X": [0.0, 1.0, np.nan], "Y": [1.0, 2.0, 3.0]}), None),
        (IGCI(), pd.DataFrame({"X": list("aabbab"), "Y": list("xyxyxy")}), "continuous"),
    ],
)
def test_invalid_input_raises(estimator, data, match):
    with pytest.raises(ValueError, match=match):
        estimator.fit(data)
