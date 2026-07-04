import numpy as np
import pandas as pd
import pytest

from pgmpy.causal_discovery import IGCI


def _near_deterministic_data(n=500, seed=0):
    """Generate near-deterministic X -> Y data with Y = X ** 3 + tiny noise."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    y = x**3 + rng.normal(0, 1e-3, n)
    return pd.DataFrame({"X": x, "Y": y})


def test_init():
    est = IGCI(scoring="slope", ref_measure="uniform")
    assert est.get_params() == {"scoring": "slope", "ref_measure": "uniform"}


def test_fit_recovers_direction():
    est = IGCI().fit(_near_deterministic_data())
    assert list(est.causal_graph_.edges()) == [("X", "Y")]
    assert est.forward_score_ == pytest.approx(0.19854159536238275, rel=1e-4)
    assert est.backward_score_ == pytest.approx(1.3206465666212361, rel=1e-4)
    assert est.forward_score_ < est.backward_score_
    assert est.adjacency_matrix_.loc["X", "Y"] == 1
    assert est.adjacency_matrix_.loc["Y", "X"] == 0


def test_reproducible():
    data = _near_deterministic_data()
    est1 = IGCI().fit(data)
    est2 = IGCI().fit(data)
    assert est1.forward_score_ == est2.forward_score_
    assert est1.backward_score_ == est2.backward_score_


@pytest.mark.parametrize("scoring", ["slope", "entropy"])
@pytest.mark.parametrize("ref_measure", ["uniform", "gaussian"])
def test_estimators_recover_direction(scoring, ref_measure):
    est = IGCI(scoring=scoring, ref_measure=ref_measure).fit(_near_deterministic_data())
    assert list(est.causal_graph_.edges()) == [("X", "Y")]


def test_tied_observations_recover_direction():
    # Repeated X values (rounded to 2 decimals) must not break the estimators.
    rng = np.random.default_rng(1)
    x = np.round(rng.uniform(0, 1, 500), 2)
    y = x**3 + rng.normal(0, 1e-3, 500)
    data = pd.DataFrame({"X": x, "Y": y})
    for scoring in ("slope", "entropy"):
        est = IGCI(scoring=scoring).fit(data)
        assert np.isfinite(est.forward_score_) and np.isfinite(est.backward_score_)
        assert list(est.causal_graph_.edges()) == [("X", "Y")]


@pytest.mark.parametrize(
    ("estimator", "data", "match"),
    [
        (IGCI(scoring="bogus"), _near_deterministic_data(n=50), "scoring"),
        (IGCI(ref_measure="bogus"), _near_deterministic_data(n=50), "ref_measure"),
        (IGCI(), pd.DataFrame({"X": [1.0, 1.0, 1.0], "Y": [1.0, 2.0, 3.0]}), "constant"),
        (IGCI(), pd.DataFrame({"X": [0.0, 1.0, 2.0], "Y": [1.0, 2.0, 3.0], "Z": [2.0, 1.0, 0.0]}), "exactly two"),
        (IGCI(), pd.DataFrame({"X": [0.0, 1.0, np.nan], "Y": [1.0, 2.0, 3.0]}), None),
        (IGCI(), pd.DataFrame({"X": list("aabbab"), "Y": list("xyxyxy")}), "continuous"),
    ],
)
def test_invalid_input_raises(estimator, data, match):
    with pytest.raises(ValueError, match=match):
        estimator.fit(data)
