import numpy as np
import pandas as pd
import pytest

from pgmpy.causal_discovery import IGCI


@pytest.fixture
def nonlinear_data():
    """Generate near-deterministic data with X -> Y."""
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 500)
    y = x**3 + rng.normal(0, 1e-3, 500)
    return pd.DataFrame({"X": x, "Y": y})


def test_init():
    est = IGCI()
    assert est.get_params() == {
        "score": "slope_weighted",
        "ref_measure": "uniform",
    }


def test_slope_score(nonlinear_data):
    from pgmpy.causal_discovery.bivariate_scores import SlopeScore, get_bivariate_score

    score = get_bivariate_score("slope", algorithm="igci")
    assert isinstance(score, SlopeScore)
    assert score.get_tag("name") == "slope"
    assert score.get_tag("supported_algorithms") == ["igci"]

    est = IGCI(score="slope").fit(nonlinear_data)
    assert list(est.causal_graph_.edges()) == [("X", "Y")]
    assert est.forward_score_ == pytest.approx(0.19854159536238275, rel=1e-4)
    assert est.backward_score_ == pytest.approx(1.3206465666212361, rel=1e-4)
    assert est.adjacency_matrix_.loc["X", "Y"] == 1

    with pytest.raises(ValueError, match="slope_weighted"):
        score([0, 0, 1], [0, 1, 2])


def test_weighted_slope_score():
    from pgmpy.causal_discovery.bivariate_scores import WeightedSlopeScore, get_bivariate_score

    score = get_bivariate_score("slope_weighted", algorithm="igci")
    assert isinstance(score, WeightedSlopeScore)
    assert score.get_tag("name") == "slope_weighted"
    assert score.get_tag("supported_algorithms") == ["igci"]

    rng = np.random.default_rng(1)
    x = np.round(rng.uniform(0, 1, 500), 2)
    y = x**3 + rng.normal(0, 1e-3, 500)
    est = IGCI(score="slope_weighted").fit(pd.DataFrame({"X": x, "Y": y}))
    assert list(est.causal_graph_.edges()) == [("X", "Y")]


def test_entropy_score(nonlinear_data):
    from pgmpy.causal_discovery.bivariate_scores import EntropyDifferenceScore, get_bivariate_score

    score = get_bivariate_score("entropy", algorithm="igci")
    assert isinstance(score, EntropyDifferenceScore)
    assert score.method == "spacing"
    assert score.get_tag("name") == "entropy"
    assert score.get_tag("supported_algorithms") == ["igci"]

    est = IGCI(score="entropy").fit(nonlinear_data)
    assert list(est.causal_graph_.edges()) == [("X", "Y")]

    cause = np.linspace(0.1, 1, 100)
    effect = cause**3
    natural_score = score(cause, effect)
    base_two_score = EntropyDifferenceScore(base=2)(cause, effect)
    assert base_two_score == pytest.approx(natural_score / np.log(2))
    assert np.isfinite(EntropyDifferenceScore(method="vasicek", window_length=5)(cause, effect))

    with pytest.raises(ValueError):
        IGCI(score=EntropyDifferenceScore(method="bogus")).fit(nonlinear_data)


def test_custom_score():
    from pgmpy.causal_discovery.bivariate_scores import get_bivariate_score

    score = lambda cause, effect: np.mean(effect) - np.mean(cause)
    assert get_bivariate_score(score, algorithm="igci") is score


@pytest.mark.parametrize("ref_measure", ["uniform", "gaussian"])
def test_reference_measures(nonlinear_data, ref_measure):
    est = IGCI(ref_measure=ref_measure).fit(nonlinear_data)
    assert list(est.causal_graph_.edges()) == [("X", "Y")]


def test_clone_preserves_score_instance():
    from sklearn.base import clone

    from pgmpy.causal_discovery.bivariate_scores import EntropyDifferenceScore

    cloned = clone(IGCI(score=EntropyDifferenceScore(method="vasicek")))
    assert isinstance(cloned.score, EntropyDifferenceScore)
    assert cloned.score.method == "vasicek"


@pytest.mark.parametrize(
    ("estimator", "data", "match"),
    [
        (IGCI(score="gauss"), pd.DataFrame({"X": [0, 1], "Y": [0, 1]}), "IGCI"),
        (IGCI(ref_measure="bogus"), pd.DataFrame({"X": [0, 1], "Y": [0, 1]}), "ref_measure"),
        (IGCI(), pd.DataFrame({"X": [1, 1, 1], "Y": [1, 2, 3]}), "constant"),
        (IGCI(), pd.DataFrame({"X": [0, 1], "Y": [1, 2], "Z": [2, 1]}), "exactly two"),
        (IGCI(), pd.DataFrame({"X": [0, 1, np.nan], "Y": [1, 2, 3]}), None),
        (IGCI(), pd.DataFrame({"X": list("aabbab"), "Y": list("xyxyxy")}), "continuous"),
    ],
)
def test_invalid_input_raises(estimator, data, match):
    with pytest.raises(ValueError, match=match):
        estimator.fit(data)
