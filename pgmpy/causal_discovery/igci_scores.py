from functools import partial

import numpy as np
from scipy.special import psi
from scipy.stats import differential_entropy


def slope_score(cause: np.ndarray, effect: np.ndarray) -> float:
    """Compute the IGCI mean log-slope score for one causal direction."""
    order = np.argsort(cause, kind="stable")
    cause, effect = np.asarray(cause)[order], np.asarray(effect)[order]

    run_start = np.r_[0, np.flatnonzero(np.diff(cause)) + 1]
    weights = np.diff(np.r_[run_start, cause.size])[:-1]
    cause, effect = cause[run_start], effect[run_start]

    if cause.size < 2:
        raise ValueError(
            "IGCI requires sufficiently distinct cause values after removing repetitions; no valid pairs remained."
        )

    cause_diff, effect_diff = np.diff(cause), np.diff(effect)
    valid = (cause_diff != 0) & (effect_diff != 0)
    if not valid.any():
        raise ValueError("IGCI requires sufficiently distinct continuous observations; no valid slope pairs remained.")

    return float(
        np.average(
            np.log(np.abs(effect_diff[valid] / cause_diff[valid])),
            weights=weights[valid],
        )
    )


def _spacing_entropy(values: np.ndarray) -> float:
    """Estimate entropy using the sorted-spacing estimator from IGCI."""
    values = np.sort(values)
    if values.size < 2:
        raise ValueError("IGCI entropy estimation requires at least two observations.")

    positive_deltas = np.diff(values)
    positive_deltas = positive_deltas[positive_deltas > 0]
    return float(psi(values.size) - psi(1) + np.log(positive_deltas).sum() / (values.size - 1))


def entropy_score(cause: np.ndarray, effect: np.ndarray, method: str = "auto") -> float:
    """Compute the IGCI marginal-entropy score ``H(effect) - H(cause)``."""
    if method in ("auto", "spacing"):
        cause_entropy = _spacing_entropy(np.asarray(cause))
        effect_entropy = _spacing_entropy(np.asarray(effect))
    else:
        cause_entropy = differential_entropy(cause, method=method)
        effect_entropy = differential_entropy(effect, method=method)

    if not np.isfinite(cause_entropy) or not np.isfinite(effect_entropy):
        raise ValueError(f"Entropy estimation with method {method!r} failed for the given sample.")
    return float(effect_entropy - cause_entropy)


def get_igci_score(scoring, entropy_method: str = "auto"):
    """Return the IGCI score callable selected by ``scoring``."""
    if callable(scoring):
        return scoring
    if scoring == "slope":
        return slope_score
    if scoring == "entropy":
        valid_methods = ("spacing", "vasicek", "ebrahimi", "van es", "correa", "auto")
        if entropy_method not in valid_methods:
            raise ValueError(f"entropy_method must be one of {valid_methods}. Got: {entropy_method!r}")
        return partial(entropy_score, method=entropy_method)
    raise ValueError(f"scoring must be one of ('slope', 'entropy') or a callable. Got: {scoring!r}")
