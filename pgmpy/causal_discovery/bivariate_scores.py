"""Scoring methods for bivariate causal discovery.

This module contains the built-in scores used by ANM and IGCI. Each score defines its name and
supported algorithms, and users can also pass configured score objects or custom callables.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.special import psi
from scipy.stats import differential_entropy
from skbase.base import BaseObject
from skbase.lookup import all_objects

from pgmpy.ci_tests import get_ci_test


def _ensure_finite(value, name="Score") -> float:
    """Return ``value`` as a float and raise if it is not finite."""
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} is not finite for the given sample.")
    return value


def _estimate_entropy(values, method="auto", window_length=None, base=None) -> float:
    """Estimate differential entropy using a spacing-based or SciPy estimator."""
    values = np.asarray(values)

    if base is not None and (base <= 0 or base == 1):
        raise ValueError("base must be positive and not equal to 1.")

    if method != "spacing":
        entropy = differential_entropy(
            values,
            method=method,
            window_length=window_length,
            base=base,
        )
        return _ensure_finite(entropy, name="Entropy estimate")

    if window_length is not None:
        raise ValueError("window_length is not supported by the spacing estimator.")

    values = np.sort(values)
    if values.size < 2:
        raise ValueError("Entropy estimation requires at least two observations.")

    deltas = np.diff(values)
    deltas = deltas[deltas > 0]
    if deltas.size == 0:
        raise ValueError("Spacing entropy requires distinct observations.")

    entropy = psi(values.size) - psi(1) + np.log(deltas).sum() / (values.size - 1)
    if base is not None:
        entropy /= np.log(base)
    return _ensure_finite(entropy, name="Entropy estimate")


def _estimate_pair_entropies(x, y, method="auto", window_length=None, base=None):
    """Estimate the marginal entropies of two samples."""
    kwargs = {"method": method, "window_length": window_length, "base": base}
    return _estimate_entropy(x, **kwargs), _estimate_entropy(y, **kwargs)


class BaseBivariateScore(BaseObject):
    """Base class for a score that compares two one-dimensional samples.

    Concrete subclasses define ``name`` and ``supported_algorithms`` tags for built-in lookup.
    """

    _tags = {"name": None, "supported_algorithms": []}

    def __call__(self, x, y) -> float:
        """Compute the score for two samples.

        Parameters
        ----------
        x : array-like
            First one-dimensional sample.
        y : array-like
            Second one-dimensional sample.

        Returns
        -------
        float
            Score for the two samples.
        """
        raise NotImplementedError


class IndependenceScore(BaseBivariateScore):
    """
    Dependence score from a conditional-independence test.

    Runs an unconditional CI test between ``x`` and ``y``. A smaller score means weaker dependence.

    Parameters
    ----------
    ci_test : str or pgmpy.ci_tests.BaseCITest, default="pearsonr"
        The independence test, resolved via :func:`pgmpy.ci_tests.get_ci_test`.

    criterion : {"effect_size", "statistic", "p_value"}, default="effect_size"
        CI-test output to return. Each option is transformed so that smaller is better:

        - ``"effect_size"`` -- the test's ``effect_size_`` (a non-negative dependence magnitude);
        - ``"statistic"`` -- the absolute test statistic ``|statistic_|``;
        - ``"p_value"`` -- the negative p-value ``-p_value_`` (a larger p-value, i.e. more evidence
          of independence, gives a smaller score).
    """

    _tags = {"name": "independence", "supported_algorithms": ["anm"]}

    def __init__(self, ci_test="pearsonr", criterion="effect_size"):
        self.ci_test = ci_test
        self.criterion = criterion
        super().__init__()

    def __call__(self, x, y) -> float:
        data = pd.DataFrame({"_x": np.asarray(x), "_y": np.asarray(y)})
        test = get_ci_test(test=self.ci_test, data=data)
        test.run_test("_x", "_y", Z=[])

        if self.criterion == "effect_size":
            score = test.effect_size_
        elif self.criterion == "statistic":
            score = abs(test.statistic_)
        elif self.criterion == "p_value":
            score = -test.p_value_
        else:
            raise ValueError(
                f"Unknown criterion: {self.criterion!r}. Must be one of 'effect_size', 'statistic', 'p_value'."
            )
        return _ensure_finite(score)


class EntropyScore(BaseBivariateScore):
    """
    Sum of marginal entropies, ``H(x) + H(y)``.

    ANM uses ``x`` as the proposed cause and ``y`` as the regression residual.

    Parameters
    ----------
    method : {"auto", "spacing", "vasicek", "van es", "ebrahimi", "correa"}, default="auto"
        Entropy estimator. ``"spacing"`` uses sorted sample spacings; other values are passed to
        :func:`scipy.stats.differential_entropy`.

    window_length : int, optional
        Window length for SciPy estimators. Must be ``None`` when ``method="spacing"``.

    base : float, optional
        Logarithm base. The default uses the natural logarithm.
    """

    _tags = {"name": "entropy", "supported_algorithms": ["anm"]}

    def __init__(self, method="auto", window_length=None, base=None):
        self.method = method
        self.window_length = window_length
        self.base = base
        super().__init__()

    def __call__(self, x, y) -> float:
        x_entropy, y_entropy = _estimate_pair_entropies(
            x,
            y,
            method=self.method,
            window_length=self.window_length,
            base=self.base,
        )
        return _ensure_finite(x_entropy + y_entropy)


class EntropyDifferenceScore(BaseBivariateScore):
    """
    Difference of marginal entropies, ``H(y) - H(x)``.

    IGCI uses ``x`` as the proposed cause and ``y`` as its effect.

    Parameters
    ----------
    method : {"spacing", "auto", "vasicek", "van es", "ebrahimi", "correa"}, default="spacing"
        Entropy estimator. ``"spacing"`` uses sorted sample spacings; other values are passed to
        :func:`scipy.stats.differential_entropy`.

    window_length : int, optional
        Window length for SciPy estimators. Must be ``None`` when ``method="spacing"``.

    base : float, optional
        Logarithm base. The default uses the natural logarithm.
    """

    _tags = {"name": "entropy", "supported_algorithms": ["igci"]}

    def __init__(self, method="spacing", window_length=None, base=None):
        self.method = method
        self.window_length = window_length
        self.base = base
        super().__init__()

    def __call__(self, x, y) -> float:
        x_entropy, y_entropy = _estimate_pair_entropies(
            x,
            y,
            method=self.method,
            window_length=self.window_length,
            base=self.base,
        )
        return _ensure_finite(y_entropy - x_entropy)


class GaussScore(BaseBivariateScore):
    """
    Gaussian score, ``log Var(x) + log Var(y)``.

    This is a fast approximation to :class:`EntropyScore` for approximately Gaussian data.
    """

    _tags = {"name": "gauss", "supported_algorithms": ["anm"]}

    def __call__(self, x, y) -> float:
        score = np.log(np.var(np.asarray(x))) + np.log(np.var(np.asarray(y)))
        return _ensure_finite(score)


class SlopeScore(BaseBivariateScore):
    """
    Mean log-slope score for IGCI.

    This implements Equation 19 from the IGCI method. Repeated cause or effect values are not
    supported; use :class:`WeightedSlopeScore` when the cause contains repetitions.
    """

    _tags = {"name": "slope", "supported_algorithms": ["igci"]}

    def __call__(self, cause, effect) -> float:
        cause = np.asarray(cause)
        effect = np.asarray(effect)

        order = np.argsort(cause, kind="stable")
        cause = cause[order]
        effect = effect[order]

        if cause.size < 2:
            raise ValueError("SlopeScore requires at least two observations.")

        cause_diff = np.diff(cause)
        effect_diff = np.diff(effect)
        if np.any(cause_diff == 0):
            raise ValueError("SlopeScore does not support repeated cause values; use score='slope_weighted' instead.")
        if np.any(effect_diff == 0):
            raise ValueError("SlopeScore does not support repeated effect values.")

        score = np.mean(np.log(np.abs(effect_diff / cause_diff)))
        return _ensure_finite(score)


class WeightedSlopeScore(BaseBivariateScore):
    """
    Repetition-aware log-slope score for IGCI.

    This implements Equation 21 from the IGCI method and weights distinct cause values by their
    original multiplicities.
    """

    _tags = {"name": "slope_weighted", "supported_algorithms": ["igci"]}

    def __call__(self, cause, effect) -> float:
        cause = np.asarray(cause)
        effect = np.asarray(effect)

        order = np.argsort(cause, kind="stable")
        cause = cause[order]
        effect = effect[order]

        run_start = np.r_[0, np.flatnonzero(np.diff(cause)) + 1]
        multiplicities = np.diff(np.r_[run_start, cause.size])
        cause = cause[run_start]
        effect = effect[run_start]

        if cause.size < 2:
            raise ValueError(
                "IGCI requires sufficiently distinct cause values after removing repetitions; no valid pairs remained."
            )

        cause_diff = np.diff(cause)
        effect_diff = np.diff(effect)
        valid = effect_diff != 0
        if not valid.any():
            raise ValueError(
                "IGCI requires sufficiently distinct continuous observations; no valid slope pairs remained."
            )

        score = np.average(
            np.log(np.abs(effect_diff[valid] / cause_diff[valid])),
            weights=multiplicities[:-1][valid],
        )
        return _ensure_finite(score)


def get_bivariate_score(score: str | Callable, algorithm: str) -> Callable:
    """Return a score selected by name or supplied by the user.

    Parameters
    ----------
    score : str or callable
        Built-in score name, configured score object, or custom callable.
    algorithm : {"anm", "igci"}
        Causal discovery algorithm that will use the score.

    Returns
    -------
    callable
        Resolved score callable.

    Raises
    ------
    ValueError
        If ``score`` is an unknown name or is not callable.
    """
    algorithm = algorithm.lower()

    if isinstance(score, str):
        scores = all_objects(
            object_types=BaseBivariateScore,
            package_name="pgmpy.causal_discovery",
            return_names=False,
            filter_tags={
                "name": score.lower(),
                "supported_algorithms": algorithm,
            },
        )
        if scores:
            return scores[0]()
        raise ValueError(f"Unknown {algorithm.upper()} score: {score!r}.")

    if callable(score) and not isinstance(score, type):
        return score

    raise ValueError(f"Invalid {algorithm.upper()} score: {score!r}. Pass a built-in name or callable.")
