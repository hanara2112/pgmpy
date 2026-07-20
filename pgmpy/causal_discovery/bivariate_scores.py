import numpy as np
import pandas as pd
from scipy.special import psi
from scipy.stats import differential_entropy
from skbase.base import BaseObject
from skbase.lookup import all_objects

from pgmpy.ci_tests import BaseCITest, get_ci_test


def _ensure_finite(value, name="Score") -> float:
    """Return ``value`` as a float and raise if it is not finite."""
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} is not finite for the given sample.")
    return value


def _scipy_entropy(values, method="auto", window_length=None, base=None) -> float:
    """Estimate differential entropy using SciPy."""
    entropy_kwargs = {"method": method, "base": base}
    if window_length is not None:
        entropy_kwargs["window_length"] = window_length

    entropy = differential_entropy(np.asarray(values), **entropy_kwargs)
    return _ensure_finite(entropy, name="Entropy estimate")


def _spacing_entropy(values, base=None) -> float:
    """Estimate differential entropy using consecutive sorted spacings."""
    if base is not None and (base <= 0 or base == 1):
        raise ValueError("base must be positive and not equal to 1.")

    values = np.sort(np.asarray(values))
    if values.size < 2:
        raise ValueError("Entropy estimation requires at least two observations.")

    deltas = np.diff(values)
    if np.any(deltas == 0):
        raise ValueError("Spacing entropy requires distinct observations.")

    entropy = psi(values.size) - psi(1) + np.log(deltas).sum() / (values.size - 1)
    if base is not None:
        entropy /= np.log(base)
    return _ensure_finite(entropy, name="Entropy estimate")


class BaseBivariateScore(BaseObject):
    """Base class for scores that compare two one-dimensional samples.

    Subclasses are called as ``score(x, y)`` and return a float. A smaller score indicates the
    preferred direction. Each subclass defines ``name`` and ``supported_algorithms`` tags for
    built-in lookup.
    """

    _tags = {"name": None, "supported_algorithms": []}

    def __call__(self, x, y) -> float:
        raise NotImplementedError


class IndependenceScore(BaseBivariateScore):
    """
    Dependence score from a conditional-independence test.

    Runs an unconditional CI test between ``x`` and ``y``. A smaller score means weaker dependence.

    Parameters
    ----------
    ci_test : str or pgmpy.ci_tests.BaseCITest, default="pearsonr"
        The independence test, resolved via :func:`pgmpy.ci_tests.get_ci_test`.
        The test must provide the output selected by ``criterion``.

    criterion : {"effect_size", "statistic", "p_value"}, default="effect_size"
        Which CI-test output to use. Each option is transformed so that a smaller value means
        weaker dependence:

        - ``"effect_size"`` returns the test's non-negative dependence magnitude.
        - ``"statistic"`` returns the absolute test statistic.
        - ``"p_value"`` returns the negative p-value.
    """

    _tags = {"name": "independence", "supported_algorithms": ["anm"]}

    def __init__(self, ci_test="pearsonr", criterion="effect_size"):
        self.ci_test = ci_test
        self.criterion = criterion
        super().__init__()

    def __call__(self, x, y) -> float:
        data = pd.DataFrame({"_x": np.asarray(x), "_y": np.asarray(y)})
        if isinstance(self.ci_test, BaseCITest):
            test = self.ci_test.clone().set_params(data=data)
        else:
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
    Differential-entropy score, ``H(x) + H(y)`` :cite:p:`mooij_2016`.

    A smaller value means a better-fitting direction. The parameters are forwarded to
    :func:`scipy.stats.differential_entropy`.

    Parameters
    ----------
    method : {"auto", "vasicek", "van es", "ebrahimi", "correa"}, default="auto"
        Differential-entropy estimator.

    window_length : int, optional
        Window length for the spacing-based estimators. The default is chosen by SciPy.

    base : float, optional
        Logarithm base for the entropy. The default uses the natural logarithm.
    """

    _tags = {"name": "entropy", "supported_algorithms": ["anm"]}

    def __init__(
        self,
        method="auto",
        window_length=None,
        base=None,
    ):
        self.method = method
        self.window_length = window_length
        self.base = base
        super().__init__()

    def __call__(self, x, y) -> float:
        entropy_kwargs = {
            "method": self.method,
            "window_length": self.window_length,
            "base": self.base,
        }
        return _scipy_entropy(x, **entropy_kwargs) + _scipy_entropy(y, **entropy_kwargs)


class EntropyDifferenceScore(BaseBivariateScore):
    """
    Difference of marginal entropies, ``H(y) - H(x)``.

    Parameters
    ----------
    method : {"spacing", "auto", "vasicek", "van es", "ebrahimi", "correa"}, default="spacing"
        Entropy estimator. ``"spacing"`` uses consecutive sorted spacings; other values are passed
        to :func:`scipy.stats.differential_entropy`.

    window_length : int, optional
        Window length for SciPy estimators. Must be ``None`` when ``method="spacing"``.

    base : float, optional
        Logarithm base for the entropy. The default uses the natural logarithm.
    """

    _tags = {"name": "entropy", "supported_algorithms": ["igci"]}

    def __init__(
        self,
        method="spacing",
        window_length=None,
        base=None,
    ):
        self.method = method
        self.window_length = window_length
        self.base = base
        super().__init__()

    def __call__(self, x, y) -> float:
        if self.method == "spacing":
            if self.window_length is not None:
                raise ValueError("window_length is not supported by the spacing estimator.")
            return _spacing_entropy(y, base=self.base) - _spacing_entropy(x, base=self.base)

        entropy_kwargs = {
            "method": self.method,
            "window_length": self.window_length,
            "base": self.base,
        }
        return _scipy_entropy(y, **entropy_kwargs) - _scipy_entropy(x, **entropy_kwargs)


class GaussScore(BaseBivariateScore):
    """
    Gaussian (log-variance) score, ``log Var(x) + log Var(y)`` :cite:p:`mooij_2016`.

    The Gaussian special case of :class:`EntropyScore`. A smaller value means a better-fitting
    direction. This score is unreliable when identifiability depends on non-Gaussian noise; prefer
    :class:`EntropyScore` or :class:`IndependenceScore` in that case.
    """

    _tags = {"name": "gauss", "supported_algorithms": ["anm"]}

    def __call__(self, x, y) -> float:
        score = np.log(np.var(np.asarray(x))) + np.log(np.var(np.asarray(y)))
        return _ensure_finite(score)


class SlopeScore(BaseBivariateScore):
    """
    Repetition-aware log-slope score for IGCI :cite:p:`mooij_2016`.

    Computes a weighted average of the log slopes between neighboring observations after sorting
    by ``x``. Repeated ``x`` values are weighted by their frequency, and zero ``y`` spacings are ignored.
    """

    _tags = {"name": "slope", "supported_algorithms": ["igci"]}

    def __call__(self, x, y) -> float:
        x = np.asarray(x)
        y = np.asarray(y)

        order = np.argsort(x, kind="stable")
        x = x[order]
        y = y[order]

        run_start = np.r_[0, np.flatnonzero(np.diff(x)) + 1]
        multiplicities = np.diff(np.r_[run_start, x.size])
        x = x[run_start]
        y = y[run_start]

        if x.size < 2:
            raise ValueError(
                "SlopeScore requires sufficiently distinct x values after removing repetitions; "
                "no valid pairs remained."
            )

        x_diff = np.diff(x)
        y_diff = np.diff(y)
        valid = y_diff != 0
        if not valid.any():
            raise ValueError("SlopeScore requires at least one non-zero y spacing.")

        score = np.average(
            np.log(np.abs(y_diff[valid] / x_diff[valid])),
            weights=multiplicities[:-1][valid],
        )
        return _ensure_finite(score)


def get_bivariate_score(score, algorithm):
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
