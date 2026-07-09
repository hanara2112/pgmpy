import networkx as nx
import numpy as np
import pandas as pd
from scipy.special import psi
from scipy.stats import differential_entropy

from pgmpy.base import DAG
from pgmpy.causal_discovery._base import BaseCausalDiscovery
from pgmpy.utils import get_dataset_type

_ENTROPY_METHODS = ("spacing", "vasicek", "ebrahimi", "van es", "correa", "auto")


class IGCI(BaseCausalDiscovery):
    """
    Bivariate causal discovery using Information-Geometric Causal Inference (IGCI)
    :cite:p:`mooij_2016,janzing_2012`.

    Given two continuous variables, IGCI orients the edge between them under the
    deterministic, invertible model ``Y = f(X)``, assuming:

    - No unobserved confounders.
    - ``Y = f(X)`` is monotonic and close to deterministic (little noise).
    - The cause distribution and ``f`` are independent.

    IGCI scores each direction and orients toward the lower score. Ties use the
    first column as cause.

    Parameters
    ----------
    scoring : str, default="slope"
        Direction score to use. One of:

        - ``"slope"``: weighted mean of log-slopes along cause-ordered points.
        - ``"entropy"``: estimated marginal entropy of the effect minus that of the cause.

    ref_measure : str, default="uniform"
        Affine preprocessing for each variable before scoring. One of:

        - ``"uniform"``: min-max scaling to ``[0, 1]``.
        - ``"gaussian"``: zero mean and unit variance.

    entropy_method : str, default="auto"
        Entropy estimator when ``scoring="entropy"``. ``"auto"`` and ``"spacing"`` use
        the IGCI sorted-spacing estimator. Other values are passed to
        ``scipy.stats.differential_entropy``.

    Attributes
    ----------
    causal_graph_ : pgmpy.base.DAG
        Learned graph with the single oriented edge.

    adjacency_matrix_ : pd.DataFrame
        Adjacency matrix of causal_graph_.

    forward_score_ : float
        Score when the first column is treated as cause. Lower is better.

    backward_score_ : float
        Score when the second column is treated as cause. Lower is better.

    n_features_in_ : int
        The number of features in the data used to learn the causal graph.

    feature_names_in_ : np.ndarray
        The feature names in the data used to learn the causal graph.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pgmpy.causal_discovery import IGCI
    >>> rng = np.random.default_rng(42)
    >>> x = rng.uniform(0, 1, 500)
    >>> df = pd.DataFrame({"X": x, "Y": x**3 + rng.normal(0, 1e-3, 500)})
    >>> igci = IGCI().fit(df)
    >>> list(igci.causal_graph_.edges())
    [('X', 'Y')]
    >>> igci.forward_score_.round(5)
    np.float64(0.22245)
    >>> igci.backward_score_.round(5)
    np.float64(1.66886)
    >>> IGCI(scoring="entropy").fit(df).forward_score_.round(5)
    np.float64(-0.70337)

    References
    ----------
    - :cite:p:`mooij_2016`
    - :cite:p:`janzing_2012`

    """

    def __init__(self, scoring="slope", ref_measure="uniform", entropy_method="auto"):
        self.scoring = scoring
        self.ref_measure = ref_measure
        self.entropy_method = entropy_method

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.categorical = False
        return tags

    def _marginal_entropy(self, x: np.ndarray) -> float:
        """
        Estimate differential entropy of one normalized variable.

        Parameters
        ----------
        x : np.ndarray
            One-dimensional normalized sample.

        Returns
        -------
        float
            Estimated differential entropy.
        """
        xs = np.sort(x)
        if xs.size < 2:
            raise ValueError("IGCI entropy estimation requires at least two observations.")
        deltas = np.diff(xs)
        spacing_sum = np.sum(np.log(deltas[deltas > 0]))
        return psi(xs.size) - psi(1.0) + spacing_sum / (xs.size - 1)

    def _slope_score(self, cause: np.ndarray, effect: np.ndarray) -> float:
        """
        Score the ``cause -> effect`` direction via log-slopes.

        Parameters
        ----------
        cause : np.ndarray
            Normalized candidate cause sample.
        effect : np.ndarray
            Normalized candidate effect sample, aligned with ``cause``.

        Returns
        -------
        float
            Direction score; compared against the reverse direction in ``fit``.
        """
        order = np.argsort(cause, kind="stable")
        cause = cause[order]
        effect = effect[order]

        run_start = np.concatenate(([0], np.flatnonzero(cause[1:] != cause[:-1]) + 1))
        multiplicities = np.diff(np.concatenate((run_start, [cause.size])))
        unique_cause = cause[run_start]
        unique_effect = effect[run_start]

        if unique_cause.size < 2:
            raise ValueError(
                "IGCI requires sufficiently distinct cause values after removing repetitions; no valid pairs remained."
            )

        dc = np.diff(unique_cause)
        de = np.diff(unique_effect)
        weights = multiplicities[:-1]
        valid = (dc != 0) & (de != 0)
        if not valid.any():
            raise ValueError(
                "IGCI requires sufficiently distinct continuous observations; no valid slope pairs remained."
            )

        weights = weights[valid]
        return np.sum(weights * np.log(np.abs(de[valid] / dc[valid]))) / weights.sum()

    def _direction_score(self, cause: np.ndarray, effect: np.ndarray) -> float:
        """Return the IGCI score for the ``cause -> effect`` ordering."""
        if self.scoring == "slope":
            return self._slope_score(cause, effect)

        if self.entropy_method in ("spacing", "auto"):
            h_effect = self._marginal_entropy(effect)
            h_cause = self._marginal_entropy(cause)
        else:
            h_effect = differential_entropy(effect, method=self.entropy_method)
            h_cause = differential_entropy(cause, method=self.entropy_method)
        if not np.isfinite(h_effect) or not np.isfinite(h_cause):
            raise ValueError(f"Entropy estimation with method {self.entropy_method!r} failed for the given sample.")
        return h_effect - h_cause

    def _fit(self, X: pd.DataFrame):
        """
        Orient the edge between the two variables in ``X`` using IGCI.

        Parameters
        ----------
        X : pd.DataFrame
            The data to learn the causal structure from. Must contain exactly
            two continuous variables.

        Returns
        -------
        self : IGCI
            Returns the instance with the fitted attributes set.
        """
        # Step 1: Validate hyperparameters.
        if self.scoring not in ("slope", "entropy"):
            raise ValueError(f"scoring must be one of ('slope', 'entropy'). Got: {self.scoring!r}")
        if self.ref_measure not in ("uniform", "gaussian"):
            raise ValueError(f"ref_measure must be one of ('uniform', 'gaussian'). Got: {self.ref_measure!r}")
        if self.scoring == "entropy" and self.entropy_method not in _ENTROPY_METHODS:
            raise ValueError(f"entropy_method must be one of {_ENTROPY_METHODS}. Got: {self.entropy_method!r}")

        # Step 2: Validate the input data.
        if X.shape[1] != 2:
            raise ValueError(f"IGCI requires exactly two variables, got {X.shape[1]}.")
        if get_dataset_type(X) != "continuous":
            raise ValueError("IGCI requires continuous (numeric) variables; got non-continuous data.")

        x, y = self.feature_names_in_
        for col in (x, y):
            if X[col].std() == 0:
                raise ValueError(f"Variable '{col}' is constant; IGCI requires non-constant variables.")

        # Step 3: Affine normalization to the reference measure.
        x_vals = X[x].to_numpy(dtype=float)
        y_vals = X[y].to_numpy(dtype=float)
        if self.ref_measure == "uniform":
            x_norm = (x_vals - x_vals.min()) / (x_vals.max() - x_vals.min())
            y_norm = (y_vals - y_vals.min()) / (y_vals.max() - y_vals.min())
        else:
            x_norm = (x_vals - x_vals.mean()) / x_vals.std()
            y_norm = (y_vals - y_vals.mean()) / y_vals.std()

        # Step 4: Score both directions.
        self.forward_score_ = self._direction_score(cause=x_norm, effect=y_norm)
        self.backward_score_ = self._direction_score(cause=y_norm, effect=x_norm)

        # Step 5: Orient the edge and store the fitted attributes.
        edge = (x, y) if self.forward_score_ <= self.backward_score_ else (y, x)
        self.causal_graph_ = DAG([edge])
        self.adjacency_matrix_ = nx.to_pandas_adjacency(self.causal_graph_, nodelist=[x, y], weight=None, dtype="int")

        return self
