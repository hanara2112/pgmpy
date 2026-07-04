import networkx as nx
import numpy as np
import pandas as pd
from scipy.special import digamma

from pgmpy.base import DAG
from pgmpy.causal_discovery._base import BaseCausalDiscovery
from pgmpy.utils import get_dataset_type


class IGCI(BaseCausalDiscovery):
    """
    Bivariate causal discovery using Information-Geometric Causal Inference (IGCI) :cite:p:`mooij_2016`.

    Given two continuous variables, IGCI orients the edge between them under the
    deterministic, invertible model ``Y = f(X)``, assuming:

    - Causal sufficiency (no unobserved confounders);
    - The mechanism ``f`` is deterministic and invertible (monotonic), which holds
      only approximately under low observation noise;
    - The cause distribution and the mechanism ``f`` are chosen independently, so
      the density of the cause carries no information about ``f``.

    For each ordering it estimates the IGCI objective implied by this
    independent-mechanism assumption and orients the edge toward the smaller
    score. A direction is always returned; on a tie it falls back to the first
    column ordering.

    Parameters
    ----------
    scoring : str, default="slope"
        Estimator used for the per-direction score. Options are:

        - ``"slope"``: mean log absolute finite-difference slope between consecutive
          observations, a finite-difference approximation to the mean of ``log |f'|``.
        - ``"entropy"``: difference of the empirical differential entropies.

    ref_measure : str, default="uniform"
        Reference measure the data is normalized to before scoring. Options are:

        - ``"uniform"``: rescale each variable to ``[0, 1]``.
        - ``"gaussian"``: standardize each variable to zero mean and unit variance.

    Attributes
    ----------
    causal_graph_ : pgmpy.base.DAG
        The learned causal graph with the single oriented edge.

    adjacency_matrix_ : pd.DataFrame
        Adjacency matrix representation of ``causal_graph_``.

    forward_score_ : float
        IGCI score for the first-column -> second-column direction. A smaller
        score means that direction better matches the IGCI assumption.

    backward_score_ : float
        IGCI score for the second-column -> first-column direction. The edge is
        oriented toward whichever direction has the smaller score.

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
    >>> igci.causal_graph_.edges()
    OutEdgeView([('X', 'Y')])
    >>> igci.forward_score_.round(5)
    np.float64(0.22245)
    >>> igci.backward_score_.round(5)
    np.float64(1.66886)

    References
    ----------
    - :cite:p:`mooij_2016`

    """

    def __init__(self, scoring="slope", ref_measure="uniform"):
        self.scoring = scoring
        self.ref_measure = ref_measure

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.categorical = False
        return tags

    def _fit(self, X: pd.DataFrame):
        """
        The fitting procedure for the IGCI algorithm.

        Parameters
        ----------
        X : pd.DataFrame
            The data to learn the causal structure from.

        Returns
        -------
        self : pgmpy.causal_discovery.IGCI
            Returns the instance with the fitted attributes.
        """
        # Step 0: Validate the inputs.
        if X.shape[1] != 2:
            raise ValueError(f"IGCI requires exactly two variables, got {X.shape[1]}.")

        if get_dataset_type(X) != "continuous":
            raise ValueError("IGCI requires continuous (numeric) variables; got non-continuous data.")

        for param, allowed in (("scoring", ("slope", "entropy")), ("ref_measure", ("uniform", "gaussian"))):
            value = getattr(self, param)
            if value not in allowed:
                raise ValueError(f"{param} must be one of {allowed}. Got: {value!r}")

        x, y = self.feature_names_in_
        for col in (x, y):
            if X[col].std() == 0:
                raise ValueError(f"Variable '{col}' is constant; IGCI requires non-constant variables.")

        # Step 1: Normalize each variable and score both directions.
        x_norm = self._normalize(X[x].to_numpy(dtype=float))
        y_norm = self._normalize(X[y].to_numpy(dtype=float))
        self.forward_score_ = self._direction_score(cause=x_norm, effect=y_norm)
        self.backward_score_ = self._direction_score(cause=y_norm, effect=x_norm)

        # Step 2: Orient the edge toward the direction with the smaller score.
        edge = (x, y) if self.forward_score_ <= self.backward_score_ else (y, x)
        self.causal_graph_ = DAG([edge])
        self.adjacency_matrix_ = nx.to_pandas_adjacency(self.causal_graph_, nodelist=[x, y], weight=None, dtype="int")

        return self

    def _normalize(self, values: np.ndarray) -> np.ndarray:
        """Normalize a variable according to the configured reference measure."""
        if self.ref_measure == "uniform":
            return (values - values.min()) / (values.max() - values.min())
        return (values - values.mean()) / values.std()

    def _direction_score(self, cause: np.ndarray, effect: np.ndarray) -> float:
        """
        Estimate the IGCI objective for the ``cause -> effect`` ordering; a smaller score favors that direction.

        The estimator is selected by ``scoring`` (see the class docstring). The ``"slope"`` variant sorts by ``cause``
        and ignores pairs whose finite differences vanish.

        Parameters
        ----------
        cause : np.ndarray
            The normalized candidate cause variable.

        effect : np.ndarray
            The normalized candidate effect variable.

        Returns
        -------
        score : float
            The IGCI score for the ``cause -> effect`` direction. A smaller value indicates a better-fitting direction.
        """
        if self.scoring == "entropy":
            return self._entropy(effect) - self._entropy(cause)

        order = np.argsort(cause, kind="stable")
        cause_sorted, effect_sorted = cause[order], effect[order]
        dc = np.diff(cause_sorted)
        de = np.diff(effect_sorted)
        mask = (dc != 0) & (de != 0)
        if not mask.any():
            raise ValueError("IGCI requires sufficiently distinct continuous observations; no valid pairs remained.")
        return np.mean(np.log(np.abs(de[mask] / dc[mask])))

    @staticmethod
    def _entropy(values: np.ndarray) -> float:
        """
        Estimate the differential entropy of a variable.

        Uses the 1-spacing (nearest-neighbor) estimator on the sorted values. Duplicate values are removed first, as
        repeated observations would otherwise send the estimate to negative infinity.

        Parameters
        ----------
        values : np.ndarray
            The (normalized) variable to estimate the entropy of.

        Returns
        -------
        entropy : float
            The estimated differential entropy.
        """
        unique_values = np.unique(values)
        n = len(unique_values)
        return np.sum(np.log(np.diff(unique_values))) / (n - 1) + digamma(n) - digamma(1)
