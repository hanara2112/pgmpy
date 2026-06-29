import networkx as nx
import numpy as np
import pandas as pd
from scipy.special import digamma

from pgmpy.base import DAG
from pgmpy.causal_discovery._base import BaseCausalDiscovery
from pgmpy.utils import get_dataset_type


class IGCI(BaseCausalDiscovery):
    """
    Bivariate causal discovery using Information-Geometric Causal Inference (IGCI).

    Works for near-deterministic relationships Y = f(X) with f monotonic. Each
    direction gets a score (lower = more likely the cause), and the smaller one
    wins. Based on Janzing et al. (2012).

    Parameters
    ----------
    scoring : {"slope", "entropy"}, default="slope"
        Which estimator to use for the direction score.

    ref_measure : {"uniform", "gaussian"}, default="uniform"
        How to normalize the data before scoring. "uniform" rescales to [0, 1],
        "gaussian" standardizes to zero mean and unit variance.

    random_state : int, default=None
        Not used (IGCI is deterministic). Kept for a consistent interface.

    Attributes
    ----------
    causal_graph_ : pgmpy.base.DAG
        The learned graph with the single oriented edge.

    adjacency_matrix_ : pd.DataFrame
        Adjacency matrix of causal_graph_.

    direction_score_ : float
        S(Y -> X) - S(X -> Y). Positive means the first column causes the second.

    References
    ----------
    Janzing et al. Information-geometric approach to inferring causal directions.
    Artificial Intelligence, 182-183:1-31, 2012.
    """

    def __init__(self, scoring="slope", ref_measure="uniform", random_state=None):
        self.scoring = scoring
        self.ref_measure = ref_measure
        self.random_state = random_state

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.categorical = False
        return tags

    def _fit(self, X: pd.DataFrame):
        if X.shape[1] != 2:
            raise ValueError(f"IGCI requires exactly two variables, got {X.shape[1]}.")

        if get_dataset_type(X) != "continuous":
            raise ValueError("IGCI requires continuous (numeric) variables; got non-continuous data.")

        if self.scoring not in ("slope", "entropy"):
            raise ValueError(f"scoring must be one of ('slope', 'entropy'). Got: {self.scoring!r}")

        if self.ref_measure not in ("uniform", "gaussian"):
            raise ValueError(f"ref_measure must be one of ('uniform', 'gaussian'). Got: {self.ref_measure!r}")

        x, y = X.columns
        for col in (x, y):
            if X[col].std() == 0:
                raise ValueError(f"Variable '{col}' is constant; IGCI requires non-constant variables.")

        x_norm = self._normalize(X[x].to_numpy(dtype=float))
        y_norm = self._normalize(X[y].to_numpy(dtype=float))

        score_forward = self._direction_score(x_norm, y_norm)
        score_backward = self._direction_score(y_norm, x_norm)
        self.direction_score_ = score_backward - score_forward

        edge = (x, y) if self.direction_score_ >= 0 else (y, x)
        self.causal_graph_ = DAG([edge])
        self.adjacency_matrix_ = nx.to_pandas_adjacency(self.causal_graph_, nodelist=[x, y], weight=None, dtype="int")

        return self

    def _normalize(self, values):
        if self.ref_measure == "uniform":
            return (values - values.min()) / (values.max() - values.min())
        return (values - values.mean()) / values.std()

    def _direction_score(self, cause, effect):
        # lower score => this direction is more likely
        if self.scoring == "entropy":
            return self._entropy(effect) - self._entropy(cause)

        # slope: average log-slope between points sorted by the cause.
        # skip ties so we don't take log(0) or divide by 0.
        order = np.argsort(cause, kind="stable")
        cause_sorted, effect_sorted = cause[order], effect[order]
        d_cause = np.diff(cause_sorted)
        d_effect = np.diff(effect_sorted)
        mask = (d_cause != 0) & (d_effect != 0)
        return float(np.sum(np.log(np.abs(d_effect[mask] / d_cause[mask]))) / (len(cause) - 1))

    @staticmethod
    def _entropy(values):
        # 1-NN spacing entropy estimator (gaps of the sorted values)
        n = len(values)
        gaps = np.diff(np.sort(values))
        gaps = gaps[gaps != 0]
        return float(np.sum(np.log(gaps)) / (n - 1) + digamma(n) - digamma(1))
