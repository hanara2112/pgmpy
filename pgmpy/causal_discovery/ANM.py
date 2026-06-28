import networkx as nx
import pandas as pd
from sklearn.base import clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.gaussian_process.kernels import ConstantKernel as C

from pgmpy.base import DAG
from pgmpy.causal_discovery._base import BaseCausalDiscovery
from pgmpy.ci_tests import get_ci_test
from pgmpy.utils import get_dataset_type


class ANM(BaseCausalDiscovery):
    """
    Bivariate causal discovery using Additive Noise Models (ANM) [1]_.

    Given two continuous variables, ANM orients the edge between them by assuming
    ``effect = f(cause) + noise`` with the noise independent of the cause. It fits
    a regressor in both directions and picks the one whose residuals are most
    independent of the input. It always returns a direction; use
    ``direction_score_`` to judge how confident that call is.

    Parameters
    ----------
    regressor : sklearn regressor, default=None
        Regressor used to estimate ``f``. If ``None``, a
        :class:`~sklearn.gaussian_process.GaussianProcessRegressor` is used.

    ci_test : str or BaseCITest instance, default=None
        Independence test between the cause and the residuals. If ``None``, uses
        ``"gcm"``, which only detects *linear* residual dependence.

    random_state : int, default=None
        Random seed for the default regressor.

    Attributes
    ----------
    causal_graph_ : pgmpy.base.DAG
        The learned causal graph with the single oriented edge.

    adjacency_matrix_ : pd.DataFrame
        Adjacency matrix representation of ``causal_graph_``.

    direction_score_ : float
        Orientation confidence in ``[-1, 1]``: the residual-dependence effect size of
        the reverse direction minus that of the forward direction. ``> 0`` means the
        first column causes the second, ``< 0`` the reverse, and values near ``0``
        mean low confidence.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pgmpy.causal_discovery import ANM
    >>> rng = np.random.default_rng(42)
    >>> x = rng.uniform(-2, 2, 500)
    >>> df = pd.DataFrame({"X": x, "Y": x**3 + rng.normal(0, 0.5, 500)})
    >>> anm = ANM(random_state=42).fit(df)
    >>> anm.causal_graph_.edges()
    OutEdgeView([('X', 'Y')])

    References
    ----------
    .. [1] Hoyer, P. O., Janzing, D., Mooij, J. M., Peters, J., & Schölkopf, B.
           Nonlinear causal discovery with additive noise models. In NeurIPS, 2009.

    """

    def __init__(self, regressor=None, ci_test=None, random_state=None):
        self.regressor = regressor
        self.ci_test = ci_test
        self.random_state = random_state

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.categorical = False
        return tags

    def _fit(self, X: pd.DataFrame):
        """Orient the edge between the two variables in ``X`` and set the fitted attributes."""
        if X.shape[1] != 2:
            raise ValueError(f"ANM requires exactly two variables, got {X.shape[1]}.")

        if get_dataset_type(X) != "continuous":
            raise ValueError("ANM requires continuous (numeric) variables; got non-continuous data.")

        x, y = X.columns
        for col in (x, y):
            if X[col].std() == 0:
                raise ValueError(f"Variable '{col}' is constant; ANM requires non-constant variables.")

        score_forward = self._direction_score(X[x], X[y])
        score_backward = self._direction_score(X[y], X[x])
        self.direction_score_ = score_backward - score_forward

        edge = (x, y) if self.direction_score_ >= 0 else (y, x)
        self.causal_graph_ = DAG([edge])
        self.adjacency_matrix_ = nx.to_pandas_adjacency(self.causal_graph_, nodelist=[x, y], weight=None, dtype="int")

        return self

    def _direction_score(self, cause: pd.Series, effect: pd.Series) -> float:
        """Regress ``effect`` on ``cause`` and return the cause-residual dependence (lower is more independent)."""
        if self.regressor is None:
            # RBF and Gaussian-noise GP kernel (Hoyer et al., 2009); args are (init, (lower, upper)) opt bounds.
            regressor = GaussianProcessRegressor(
                kernel=C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(0.1, (1e-10, 1e1)),
                random_state=self.random_state,
            )
        else:
            regressor = clone(self.regressor)

        cause_2d = cause.to_numpy().reshape(-1, 1)
        regressor.fit(cause_2d, effect.to_numpy())
        residual = effect.to_numpy() - regressor.predict(cause_2d)

        resid_data = pd.DataFrame({cause.name: cause.to_numpy(), "_residual": residual})
        ci_test = get_ci_test(test="gcm" if self.ci_test is None else self.ci_test, data=resid_data)
        ci_test.run_test(cause.name, "_residual", Z=[])

        if ci_test.effect_size_ is None:
            raise ValueError(
                f"CI test '{type(ci_test).__name__}' does not expose an effect size, which ANM needs to score "
                "residual dependence. Use a test that sets effect_size_, such as the default 'gcm'."
            )
        return ci_test.effect_size_
