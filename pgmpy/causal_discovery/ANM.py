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

    Given two continuous variables, ANM orients the edge between them under the additive noise model
    ``effect = f(cause) + noise``, assuming:

    - the noise is statistically independent of the cause;
    - the relationship is nonlinear and/or the noise is non-Gaussian, so that no valid additive-noise model exists in
      the reverse direction (the linear-Gaussian case is not identifiable);
    - the two variables are genuinely causally related (ANM only orients the edge; it does not test whether one
      exists).

    It fits a regressor in both directions and orients the edge toward the one whose residuals are more independent of
    the input. Compare ``forward_score_`` and ``backward_score_`` to judge how decisive that call is.

    Parameters
    ----------
    regressor : sklearn regressor, default=None
        Regressor used to estimate ``f``. If ``None``, a :class:`~sklearn.gaussian_process.GaussianProcessRegressor`
        is used.

    ci_test : str or BaseCITest instance, default=None
        Independence test between the cause and the residuals; must expose an effect size. If ``None``, a default test
        is chosen automatically from the data type.

    random_state : int, default=None
        Random seed for the default regressor.

    Attributes
    ----------
    causal_graph_ : pgmpy.base.DAG
        The learned causal graph with the single oriented edge.

    adjacency_matrix_ : pd.DataFrame
        Adjacency matrix representation of ``causal_graph_``.

    forward_score_ : float
        Residual-dependence score for the first-column -> second-column direction. A smaller absolute value means the
        residuals are more independent of the cause.

    backward_score_ : float
        Residual-dependence score for the second-column -> first-column direction. The edge is oriented toward
        whichever direction has the smaller absolute score.

    n_features_in_ : int
        The number of features in the data used to learn the causal graph.

    feature_names_in_ : np.ndarray
        The feature names in the data used to learn the causal graph.

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
        """
        The fitting procedure for the ANM algorithm.

        Parameters
        ----------
        X : pd.DataFrame
            The data to learn the causal structure from.

        Returns
        -------
        self : pgmpy.causal_discovery.ANM
            Returns the instance with the fitted attributes.
        """
        if X.shape[1] != 2:
            raise ValueError(f"ANM requires exactly two variables, got {X.shape[1]}.")

        if get_dataset_type(X) != "continuous":
            raise ValueError("ANM requires continuous (numeric) variables; got non-continuous data.")

        x, y = self.feature_names_in_
        for col in (x, y):
            if X[col].std() == 0:
                raise ValueError(f"Variable '{col}' is constant; ANM requires non-constant variables.")

        self.forward_score_ = self._direction_score(cause=X[[x]], effect=X[y])
        self.backward_score_ = self._direction_score(cause=X[[y]], effect=X[x])

        edge = (x, y) if abs(self.forward_score_) <= abs(self.backward_score_) else (y, x)
        self.causal_graph_ = DAG([edge])
        self.adjacency_matrix_ = nx.to_pandas_adjacency(self.causal_graph_, nodelist=[x, y], weight=None, dtype="int")

        return self

    def _direction_score(self, cause: pd.DataFrame, effect: pd.Series) -> float:
        """
        Score the residual dependence for the ``cause -> effect`` direction.

        Fits the regressor to predict ``effect`` from ``cause``, then measures how dependent the resulting residuals
        are on ``cause`` using the configured CI test's effect size.

        Parameters
        ----------
        cause : pd.DataFrame
            The candidate cause variable, as a single-column frame (the regressor's 2D input).

        effect : pd.Series
            The candidate effect variable, regressed on ``cause``.

        Returns
        -------
        score : float
            The CI-test effect size between ``cause`` and the residuals. A smaller absolute value indicates more
            independent residuals, i.e. a better-fitting direction.
        """
        if self.regressor is None:
            # RBF and Gaussian-noise GP kernel (Hoyer et al., 2009); args are (init, (lower, upper)) opt bounds.
            regressor = GaussianProcessRegressor(
                kernel=C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(0.1, (1e-10, 1e1)),
                random_state=self.random_state,
            )
        else:
            regressor = clone(self.regressor)

        regressor.fit(cause, effect)
        residual = effect - regressor.predict(cause)

        cause_name = cause.columns[0]
        resid_data = cause.assign(_residual=residual)
        ci_test = get_ci_test(test=self.ci_test, data=resid_data)
        ci_test.run_test(cause_name, "_residual", Z=[])

        if ci_test.effect_size_ is None:
            raise ValueError(
                f"CI test '{type(ci_test).__name__}' does not expose an effect size, which ANM needs to score "
                "residual dependence. Use a test that sets effect_size_, such as 'pearsonr' or 'gcm'."
            )
        return ci_test.effect_size_
