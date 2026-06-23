import pandas as pd

from pgmpy.causal_discovery._base import BaseCausalDiscovery


class ANM(BaseCausalDiscovery):
    """Bivariate causal discovery with additive noise models.

    The additive noise model assumes that the effect can be written as
    ``Y = f(X) + N`` where ``N`` is independent of ``X``. The algorithm fits
    both directions and prefers the one whose regression residuals are
    independent of the input variable.

    Parameters
    ----------
    regressor : sklearn regressor, default=None
        Regressor used to estimate ``f``. When ``None``, a
        :class:`~sklearn.gaussian_process.GaussianProcessRegressor` is used.

    ci_test : str or BaseCITest, default=None
        Independence test applied to the input and the regression residuals.
        When ``None``, the default continuous-data test from
        :func:`~pgmpy.ci_tests.get_ci_test` is used.

    random_state : int, default=None
        Random seed for the default regressor.

    Attributes
    ----------
    causal_graph_ : pgmpy.base.DAG
        The learned causal graph.

    adjacency_matrix_ : pd.DataFrame
        Adjacency matrix representation of ``causal_graph_``.

    direction_score_ : float
        ``S(Y -> X) - S(X -> Y)`` where ``S`` is the direction score. Positive
        values mean the forward column order wins.

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
        raise NotImplementedError
