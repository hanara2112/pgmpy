import numpy as np
import pandas as pd
from scipy import stats
from skbase.utils.dependencies import _safe_import
from sklearn.base import clone
from sklearn.cross_decomposition import CCA

from ._base import _BaseCITest


class PillaiTrace(_BaseCITest):
    r"""
    Pillai's trace test for conditional independence with mixed data [1].

    This test first residualizes :math:`X` and :math:`Y` with respect to :math:`[1, Z]` using an estimator (XGBoost by
    default, but any sklearn-compatible estimator can be provided). For a continuous target :math:`T`, the residual is

    .. math::
        r_T = T - \hat{T}(Z).

    For a categorical target :math:`T` with :math:`K` categories, let :math:`D_T \in \{0, 1\}^{n \times K}` be the
    dummy-encoded matrix of :math:`T`, and let :math:`\hat{D}_T(Z)` denote the predicted class probabilities from the
    classifier. The residual matrix [2] is defined as (last column dropped to avoid colinearity):

    .. math::
        R_T = \operatorname{drop\_last}\left(D_T - \hat{D}_T(Z)\right),

    Let :math:`R_X \in \mathbb{R}^{n \times p}` and :math:`R_Y \in \mathbb{R}^{n \times q}` be the residuals of
    :math:`X` and :math:`Y`, and :math:`\rho = \rho_1, \ldots, \rho_s` be the canonical correlations between them. The
    Pillai's trace statistic is:

    .. math::
        V = \sum_{i=1}^{s} \rho_i^2.

    The p-value is computed using :math:`F`-approximation as (:math:`p` and :math:`q` are the number of columns in
    residual matrices):

    .. math::
        F = \frac{V / (pq)}{(s - V) / \left[s (n - 1 + s - p - q)\right]}
          = \frac{V}{pq} \cdot \frac{s (n - 1 + s - p - q)}{s - V},

    with numerator degrees of freedom :math:`df_1 = pq` and denominator degrees of freedom
    :math:`df_2 = s (n - 1 + s - p - q)`, where :math:`n` is the sample size.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset in which to test the independence condition.

    estimator : estimator instance, optional
        Any sklearn-compatible estimator with ``fit``, ``predict``, and ``predict_proba``
        (if testing discrete variables) methods. If ``None`` (default), uses XGBoost
        (``XGBClassifier`` for categorical targets, ``XGBRegressor`` for continuous).
        Categorical Z columns are one-hot encoded when a custom estimator is used, since
        most sklearn estimators do not handle them natively.

    seed : int, optional
        Random seed for the default XGBoost estimator. Ignored when a custom
        ``estimator`` is provided.

    Attributes
    ----------
    statistic_ : float
        Pillai's trace statistic :math:`V`. Set after calling the test.
    p_value_ : float
        The p-value for the test, computed via F-approximation. Set after calling the test.

    References
    ----------
    .. [1] Ankan, Ankur, and Johannes Textor. "A simple unified approach to testing high-dimensional conditional
           independences for categorical and ordinal data." Proceedings of the AAAI Conference on Artificial
           Intelligence.
    .. [2] Li, C.; and Shepherd, B. E. 2010. Test of Association Between Two Ordinal Variables While Adjusting for
           Covariates. Journal of the American Statistical Association.
    .. [3] Muller, K. E. and Peterson B. L. (1984) Practical Methods for computing power in testing the multivariate
           general linear hypothesis. Computational Statistics & Data Analysis.
    """

    _tags = {
        "name": "pillai",
        "data_types": ("discrete", "continuous", "mixed"),
        "default_for": "mixed",
        "requires_data": True,
    }

    def __init__(self, data: pd.DataFrame, estimator=None, seed=None):
        self.data = data
        self.estimator = estimator
        self.seed = seed
        super().__init__()

    def _fit_predict(self, target_col, Z_data, xgboost=None):
        """Fit an estimator on Z_data to predict target_col, return predictions and category index."""
        is_cat = self.data.loc[:, target_col].dtype == "category"
        target_data = self.data.loc[:, target_col]
        cat_index = None

        if self.estimator is not None:
            model = clone(self.estimator)
        else:
            enable_categorical = any(Z_data.dtypes == "category")
            model_cls = xgboost.XGBClassifier if is_cat else xgboost.XGBRegressor
            model = model_cls(
                enable_categorical=enable_categorical,
                seed=self.seed,
                random_state=self.seed,
            )

        if is_cat:
            if self.estimator is not None and not hasattr(model, "predict_proba"):
                raise ValueError(
                    f"The provided estimator ({type(model).__name__}) must have a "
                    f"`predict_proba` method for discrete variable '{target_col}'."
                )
            y_encoded, cat_index = pd.factorize(target_data)
            model.fit(Z_data, y_encoded)
            pred = model.predict_proba(Z_data)
        else:
            model.fit(Z_data, target_data)
            pred = model.predict(Z_data)

        return pred, cat_index

    def run_test(
        self,
        X: str,
        Y: str,
        Z: list,
    ):
        """
        Compute Pillai's trace statistic and p-value.

        Sets ``self.statistic_`` (Pillai's trace) and ``self.p_value_``.

        Parameters
        ----------
        X : str
            The first variable for testing X _|_ Y | Z.
        Y : str
            The second variable for testing X _|_ Y | Z.
        Z : list
            Conditioning variables.

        Returns
        -------
        statistic : float
            The Pillai's trace statistic.
        p_value : float
            The p-value.
        """
        data = self.data

        # Step 1: Add an intercept column so the estimator always has at least one feature.
        Z = Z + ["_intercept_Z"]
        data = data.assign(_intercept_Z=np.ones(data.shape[0]))

        # Step 2: Prepare Z data. Custom sklearn estimators need one-hot encoded Z;
        # XGBoost handles categorical columns natively.
        xgboost = None
        if self.estimator is not None:
            Z_data = pd.get_dummies(data.loc[:, Z])
        else:
            xgboost = _safe_import("xgboost")
            Z_data = data.loc[:, Z]

        # Step 3: Get predictions for X and Y given Z.
        pred_x, x_cat_index = self._fit_predict(X, Z_data, xgboost)
        pred_y, y_cat_index = self._fit_predict(Y, Z_data, xgboost)

        # Step 4: Compute the residuals.
        def get_residuals(col_name, pred, cat_index):
            if data.loc[:, col_name].dtype == "category":
                dummies = pd.get_dummies(data.loc[:, col_name]).loc[:, cat_index.categories[cat_index.codes]]
                return (dummies - pred).iloc[:, :-1]
            else:
                return data.loc[:, col_name] - pred

        res_x = get_residuals(X, pred_x, x_cat_index)
        res_y = get_residuals(Y, pred_y, y_cat_index)

        if isinstance(res_x, pd.Series):
            res_x = res_x.to_frame()
        if isinstance(res_y, pd.Series):
            res_y = res_y.to_frame()

        # Step 5: Compute Pillai's trace via CCA.
        n_components = min(res_x.shape[1], res_y.shape[1])
        cca = CCA(scale=False, n_components=n_components)
        res_x_c, res_y_c = cca.fit_transform(res_x, res_y)

        cancor = []
        for i in range(n_components):
            cancor.append(np.corrcoef(res_x_c[:, [i]].T, res_y_c[:, [i]].T)[0, 1])

        coef = (np.array(cancor) ** 2).sum()

        # Step 6: Compute p-value using F-approximation.
        s = min(res_x.shape[1], res_y.shape[1])
        df1 = res_x.shape[1] * res_y.shape[1]
        df2 = s * (data.shape[0] - 1 + s - res_x.shape[1] - res_y.shape[1])
        f_stat = (coef / df1) * (df2 / (s - coef))
        p_value = 1 - stats.f.cdf(f_stat, df1, df2)

        self.statistic_ = coef
        self.p_value_ = p_value

        return self.statistic_, self.p_value_
