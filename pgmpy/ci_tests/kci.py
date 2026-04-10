import numpy as np
import pandas as pd
from scipy import stats
from sklearn.gaussian_process.kernels import RBF, Kernel

from .hsic import HSIC


def _gamma_pvalue_from_moments(test_stat: float, mean: float, var: float) -> float:
    """Return a Gamma-approximation p-value from precomputed null moments."""
    if var <= 0 or mean <= 0:
        return 1.0

    k = mean**2 / var
    theta = var / mean
    return float(1.0 - stats.gamma.cdf(test_stat, a=k, scale=theta))


class KCI(HSIC):
    r"""
    Kernel-based Conditional Independence (KCI) test [1].

    Extends :class:`HSIC` to handle the conditional case
    :math:`X \perp\!\!\!\perp Y \mid Z`. When ``Z`` is empty, the test
    falls back to the HSIC V-statistic (unconditional path, inherited from
    :class:`HSIC`).

    Given *n* i.i.d. observations of continuous variables :math:`X`, :math:`Y`, and
    :math:`Z`, define the centred kernel matrix:

    .. math::
        \tilde{K}_X = H K_X H, \quad
        H = I - \tfrac{1}{n}\mathbf{1}\mathbf{1}^T.

    **Unconditional test** (:math:`Z = \emptyset`).
    Delegates to :class:`HSIC`. The statistic reduces to the HSIC V-statistic
    (Theorem 4, Eq. 8 of [1]):

    .. math::
        T_{UI} = \operatorname{Tr}(\tilde{K}_X \tilde{K}_Y).

    **Conditional test** (:math:`Z \neq \emptyset`).
    Let :math:`\ddot{X} = (X, Z)`. Following the causal-learn reference
    implementation [2] (rather than the product-kernel formulation in the paper),
    :math:`\ddot{X}` is formed by concatenating :math:`X` with :math:`0.5 \cdot Z`
    and evaluated with a single RBF kernel. The residualization operator is (Eq. 10 of [1]):

    .. math::
        R_Z = I - K_Z (K_Z + \varepsilon I)^{-1}
            \;\; \equiv \;\; \varepsilon (K_Z + \varepsilon I)^{-1}.

    The residual kernel matrices (Eqs. 11–12 of [1]) are:

    .. math::
        \tilde{K}_{\ddot{X}|Z} = R_Z \tilde{K}_{\ddot{X}} R_Z, \qquad
        \tilde{K}_{Y|Z} = R_Z \tilde{K}_Y R_Z.

    The KCI test statistic (Proposition 5, Eq. 13 of [1]) is:

    .. math::
        T_{CI} = \frac{1}{n}\operatorname{Tr}
            \bigl(\tilde{K}_{\ddot{X}|Z}\tilde{K}_{Y|Z}\bigr).

    **Null distribution.** Under :math:`H_0`, :math:`T_{CI}` follows a weighted
    sum of chi-squared variables. A Gamma approximation is used in practice
    (Sec. 3.4 of [1]):

    .. math::
        T \sim \Gamma(k, \theta), \quad
        k = \frac{E[T]^2}{\operatorname{Var}[T]}, \quad
        \theta = \frac{\operatorname{Var}[T]}{E[T]}.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataset containing variables :math:`X`, :math:`Y`, and optionally :math:`Z`.

    kernel_X : sklearn.gaussian_process.kernels.Kernel or None, default=None
        Kernel for :math:`X`. If ``None``, an RBF kernel is built automatically
        using the heuristic specified by ``bandwidth``.

    kernel_Y : sklearn.gaussian_process.kernels.Kernel or None, default=None
        Kernel for :math:`Y`. Same behaviour as ``kernel_X``.

    kernel_Z : sklearn.gaussian_process.kernels.Kernel or None, default=None
        Kernel for :math:`Z`. Same behaviour as ``kernel_X``.

    bandwidth : {"empirical", "median"}, default="empirical"
        Bandwidth selection heuristic used when any of the kernel arguments
        is ``None``. ``"empirical"`` uses the piecewise rule from the KCI
        Matlab reference; ``"median"`` uses the median pairwise distance.

    epsilon : float, default=1e-3
        Regularization parameter :math:`\varepsilon` added to the diagonal of
        :math:`K_Z` before inversion for numerical stability
        (:math:`R_Z = \varepsilon (K_Z + \varepsilon I)^{-1}`).
        Matches the value hardcoded in the causal-learn reference [2].

    Attributes
    ----------
    statistic_ : float
        Test statistic (:math:`T_{UI}` or :math:`T_{CI}`).

    p_value_ : float
        P-value computed via Gamma approximation.

    References
    ----------
    .. [1] Zhang, K., Peters, J., Janzing, D., & Schölkopf, B. (2011).
        Kernel-based Conditional Independence Test and Application in Causal Discovery.
        UAI 2011. https://arxiv.org/abs/1202.3775
    .. [2] Zheng, X., et al. causal-learn: Causal Discovery in Python.
        https://causal-learn.readthedocs.io/en/latest/

    Examples
    --------
    Test unconditional independence (delegates to HSIC):

    >>> import numpy as np
    >>> import pandas as pd
    >>> from pgmpy.ci_tests import KCI
    >>> rng = np.random.default_rng(seed=42)
    >>> data = pd.DataFrame(rng.standard_normal((300, 3)), columns=["X", "Y", "Z"])
    >>> test = KCI(data=data)
    >>> test("X", "Y", [], significance_level=0.05)
    True
    >>> test.statistic_  # doctest: +SKIP
    139.97...
    >>> test.p_value_  # doctest: +SKIP
    0.39...

    Test conditional independence:

    >>> test("X", "Y", ["Z"], significance_level=0.05)
    True

    Use the median heuristic bandwidth:

    >>> test = KCI(data=data, bandwidth="median")
    >>> test("X", "Y", [], significance_level=0.05)
    True

    Pass custom kernels:

    >>> from sklearn.gaussian_process.kernels import RBF
    >>> test = KCI(data=data, kernel_X=RBF(length_scale=2.0))
    >>> test("X", "Y", [], significance_level=0.05)
    True
    """

    _tags = {
        "name": "kci",
        "data_types": ("continuous",),
        "default_for": None,
        "requires_data": True,
    }

    def __init__(
        self,
        data: pd.DataFrame,
        kernel_X: Kernel | None = None,
        kernel_Y: Kernel | None = None,
        kernel_Z: Kernel | None = None,
        bandwidth: str = "empirical",
        epsilon: float = 1e-3,
    ):
        super().__init__(data=data, kernel_X=kernel_X, kernel_Y=kernel_Y, bandwidth=bandwidth)
        self.kernel_Z = kernel_Z
        self.epsilon = epsilon

    def _empirical_width_kci(self, Z: np.ndarray) -> float:
        """Piecewise RBF length_scale for conditional path [2]. length_scale = w * sqrt(d)."""
        n = Z.shape[0]
        if n < 200:
            width = 1.2
        elif n < 1200:
            width = 0.7
        else:
            width = 0.4
        return width * np.sqrt(Z.shape[1])

    def _get_length_scale_kci(self, Z: np.ndarray) -> float:
        """Dispatch to the bandwidth heuristic for the conditional path."""
        if self.bandwidth == "median":
            return self._median_width(Z)
        return self._empirical_width_kci(Z)

    def _get_uu_prod(self, KxR: np.ndarray, KyR: np.ndarray, thresh: float = 1e-5) -> np.ndarray:
        """Truncated eigendecomposition product UU^T used for null-moment estimation."""
        wx, vx = np.linalg.eigh(0.5 * (KxR + KxR.T))
        wy, vy = np.linalg.eigh(0.5 * (KyR + KyR.T))

        wx, vx = wx[np.argsort(-wx)], vx[:, np.argsort(-wx)]
        wy, vy = wy[np.argsort(-wy)], vy[:, np.argsort(-wy)]

        vx = vx[:, wx > np.max(wx) * thresh]
        wx = wx[wx > np.max(wx) * thresh]
        vy = vy[:, wy > np.max(wy) * thresh]
        wy = wy[wy > np.max(wy) * thresh]

        vx = vx @ np.diag(np.sqrt(wx))
        vy = vy @ np.diag(np.sqrt(wy))

        n = KxR.shape[0]
        num_eigx, num_eigy = vx.shape[1], vy.shape[1]
        size_u = num_eigx * num_eigy
        uu = np.zeros((n, size_u))
        for i in range(num_eigx):
            for j in range(num_eigy):
                uu[:, i * num_eigy + j] = vx[:, i] * vy[:, j]

        return uu @ uu.T if size_u > n else uu.T @ uu

    def _conditional_test(self, x: np.ndarray, y: np.ndarray, z: np.ndarray):
        """Residualize X and Y on Z via kernel ridge regression, then test with KCI statistic."""
        n = x.shape[0]
        ls = self._get_length_scale_kci(z)

        kernel_x = self.kernel_X if self.kernel_X is not None else RBF(length_scale=ls)
        kernel_y = self.kernel_Y if self.kernel_Y is not None else RBF(length_scale=ls)
        kernel_z = self.kernel_Z if self.kernel_Z is not None else RBF(length_scale=ls)

        # Augment X with half-weighted Z following the KCI reference implementation [2].
        Kx = self._center_kernel(kernel_x(np.hstack([x, 0.5 * z])))
        Ky = self._center_kernel(kernel_y(y))
        Kz = self._center_kernel(kernel_z(z))

        Rz = self.epsilon * np.linalg.pinv(Kz + self.epsilon * np.eye(n))
        KxR = Rz @ Kx @ Rz
        KyR = Rz @ Ky @ Rz

        test_stat = float(np.sum(KxR * KyR))

        uu_prod = self._get_uu_prod(KxR, KyR)
        mean = np.trace(uu_prod)
        var = 2.0 * np.trace(uu_prod @ uu_prod)

        return test_stat, _gamma_pvalue_from_moments(test_stat, mean, var)

    def run_test(
        self,
        X: str,
        Y: str,
        Z: list,
    ):
        r"""
        Compute KCI statistic and p-value for :math:`X \perp Y \mid Z`.

        Delegates to the inherited :class:`HSIC` path when ``Z`` is empty,
        and runs the full kernel ridge-regression procedure otherwise.

        Parameters
        ----------
        X : str
            Column name of the first variable.
        Y : str
            Column name of the second variable.
        Z : list
            Column names of the conditioning variables. Pass ``[]`` for the
            unconditional (HSIC) test.

        Returns
        -------
        statistic : float
            Observed KCI statistic.
        p_value : float
            P-value from the Gamma approximation.
        """
        if len(Z) == 0:
            return super().run_test(X, Y, Z)

        data = self.data

        x = data.loc[:, X].to_numpy().reshape(-1, 1).astype(float)
        y = data.loc[:, Y].to_numpy().reshape(-1, 1).astype(float)
        z = data.loc[:, Z].to_numpy().astype(float)

        x = stats.zscore(x, ddof=1, axis=0)
        x = np.nan_to_num(x, nan=0.0)
        y = stats.zscore(y, ddof=1, axis=0)
        y = np.nan_to_num(y, nan=0.0)
        z = stats.zscore(z, ddof=1, axis=0)
        z = np.nan_to_num(z, nan=0.0)

        self.statistic_, self.p_value_ = self._conditional_test(x, y, z)
        return self.statistic_, self.p_value_
