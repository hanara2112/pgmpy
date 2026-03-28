import numpy as np
import pandas as pd
from scipy import stats
from sklearn.gaussian_process.kernels import RBF, Kernel

from ._base import _BaseCITest


class KCI(_BaseCITest):
    r"""
    Kernel-based Conditional Independence (KCI) test [1].

    This test evaluates whether :math:`X \perp\!\!\!\perp Y \mid Z` by checking whether
    the residuals of :math:`X` and :math:`Y` (after removing the effect of :math:`Z`)
    are independent in a reproducing kernel Hilbert space (RKHS). It captures
    nonlinear dependencies without explicit density estimation.

    Given *n* i.i.d. observations of continuous variables :math:`X`, :math:`Y`, and
    (optionally) :math:`Z`, define the centred kernel matrix:

    .. math::
        \tilde{K}_X = H K_X H, \quad
        H = I - \tfrac{1}{n}\mathbf{1}\mathbf{1}^T.

    **Unconditional test** (:math:`Z = \emptyset`).
    The statistic reduces to the HSIC V-statistic (Theorem 4, Eq. 8 of [1]):

    .. math::
        T_{UI} = \frac{1}{n}\operatorname{Tr}(\tilde{K}_X \tilde{K}_Y).

    Under the null, its moments are (Proposition 6(i) of [1]):

    .. math::
        E(T_{UI} \mid D) = \frac{1}{n^2}
            \operatorname{Tr}(\tilde{K}_X)\operatorname{Tr}(\tilde{K}_Y), \\
        \operatorname{Var}(T_{UI} \mid D) = \frac{2}{n^4}
            \operatorname{Tr}(\tilde{K}_X^2)\operatorname{Tr}(\tilde{K}_Y^2).

    **Conditional test** (:math:`Z \neq \emptyset`).
    To remove the influence of :math:`Z`, kernel ridge regression is applied.
    Let :math:`\ddot{X} = (X, Z)`. The residualization operator is (Eq. 10 of [1]):

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

    **Null distribution and p-value.**
    Under :math:`H_0`, the statistic follows a weighted sum of chi-square variables.
    In practice, a Gamma approximation is used (Sec. 3.4 of [1]):

    .. math::
        T \sim \Gamma(k, \theta), \quad
        k = \frac{E^2}{Var}, \quad \theta = \frac{Var}{E}.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataset containing variables :math:`X`, :math:`Y`, and optionally :math:`Z`.

    kernel_X : sklearn.gaussian_process.kernels.Kernel or None, default=None
        Kernel for :math:`X`. If ``None``, an RBF kernel is created with bandwidth
        set via the empirical width rule from the original Matlab implementation.

    kernel_Y : sklearn.gaussian_process.kernels.Kernel or None, default=None
        Kernel for :math:`Y`. Same bandwidth behaviour as ``kernel_X``.

    kernel_Z : sklearn.gaussian_process.kernels.Kernel or None, default=None
        Kernel for :math:`Z`. Same bandwidth behaviour as ``kernel_X``.

    epsilon : float, default=1e-3
        Regularization parameter :math:`\varepsilon` for kernel ridge regression
        when residualizing with respect to :math:`Z`.

    Attributes
    ----------
    statistic_ : float
        Test statistic (:math:`T_{UI}` or :math:`T_{CI}`).

    p_value_ : float
        P-value computed via Gamma approximation.

    References
    ----------
    .. [1] Zhang, K., Peters, J., Janzing, D., Schölkopf, B. (2011).
        Kernel-based Conditional Independence Test and Application in Causal Discovery.
        UAI 2011. https://arxiv.org/abs/1202.3775
    .. [2] Zheng, X., et al. causal-learn: Causal Discovery in Python.
        https://causal-learn.readthedocs.io/en/latest/

    Examples
    --------
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

    Using a custom kernel:

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
        epsilon: float = 1e-3,
    ):
        self.data = data
        self.kernel_X = kernel_X
        self.kernel_Y = kernel_Y
        self.kernel_Z = kernel_Z
        self.epsilon = epsilon
        super().__init__()

    def _empirical_width_hsic(self, X):
        """Empirical RBF length_scale for unconditional test [2]. theta = (1/w^2)*d."""
        n = X.shape[0]
        if n < 200:
            width = 0.8
        elif n < 1200:
            width = 0.5
        else:
            width = 0.3
        # theta = 1/length_scale^2 => length_scale = width / sqrt(d)
        return width / np.sqrt(X.shape[1])

    def _empirical_width_kci(self, Z):
        """Empirical RBF length_scale for conditional test [2]. theta = (1/w^2)/d."""
        n = Z.shape[0]
        if n < 200:
            width = 1.2
        elif n < 1200:
            width = 0.7
        else:
            width = 0.4
        # theta = 1/length_scale^2 => length_scale = width * sqrt(d)
        return width * np.sqrt(Z.shape[1])

    def _center_kernel(self, K):
        """Center a kernel matrix: H @ K @ H where H = I - 1/n.

        Computed in O(n^2) without forming H explicitly.
        """
        n = K.shape[0]
        K_colsums = K.sum(axis=0)
        K_allsum = K_colsums.sum()
        return K - (K_colsums[None, :] + K_colsums[:, None]) / n + (K_allsum / n**2)

    def _gamma_pvalue(self, test_stat, mean, var):
        """Compute p-value using gamma approximation."""
        if var <= 0 or mean <= 0:
            return 1.0

        k = mean**2 / var  # shape
        theta = var / mean  # scale
        return 1.0 - stats.gamma.cdf(test_stat, a=k, scale=theta)

    def _unconditional_test(self, x, y):
        """HSIC-based unconditional independence test (X _|_ Y).

        Returns ``(statistic, p_value)`` via gamma approximation.
        """
        n = x.shape[0]

        if self.kernel_X is not None:
            kernel_x = self.kernel_X
        else:
            ls = self._empirical_width_hsic(x)
            kernel_x = RBF(length_scale=ls)

        if self.kernel_Y is not None:
            kernel_y = self.kernel_Y
        else:
            ls = self._empirical_width_hsic(y)
            kernel_y = RBF(length_scale=ls)

        Kx = kernel_x(x)
        Ky = kernel_y(y)
        Kxc = self._center_kernel(Kx)
        Kyc = self._center_kernel(Ky)

        test_stat = np.sum(Kxc * Kyc)

        mean = np.trace(Kxc) * np.trace(Kyc) / n
        var = 2.0 * np.sum(Kxc**2) * np.sum(Kyc**2) / n**2

        return test_stat, self._gamma_pvalue(test_stat, mean, var)

    def _get_uu_prod(self, KxR, KyR, thresh=1e-5):
        """Compute the product matrix whose trace gives null distribution moments."""
        n = KxR.shape[0]

        wx, vx = np.linalg.eigh(0.5 * (KxR + KxR.T))
        wy, vy = np.linalg.eigh(0.5 * (KyR + KyR.T))

        idx = np.argsort(-wx)
        idy = np.argsort(-wy)
        wx, vx = wx[idx], vx[:, idx]
        wy, vy = wy[idy], vy[:, idy]

        vx = vx[:, wx > np.max(wx) * thresh]
        wx = wx[wx > np.max(wx) * thresh]
        vy = vy[:, wy > np.max(wy) * thresh]
        wy = wy[wy > np.max(wy) * thresh]

        vx = vx @ np.diag(np.sqrt(wx))
        vy = vy @ np.diag(np.sqrt(wy))

        num_eigx = vx.shape[1]
        num_eigy = vy.shape[1]
        size_u = num_eigx * num_eigy
        uu = np.zeros((n, size_u))
        for i in range(num_eigx):
            for j in range(num_eigy):
                uu[:, i * num_eigy + j] = vx[:, i] * vy[:, j]

        if size_u > n:
            uu_prod = uu @ uu.T
        else:
            uu_prod = uu.T @ uu

        return uu_prod

    def _conditional_test(self, x, y, z):
        """Kernel-based conditional independence test (X _|_ Y | Z).

        Regresses out the effect of Z from kernel matrices and returns
        ``(statistic, p_value)`` via gamma approximation.
        """
        n = x.shape[0]
        epsilon = self.epsilon

        ls = self._empirical_width_kci(z)

        if self.kernel_X is not None:
            kernel_x = self.kernel_X
        else:
            kernel_x = RBF(length_scale=ls)

        if self.kernel_Y is not None:
            kernel_y = self.kernel_Y
        else:
            kernel_y = RBF(length_scale=ls)

        if self.kernel_Z is not None:
            kernel_z = self.kernel_Z
        else:
            kernel_z = RBF(length_scale=ls)

        Kx = kernel_x(np.hstack([x, 0.5 * z]))  # [2]
        Ky = kernel_y(y)
        Kz = kernel_z(z)

        Kx = self._center_kernel(Kx)
        Ky = self._center_kernel(Ky)
        Kz = self._center_kernel(Kz)

        Rz = epsilon * np.linalg.pinv(Kz + epsilon * np.eye(n))

        KxR = Rz @ Kx @ Rz
        KyR = Rz @ Ky @ Rz

        test_stat = np.sum(KxR * KyR)

        uu_prod = self._get_uu_prod(KxR, KyR)
        mean = np.trace(uu_prod)
        var = 2.0 * np.trace(uu_prod @ uu_prod)

        return test_stat, self._gamma_pvalue(test_stat, mean, var)

    def run_test(
        self,
        X: str,
        Y: str,
        Z: list,
    ):
        """
        Compute KCI test statistic and p-value.

        Sets ``self.statistic_`` (HSIC/KCI statistic) and ``self.p_value_``.
        """
        data = self.data

        x = data.loc[:, X].to_numpy().reshape(-1, 1).astype(float)
        y = data.loc[:, Y].to_numpy().reshape(-1, 1).astype(float)

        x = stats.zscore(x, ddof=1, axis=0)
        x = np.nan_to_num(x, nan=0.0)
        y = stats.zscore(y, ddof=1, axis=0)
        y = np.nan_to_num(y, nan=0.0)

        if len(Z) == 0:
            self.statistic_, self.p_value_ = self._unconditional_test(x, y)
        else:
            z = data.loc[:, Z].to_numpy().astype(float)
            z = stats.zscore(z, ddof=1, axis=0)
            z = np.nan_to_num(z, nan=0.0)
            self.statistic_, self.p_value_ = self._conditional_test(x, y, z)

        return self.statistic_, self.p_value_
