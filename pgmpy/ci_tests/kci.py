import numpy as np
import pandas as pd
from scipy import stats
from sklearn.gaussian_process.kernels import RBF, Kernel

from .hsic import HSIC


def _gamma_pvalue_from_moments(test_stat: float, mean: float, var: float) -> float:
    """Gamma-approximation p-value from precomputed null moments."""
    if var <= 0 or mean <= 0:
        return 1.0

    k = mean**2 / var
    theta = var / mean
    return float(1.0 - stats.gamma.cdf(test_stat, a=k, scale=theta))


class KCI(HSIC):
    r"""
    Kernel-based Conditional Independence (KCI) test [1].

    Extends :class:`HSIC` to the conditional case :math:`X \perp\!\!\!\perp Y \mid Z`.
    When ``Z`` is empty the test falls back to HSIC. The conditional statistic is

    .. math::
        T_{CI} = \frac{1}{n}\operatorname{Tr}
            \bigl(\tilde{K}_{\ddot{X}|Z}\tilde{K}_{Y|Z}\bigr),

    where :math:`\tilde{K}_{\cdot|Z} = R_Z \tilde{K}_\cdot R_Z` and
    :math:`R_Z = \varepsilon (K_Z + \varepsilon I)^{-1}`. A Gamma approximation
    is used for the null distribution.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataset containing variables X, Y, and optionally Z.
    kernel_X, kernel_Y, kernel_Z : sklearn.gaussian_process.kernels.Kernel or None
        Kernels for each variable. Default: RBF with bandwidth heuristic.
    bandwidth : {"empirical", "median"}, default="empirical"
        Bandwidth heuristic when kernel is None.
    epsilon : float, default=1e-3
        Tikhonov regularization added to :math:`K_Z` before inversion to ensure
        numerical stability of the residualization operator
        :math:`R_Z = \varepsilon (K_Z + \varepsilon I)^{-1}`.

    Attributes
    ----------
    statistic_ : float
        The KCI test statistic. Set after calling the test.
    p_value_ : float
        The p-value for the test. Set after calling the test.

    References
    ----------
    .. [1] Zhang et al. (2011). Kernel-based Conditional Independence
        Test and Application in Causal Discovery. UAI 2011.
    .. [2] causal-learn: Causal Discovery in Python.
        https://causal-learn.readthedocs.io/en/latest/

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from pgmpy.ci_tests import KCI
    >>> rng = np.random.default_rng(seed=42)
    >>> data = pd.DataFrame(rng.standard_normal((300, 3)), columns=["X", "Y", "Z"])
    >>> test = KCI(data=data)
    >>> test("X", "Y", ["Z"], significance_level=0.05)
    True
    >>> test.statistic_  # doctest: +SKIP
    2.66...
    >>> test.p_value_  # doctest: +SKIP
    0.55...
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
        """Piecewise RBF length-scale for the conditional path [2]."""
        n = Z.shape[0]
        if n < 200:
            width = 1.2
        elif n < 1200:
            width = 0.7
        else:
            width = 0.4
        return width * np.sqrt(Z.shape[1])

    def _get_length_scale_kci(self, Z: np.ndarray) -> float:
        if self.bandwidth == "median":
            return self._median_width(Z)
        return self._empirical_width_kci(Z)

    def _get_uu_prod(self, KxR: np.ndarray, KyR: np.ndarray, thresh: float = 1e-5) -> np.ndarray:
        """Truncated eigendecomposition product for null-moment estimation."""
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
        n = x.shape[0]
        ls = self._get_length_scale_kci(z)

        kernel_x = self.kernel_X if self.kernel_X is not None else RBF(length_scale=ls)
        kernel_y = self.kernel_Y if self.kernel_Y is not None else RBF(length_scale=ls)
        kernel_z = self.kernel_Z if self.kernel_Z is not None else RBF(length_scale=ls)

        # Augment X with half-weighted Z, following the causal-learn reference [2].
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

    def run_test(self, X: str, Y: str, Z: list):
        """
        Compute KCI statistic and p-value.

        Sets ``self.statistic_`` and ``self.p_value_``.
        """
        if len(Z) == 0:
            return super().run_test(X, Y, Z)

        x = self.data[X].to_numpy(dtype=float).reshape(-1, 1)
        y = self.data[Y].to_numpy(dtype=float).reshape(-1, 1)
        z = self.data[Z].to_numpy(dtype=float)

        x = np.nan_to_num(stats.zscore(x, ddof=1, axis=0))
        y = np.nan_to_num(stats.zscore(y, ddof=1, axis=0))
        z = np.nan_to_num(stats.zscore(z, ddof=1, axis=0))

        self.statistic_, self.p_value_ = self._conditional_test(x, y, z)
        return self.statistic_, self.p_value_
