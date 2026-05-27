import numpy as np
import pandas as pd
from scipy import stats
from sklearn.gaussian_process.kernels import RBF, Kernel

from ._base import _CITestResult
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
    Kernel-based Conditional Independence (KCI) [1] test for conditional
    independence of continuous variables.

    Extends :class:`HSIC` to :math:`X \perp\!\!\!\perp Y \mid Z`; falls back to
    HSIC when ``Z`` is empty. The conditional statistic is

    .. math::
        T_{CI} = \frac{1}{n}\operatorname{Tr}\bigl(\tilde{K}_{\ddot{X}|Z}\tilde{K}_{Y|Z}\bigr),

    where :math:`\tilde{K}_{\cdot|Z} = R_Z \tilde{K}_\cdot R_Z` and
    :math:`R_Z = \varepsilon (K_Z + \varepsilon I)^{-1}`. Under the null,
    :math:`T_{CI}` is a weighted sum of :math:`\chi^2_1` approximated by a
    moment-matched Gamma with shape :math:`k=\mu^2/\sigma^2` and scale
    :math:`\theta=\sigma^2/\mu`, where :math:`\mu, \sigma^2` are estimated from
    the eigenspectrum of the residualized kernels [1].

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset in which to test the independence condition.
    kernel : sklearn Kernel, tuple of three Kernels, or None
        Kernel(s) for X, Y, Z; a single object is shared across all three.
        Default: RBF with bandwidth heuristic.
    bandwidth : {"heuristic", "median"}, default="heuristic"
        Bandwidth heuristic when kernel is None.
    epsilon : float, default=1e-3
        Tikhonov regularization for
        :math:`R_Z = \varepsilon (K_Z + \varepsilon I)^{-1}`.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pgmpy.ci_tests import KCI
    >>> rng = np.random.default_rng(seed=42)
    >>> data = pd.DataFrame(rng.standard_normal((300, 3)), columns=["X", "Y", "Z"])
    >>> test = KCI(data=data)
    >>> test("X", "Y", ["Z"], significance_level=0.05)
    True
    >>> round(test.statistic_, 2)
    95.19
    >>> round(test.p_value_, 2)
    0.33

    Attributes
    ----------
    statistic_ : float
        The KCI test statistic. Set after calling the test.
    p_value_ : float
        The p-value for the test. Set after calling the test.
    effect_size_ : None
        Not defined for KCI.

    References
    ----------
    .. [1] Zhang et al. (2011). Kernel-based Conditional Independence Test and Application in
           Causal Discovery. UAI 2011.
    .. [2] causal-learn: Causal Discovery in Python.
           https://causal-learn.readthedocs.io/en/latest/
    """

    _tags = {
        "name": "kci",
        "data_types": ("continuous",),
        "default_for": None,
        "requires_data": True,
        "is_symmetric": True,
    }

    def __init__(
        self,
        data: pd.DataFrame,
        kernel: Kernel | tuple | None = None,
        bandwidth: str = "heuristic",
        epsilon: float = 1e-3,
        use_cache: bool = True,
    ):
        if kernel is None or isinstance(kernel, Kernel):
            kernel_X = kernel_Y = kernel_Z = kernel
        elif isinstance(kernel, tuple) and len(kernel) == 3:
            kernel_X, kernel_Y, kernel_Z = kernel
        else:
            raise ValueError("kernel must be a sklearn Kernel, a tuple of three Kernels, or None")

        super().__init__(data=data, kernel=(kernel_X, kernel_Y), bandwidth=bandwidth, use_cache=use_cache)
        self.kernel_Z_ = kernel_Z
        self.epsilon = epsilon

    def _length_scale_kci(self, Z: np.ndarray) -> float:
        """RBF length-scale for KCI: median heuristic, or piecewise width by sample size [2]."""
        if self.bandwidth == "median":
            return self._median_width(Z)
        n = Z.shape[0]
        width = 1.2 if n < 200 else (0.7 if n < 1200 else 0.4)
        return width * np.sqrt(Z.shape[1])

    def _get_uu_prod(self, KxR: np.ndarray, KyR: np.ndarray, thresh: float = 1e-5) -> np.ndarray:
        """Gram matrix of the product spectrum; used to estimate null moments."""
        n = KxR.shape[0]
        wx, vx = np.linalg.eigh(0.5 * (KxR + KxR.T))
        wy, vy = np.linalg.eigh(0.5 * (KyR + KyR.T))

        mask_x = wx > np.max(wx) * thresh
        mask_y = wy > np.max(wy) * thresh
        vx = vx[:, mask_x] * np.sqrt(wx[mask_x])
        vy = vy[:, mask_y] * np.sqrt(wy[mask_y])

        uu = (vx[:, :, None] * vy[:, None, :]).reshape(n, -1)
        return uu @ uu.T if uu.shape[1] > n else uu.T @ uu

    def _conditional_test(self, x: np.ndarray, y: np.ndarray, z: np.ndarray):
        n = x.shape[0]
        ls = self._length_scale_kci(z)

        kernel_x = self.kernel_X_ if self.kernel_X_ is not None else RBF(length_scale=ls)
        kernel_y = self.kernel_Y_ if self.kernel_Y_ is not None else RBF(length_scale=ls)
        kernel_z = self.kernel_Z_ if self.kernel_Z_ is not None else RBF(length_scale=ls)

        # Augment X with 0.5-scaled Z (causal-learn default [2]).
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

    def _compute_result(self, X: str, Y: str, Z: list):
        """Compute KCI statistic and p-value; falls back to HSIC when Z is empty."""
        if not Z:
            return super()._compute_result(X, Y, Z)

        x = self.data[X].to_numpy(dtype=float).reshape(-1, 1)
        y = self.data[Y].to_numpy(dtype=float).reshape(-1, 1)
        z = self.data[Z].to_numpy(dtype=float)

        x_std, y_std = x.std(ddof=1), y.std(ddof=1)
        x = (x - x.mean()) / x_std if x_std > 0 else np.zeros_like(x)
        y = (y - y.mean()) / y_std if y_std > 0 else np.zeros_like(y)

        z_std = z.std(ddof=1, axis=0)
        z = np.where(z_std > 0, (z - z.mean(axis=0)) / z_std, 0.0)

        test_stat, p_value = self._conditional_test(x, y, z)
        return _CITestResult(statistic=test_stat, p_value=p_value, effect_size=None)
