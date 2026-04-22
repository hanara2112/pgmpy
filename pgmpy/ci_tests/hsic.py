import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist
from sklearn.gaussian_process.kernels import RBF, Kernel

from ._base import _BaseCITest


class HSIC(_BaseCITest):
    r"""
    Hilbert-Schmidt Independence Criterion (HSIC) [1] test for marginal
    independence of continuous variables.

    With double-centered kernel Gram matrices :math:`\tilde K_X, \tilde K_Y`,
    the V-statistic is

    .. math::
        T = \operatorname{Tr}(\tilde{K}_X \tilde{K}_Y),
        \quad \tilde{K} = HKH, \quad H = I - \tfrac{1}{n}\mathbf{1}\mathbf{1}^T.

    Under the null, :math:`T` is asymptotically a weighted sum of
    :math:`\chi^2_1` variables. The p-value is computed either from a
    moment-matched Gamma approximation with shape :math:`k=\mu^2/\sigma^2`
    and scale :math:`\theta=\sigma^2/\mu` (closed-form null moments from [2]),
    or from a permutation test on Y [1].

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset in which to test the independence condition.
    kernel : sklearn Kernel, tuple of two Kernels, or None
        Kernel(s) for X and Y; a single object is shared by both. Default: RBF
        with bandwidth heuristic.
    bandwidth : {"heuristic", "median"}, default="heuristic"
        Bandwidth heuristic when kernel is None.
    null_dist : {"gamma", "permutation"}, default="gamma"
        Null approximation. ``"gamma"`` [2] is 10-100x faster than
        ``"permutation"`` [1] and agrees within ~0.02; prefer
        ``"permutation"`` for small n or non-RBF kernels.
    n_permutations : int, default=500
        Number of permutations (used only when ``null_dist="permutation"``).
    random_state : int, np.random.Generator, or None, default=None
        Seed for permutation reproducibility.

    Examples
    --------
    >>> from pgmpy.datasets import load_dataset
    >>> from pgmpy.ci_tests import HSIC
    >>> data = load_dataset("tubingen/1").data
    >>> test = HSIC(data=data)
    >>> test("x", "y", significance_level=0.05)
    False
    >>> round(test.statistic_, 2)
    5468.71
    >>> round(test.p_value_, 2)
    0.0

    Attributes
    ----------
    statistic_ : float
        The HSIC V-statistic. Set after calling the test.
    p_value_ : float
        The p-value for the test. Set after calling the test.

    References
    ----------
    .. [1] Gretton et al. (2005). Measuring Statistical Dependence with
           Hilbert-Schmidt Norms. ALT 2005.
    .. [2] Gretton et al. (2007). A Kernel Statistical Test of Independence.
           NeurIPS 2007.
    """

    _tags = {
        "name": "hsic",
        "data_types": ("continuous",),
        "default_for": None,
        "requires_data": True,
    }

    def __init__(
        self,
        data: pd.DataFrame,
        kernel: Kernel | tuple | None = None,
        bandwidth: str = "heuristic",
        null_dist: str = "gamma",
        n_permutations: int = 500,
        random_state: int | np.random.Generator | None = None,
    ):
        if bandwidth not in ("heuristic", "median"):
            raise ValueError(f"bandwidth must be 'heuristic' or 'median', got {bandwidth!r}")
        if null_dist not in ("gamma", "permutation"):
            raise ValueError(f"null_dist must be 'gamma' or 'permutation', got {null_dist!r}")
        if n_permutations < 1:
            raise ValueError(f"n_permutations must be >= 1, got {n_permutations!r}")

        self.data = data
        if kernel is None or isinstance(kernel, Kernel):
            self.kernel_X = self.kernel_Y = kernel
        else:
            self.kernel_X, self.kernel_Y = kernel
        self.bandwidth = bandwidth
        self.null_dist = null_dist
        self.n_permutations = n_permutations
        self.random_state = random_state
        super().__init__()

    @staticmethod
    def _median_width(X: np.ndarray) -> float:
        """Median heuristic: length-scale = sqrt(2) * median(euclidean_dists)."""
        med = np.median(pdist(X, metric="euclidean"))
        return np.sqrt(2.0) * med if med > 0 else 1.0

    def _length_scale(self, X: np.ndarray) -> float:
        """RBF length-scale from ``bandwidth`` (piecewise heuristic or median)."""
        if self.bandwidth == "median":
            return self._median_width(X)
        n = X.shape[0]
        width = 0.8 if n < 200 else (0.5 if n < 1200 else 0.3)
        return width / np.sqrt(X.shape[1])

    @staticmethod
    def _center_kernel(K: np.ndarray) -> np.ndarray:
        """Double-centre a kernel matrix."""
        col_mean = K.mean(axis=0)
        return K - col_mean[None, :] - col_mean[:, None] + col_mean.mean()

    def _hsic_gamma_pvalue(self, test_stat, K, L, Kc, Lc):
        """Gamma p-value using exact null moments (Proposition 6(i) of [2])."""
        n = K.shape[0]
        # For n < 6 the variance formula breaks or is zero; return p = 1.0.
        if n < 6:
            return 1.0

        # Null variance for T/n.
        M_sq = (Kc * Lc / 6.0) ** 2
        off_diag_sq_sum = M_sq.sum() - np.trace(M_sq)
        var_hsic = off_diag_sq_sum / (n * (n - 1))
        moment_factor = 72.0 * (n - 4) * (n - 5) / (n * (n - 1) * (n - 2) * (n - 3))
        var_hsic *= moment_factor

        # Null mean for T/n from off-diagonal kernel means.
        mu_x = (K.sum() - np.trace(K)) / (n * (n - 1))
        mu_y = (L.sum() - np.trace(L)) / (n * (n - 1))
        mean_hsic = (1.0 + mu_x * mu_y - mu_x - mu_y) / n

        # Guards against degenerate inputs (see PR discussion).
        if var_hsic <= 0 or mean_hsic <= 0:
            return 1.0

        alpha = mean_hsic**2 / var_hsic
        beta = var_hsic * n / mean_hsic
        return 1.0 - stats.gamma.cdf(test_stat / n, a=alpha, scale=beta)

    def _permutation_pvalue(self, test_stat, Kxc, y, kernel_y):
        """Empirical p-value by permuting Y (Section 5.1 of [1])."""
        rng = np.random.default_rng(self.random_state)
        n = y.shape[0]
        null_stats = np.empty(self.n_permutations)
        for i in range(self.n_permutations):
            Lyc = self._center_kernel(kernel_y(y[rng.permutation(n)]))
            null_stats[i] = np.sum(Kxc * Lyc)
        return (1 + np.sum(null_stats >= test_stat)) / (1 + self.n_permutations)

    def run_test(self, X: str, Y: str, Z: list):
        r"""
        Test :math:`X \perp Y`. Z must be empty; use KCI for conditional tests.

        Returns (statistic, p_value) and sets self.statistic_, self.p_value_.
        """
        if Z:
            raise ValueError("HSIC does not support conditioning variables. Use KCI instead.")

        x = self.data[X].to_numpy(dtype=float).reshape(-1, 1)
        y = self.data[Y].to_numpy(dtype=float).reshape(-1, 1)
        x = np.nan_to_num(stats.zscore(x, ddof=1, axis=0))
        y = np.nan_to_num(stats.zscore(y, ddof=1, axis=0))

        kernel_x = self.kernel_X if self.kernel_X is not None else RBF(length_scale=self._length_scale(x))
        kernel_y = self.kernel_Y if self.kernel_Y is not None else RBF(length_scale=self._length_scale(y))

        Kx, Ky = kernel_x(x), kernel_y(y)
        Kxc, Kyc = self._center_kernel(Kx), self._center_kernel(Ky)
        test_stat = np.sum(Kxc * Kyc)

        if self.null_dist == "permutation":
            p_value = self._permutation_pvalue(test_stat, Kxc, y, kernel_y)
        else:
            p_value = self._hsic_gamma_pvalue(test_stat, Kx, Ky, Kxc, Kyc)

        self.statistic_ = float(test_stat)
        self.p_value_ = float(p_value)
        return self.statistic_, self.p_value_
