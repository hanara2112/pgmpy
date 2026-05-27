import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist
from sklearn.gaussian_process.kernels import RBF, Kernel

from ._base import _BaseCITest, _CITestResult


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
    or from a permutation test on Y [1]. Returns ``p_value_ = 1.0`` if
    ``n < 6`` (variance estimator is undefined).

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset in which to test the independence condition.
    kernel : sklearn Kernel, tuple of two Kernels, or None
        Kernel(s) for X and Y; a single object is shared by both. Default: RBF
        with bandwidth heuristic.
    bandwidth : {"heuristic", "median"}, default="heuristic"
        RBF length-scale rule when ``kernel`` is None.
        ``"heuristic"`` uses a piecewise width by sample size
        (``0.8`` if ``n < 200``, ``0.5`` if ``n < 1200``, else ``0.3``),
        scaled by ``1 / sqrt(d)`` for ``d``-dimensional inputs.
        ``"median"`` uses ``sqrt(2) * median`` of pairwise Euclidean
        distances (falling back to ``1.0`` if the median is zero).
    null_dist : {"gamma", "permutation"}, default="gamma"
        Null approximation. ``"gamma"`` [2] is faster but assumes RBF-like
        kernels; use ``"permutation"`` [1] for non-RBF kernels.
    n_permutations : int, default=100
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
    effect_size_ : None
        Not defined for HSIC.

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
        "is_symmetric": True,
    }

    def __init__(
        self,
        data: pd.DataFrame,
        kernel: Kernel | tuple | None = None,
        bandwidth: str = "heuristic",
        null_dist: str = "gamma",
        n_permutations: int = 100,
        random_state: int | np.random.Generator | None = None,
        use_cache: bool = True,
    ):
        if bandwidth not in ("heuristic", "median"):
            raise ValueError(f"bandwidth must be 'heuristic' or 'median', got {bandwidth!r}")
        if null_dist not in ("gamma", "permutation"):
            raise ValueError(f"null_dist must be 'gamma' or 'permutation', got {null_dist!r}")
        if n_permutations < 1:
            raise ValueError(f"n_permutations must be >= 1, got {n_permutations!r}")

        if kernel is None or isinstance(kernel, Kernel):
            self.kernel_X_ = self.kernel_Y_ = kernel
        elif isinstance(kernel, tuple) and len(kernel) == 2:
            self.kernel_X_, self.kernel_Y_ = kernel
        else:
            raise ValueError("kernel must be a sklearn Kernel, a tuple of two Kernels, or None")

        self.data = data
        self.bandwidth = bandwidth
        self.null_dist = null_dist
        self.n_permutations = n_permutations
        self.random_state = random_state
        super().__init__(use_cache=use_cache)

    @staticmethod
    def _median_width(X: np.ndarray) -> float:
        """Median heuristic: length-scale = sqrt(2) * median(euclidean_dists)."""
        med = np.median(pdist(X, metric="euclidean"))
        return np.sqrt(2.0) * med if med > 0 else 1.0

    def _length_scale(self, X: np.ndarray) -> float:
        """RBF length-scale: median heuristic, or piecewise width by sample size."""
        if self.bandwidth == "median":
            return self._median_width(X)
        n = X.shape[0]
        width = 0.8 if n < 200 else (0.5 if n < 1200 else 0.3)
        return width / np.sqrt(X.shape[1])

    @staticmethod
    def _center_kernel(K: np.ndarray) -> np.ndarray:
        """Double-center a kernel matrix."""
        col_mean = K.mean(axis=0)
        return K - col_mean[None, :] - col_mean[:, None] + col_mean.mean()

    def _hsic_gamma_pvalue(self, test_stat, K, L, Kc, Lc):
        """Gamma p-value using exact null moments (Proposition 6(i) of [2])."""
        n = K.shape[0]
        if n < 6:
            return 1.0

        M_sq = (Kc * Lc / 6.0) ** 2
        off_diag_sq_sum = M_sq.sum() - np.trace(M_sq)
        var_hsic = off_diag_sq_sum / (n * (n - 1))
        moment_factor = 72.0 * (n - 4) * (n - 5) / (n * (n - 1) * (n - 2) * (n - 3))
        var_hsic *= moment_factor

        mu_x = (K.sum() - np.trace(K)) / (n * (n - 1))
        mu_y = (L.sum() - np.trace(L)) / (n * (n - 1))
        mean_hsic = (1.0 + mu_x * mu_y - mu_x - mu_y) / n

        if var_hsic <= 0 or mean_hsic <= 0:
            return 1.0

        alpha = mean_hsic**2 / var_hsic
        beta = var_hsic * n / mean_hsic
        return 1.0 - stats.gamma.cdf(test_stat / n, a=alpha, scale=beta)

    def _compute_result(self, X: str, Y: str, Z: list):
        r"""
        Compute HSIC statistic and p-value for :math:`X \perp Y`.

        Z must be empty; use :class:`KCI` for conditional tests.
        """
        if Z:
            raise ValueError("HSIC does not support conditioning variables; use KCI.")

        x = self.data[X].to_numpy(dtype=float).reshape(-1, 1)
        y = self.data[Y].to_numpy(dtype=float).reshape(-1, 1)

        x_std, y_std = x.std(ddof=1), y.std(ddof=1)
        x = (x - x.mean()) / x_std if x_std > 0 else np.zeros_like(x)
        y = (y - y.mean()) / y_std if y_std > 0 else np.zeros_like(y)

        kernel_x = self.kernel_X_ if self.kernel_X_ is not None else RBF(length_scale=self._length_scale(x))
        kernel_y = self.kernel_Y_ if self.kernel_Y_ is not None else RBF(length_scale=self._length_scale(y))

        Kx, Ky = kernel_x(x), kernel_y(y)
        Kxc, Kyc = self._center_kernel(Kx), self._center_kernel(Ky)
        test_stat = np.sum(Kxc * Kyc)

        if self.null_dist == "permutation":
            # Permutation commutes with centering, so reindex Kyc instead of re-centering.
            rng = np.random.default_rng(self.random_state)
            n = y.shape[0]
            null_stats = np.empty(self.n_permutations)
            for i in range(self.n_permutations):
                perm = rng.permutation(n)
                null_stats[i] = np.sum(Kxc * Kyc[perm][:, perm])
            p_value = (1 + np.sum(null_stats >= test_stat)) / (1 + self.n_permutations)
        else:
            p_value = self._hsic_gamma_pvalue(test_stat, Kx, Ky, Kxc, Kyc)

        return _CITestResult(statistic=float(test_stat), p_value=float(p_value), effect_size=None)
