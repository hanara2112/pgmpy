import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist
from sklearn.gaussian_process.kernels import RBF, Kernel

from ._base import _BaseCITest


class HSIC(_BaseCITest):
    r"""
    Hilbert-Schmidt Independence Criterion (HSIC) test for marginal independence [1][2].

    HSIC detects nonlinear dependencies by measuring the distance between the
    joint distribution and the product of marginals in a reproducing kernel
    Hilbert space (RKHS). The V-statistic is:

    .. math::
        T = \operatorname{Tr}(\tilde{K}_X \tilde{K}_Y),
        \quad \tilde{K} = HKH, \quad H = I - \tfrac{1}{n}\mathbf{1}\mathbf{1}^T.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataset containing the variables to test.
    kernel_X, kernel_Y : sklearn.gaussian_process.kernels.Kernel or None
        Kernels for X and Y. Default: RBF with bandwidth heuristic.
    bandwidth : {"empirical", "median"}, default="empirical"
        Bandwidth heuristic when kernel is None.
    null_dist : {"gamma", "permutation"}, default="gamma"
        Null distribution method. Gamma uses exact moments from [2];
        permutation shuffles Y ``n_permutations`` times.
    n_permutations : int, default=500
        Number of permutations (only for ``null_dist="permutation"``).
    random_state : int, np.random.Generator, or None, default=None
        Seed for permutation reproducibility.

    Attributes
    ----------
    statistic_ : float
        The HSIC V-statistic. Set after calling the test.
    p_value_ : float
        The p-value for the test. Set after calling the test.

    References
    ----------
    .. [1] Gretton et al. (2005). Measuring Statistical Dependence
        with Hilbert-Schmidt Norms. ALT 2005.
    .. [2] Gretton et al. (2007). A Kernel Statistical Test of
        Independence. NeurIPS 2007.
    .. [3] Zhang et al. (2011). Kernel-based conditional independence
        test and application in causal discovery. UAI 2011.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from pgmpy.ci_tests import HSIC
    >>> rng = np.random.default_rng(seed=42)
    >>> data = pd.DataFrame(rng.standard_normal((300, 3)), columns=["X", "Y", "Z"])
    >>> test = HSIC(data=data)
    >>> test("X", "Y", [], significance_level=0.05)
    True
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
        kernel_X: Kernel | None = None,
        kernel_Y: Kernel | None = None,
        bandwidth: str = "empirical",
        null_dist: str = "gamma",
        n_permutations: int = 500,
        random_state: int | np.random.Generator | None = None,
    ):
        if bandwidth not in ("empirical", "median"):
            raise ValueError(f"bandwidth must be 'empirical' or 'median', got {bandwidth!r}")
        if null_dist not in ("gamma", "permutation"):
            raise ValueError(f"null_dist must be 'gamma' or 'permutation', got {null_dist!r}")
        if n_permutations < 1:
            raise ValueError(f"n_permutations must be >= 1, got {n_permutations!r}")

        self.data = data
        self.kernel_X = kernel_X
        self.kernel_Y = kernel_Y
        self.bandwidth = bandwidth
        self.null_dist = null_dist
        self.n_permutations = n_permutations
        self.random_state = random_state
        super().__init__()

    def _empirical_width(self, X: np.ndarray) -> float:
        """Piecewise RBF length-scale from the KCI Matlab reference [3]."""
        n = X.shape[0]
        width = 0.8 if n < 200 else (0.5 if n < 1200 else 0.3)
        return width / np.sqrt(X.shape[1])

    def _median_width(self, X: np.ndarray) -> float:
        """Median heuristic: length-scale = sqrt(2) * median(euclidean_dists)."""
        med = np.median(pdist(X, metric="euclidean"))
        return np.sqrt(2.0) * med if med > 0 else 1.0

    def _length_scale(self, X: np.ndarray) -> float:
        if self.bandwidth == "median":
            return self._median_width(X)
        return self._empirical_width(X)

    @staticmethod
    def _center_kernel(K: np.ndarray) -> np.ndarray:
        """Double-centre a kernel matrix."""
        col_mean = K.mean(axis=0)
        return K - col_mean[None, :] - col_mean[:, None] + col_mean.mean()

    def _make_kernel(self, user_kernel, X):
        if user_kernel is not None:
            return user_kernel
        return RBF(length_scale=self._length_scale(X))

    def _hsic_gamma_pvalue(self, test_stat, K, L, Kc, Lc):
        """Gamma p-value using exact null moments (Proposition 6(i) of [2])."""
        n = K.shape[0]
        if n < 6:
            return 1.0

        # Null variance for T/n.
        M_sq = (Kc * Lc / 6.0) ** 2
        var_hsic = (M_sq.sum() - np.trace(M_sq)) / (n * (n - 1))
        var_hsic *= 72.0 * (n - 4) * (n - 5) / (n * (n - 1) * (n - 2) * (n - 3))

        # Null mean for T/n from off-diagonal kernel means.
        mu_x = (K.sum() - np.trace(K)) / (n * (n - 1))
        mu_y = (L.sum() - np.trace(L)) / (n * (n - 1))
        mean_hsic = (1.0 + mu_x * mu_y - mu_x - mu_y) / n

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

        kernel_x = self._make_kernel(self.kernel_X, x)
        kernel_y = self._make_kernel(self.kernel_Y, y)

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
