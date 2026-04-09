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

    **Null distribution.** Two methods are available via ``null_dist``:

    - ``"gamma"`` (default): fits a Gamma distribution using the exact
      finite-sample moments from Proposition 6(i) of [2].
    - ``"permutation"``: empirical null by shuffling :math:`Y`
      ``n_permutations`` times.

    **Bandwidth selection.** Two heuristics are available via ``bandwidth``:

    - ``"empirical"`` (default): piecewise rule from the KCI Matlab reference [3].
    - ``"median"``: median pairwise-distance heuristic.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataset containing the variables to test.

    kernel_X : sklearn.gaussian_process.kernels.Kernel or None, default=None
        Kernel for :math:`X`. If ``None``, an RBF kernel is built using ``bandwidth``.

    kernel_Y : sklearn.gaussian_process.kernels.Kernel or None, default=None
        Kernel for :math:`Y`. Same behaviour as ``kernel_X``.

    bandwidth : {"empirical", "median"}, default="empirical"
        Bandwidth heuristic when a kernel is ``None``.

    null_dist : {"gamma", "permutation"}, default="gamma"
        Method for computing the p-value under :math:`H_0`.

    n_permutations : int, default=500
        Number of permutations. Only used when ``null_dist="permutation"``.

    random_state : int, numpy.random.Generator, or None, default=None
        Seed for the permutation null. Ignored when ``null_dist="gamma"``.

    Attributes
    ----------
    statistic_ : float
        Observed HSIC statistic :math:`T = \operatorname{Tr}(\tilde{K}_X \tilde{K}_Y)`.
    p_value_ : float
        P-value computed via the method specified by ``null_dist``.

    References
    ----------
    .. [1] Gretton, A., Bousquet, O., Smola, A., & Schölkopf, B. (2005).
        Measuring Statistical Dependence with Hilbert-Schmidt Norms. ALT 2005.
    .. [2] Gretton, A., Fukumizu, K., Teo, C. H., Song, L., Schölkopf, B., & Smola, A. J. (2007).
        A Kernel Statistical Test of Independence. NeurIPS 2007.
    .. [3] Zhang, K., Peters, J., Janzing, D., & Schölkopf, B. (2011).
        Kernel-based Conditional Independence Test and Application in Causal Discovery. UAI 2011.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from pgmpy.ci_tests import HSIC
    >>> rng = np.random.default_rng(seed=42)
    >>> data = pd.DataFrame(rng.standard_normal((300, 3)), columns=["X", "Y", "Z"])
    >>> test = HSIC(data=data)
    >>> test("X", "Y", [], significance_level=0.05)
    True
    >>> test.p_value_ > 0.05
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

    def _get_length_scale(self, X: np.ndarray) -> float:
        """Return RBF length-scale via the selected heuristic."""
        if self.bandwidth == "median":
            return self._median_width(X)
        return self._empirical_width(X)

    def _empirical_width(self, X: np.ndarray) -> float:
        """Piecewise RBF length_scale from the KCI Matlab reference [3]."""
        n = X.shape[0]
        width = 0.8 if n < 200 else (0.5 if n < 1200 else 0.3)
        return width / np.sqrt(X.shape[1])

    def _median_width(self, X: np.ndarray) -> float:
        """Median heuristic: length_scale = sqrt(2) * median(euclidean_dists)."""
        med = np.median(pdist(X, metric="euclidean"))
        return float(np.sqrt(2.0) * med) if med > 0 else 1.0

    @staticmethod
    def _center_kernel(K: np.ndarray) -> np.ndarray:
        """Doubly-centre K via H @ K @ H in O(n²) without forming H."""
        col_mean = K.mean(axis=0)
        return K - col_mean[None, :] - col_mean[:, None] + col_mean.mean()

    def _gamma_pvalue(self, test_stat: float, mean: float, var: float) -> float:
        """P-value via Gamma moment-matching from precomputed (mean, var).

        Used by :class:`KCI` for the conditional path, where the null moments
        come from eigendecomposition rather than kernel matrices.
        """
        if var <= 0 or mean <= 0:
            return 1.0
        k = mean**2 / var
        theta = var / mean
        return float(1.0 - stats.gamma.cdf(test_stat, a=k, scale=theta))

    def _hsic_gamma_pvalue(self, test_stat: float, K: np.ndarray, L: np.ndarray) -> float:
        """P-value using exact finite-sample null moments from Proposition 6(i) of [2]."""
        n = K.shape[0]
        if n < 6:
            return 1.0

        Kc, Lc = self._center_kernel(K), self._center_kernel(L)
        bone = np.ones((n, 1))

        # Exact finite-sample null variance for T / n.
        var_hsic = (np.sum((Kc * Lc / 6.0) ** 2) - np.trace((Kc * Lc / 6.0) ** 2)) / n / (n - 1)
        var_hsic *= 72.0 * (n - 4) * (n - 5) / n / (n - 1) / (n - 2) / (n - 3)

        # Exact finite-sample null mean for T / n from off-diagonal kernel means.
        K_nd = K - np.diag(np.diag(K))
        L_nd = L - np.diag(np.diag(L))
        mu_x = (bone.T @ K_nd @ bone).item() / n / (n - 1)
        mu_y = (bone.T @ L_nd @ bone).item() / n / (n - 1)
        mean_hsic = (1.0 + mu_x * mu_y - mu_x - mu_y) / n

        if var_hsic <= 0 or mean_hsic <= 0:
            return 1.0

        alpha = mean_hsic**2 / var_hsic
        beta = var_hsic * n / mean_hsic
        return float(1.0 - stats.gamma.cdf(test_stat / n, a=alpha, scale=beta))

    def _permutation_pvalue(
        self,
        test_stat: float,
        Kxc: np.ndarray,
        y: np.ndarray,
        kernel_y: Kernel,
    ) -> float:
        """Empirical p-value by permuting rows of Y (Section 4 of [1])."""
        rng = np.random.default_rng(self.random_state)
        n = y.shape[0]
        null_stats = np.empty(self.n_permutations)
        for i in range(self.n_permutations):
            Lyc = self._center_kernel(kernel_y(y[rng.permutation(n)]))
            null_stats[i] = float(np.sum(Kxc * Lyc))
        return float(np.mean(null_stats >= test_stat))

    def run_test(
        self,
        X: str,
        Y: str,
        Z: list,
    ):
        r"""
        Compute HSIC statistic and p-value for :math:`X \perp Y`.

        Sets ``self.statistic_`` and ``self.p_value_``. ``Z`` must be empty;
        use :class:`KCI` for conditional tests.
        """
        if len(Z) > 0:
            raise ValueError("HSIC is a marginal independence test and does not support conditioning. Use KCI instead.")

        x = np.nan_to_num(stats.zscore(self.data.loc[:, X].to_numpy().reshape(-1, 1).astype(float), ddof=1, axis=0))
        y = np.nan_to_num(stats.zscore(self.data.loc[:, Y].to_numpy().reshape(-1, 1).astype(float), ddof=1, axis=0))

        kernel_x = self.kernel_X if self.kernel_X is not None else RBF(length_scale=self._get_length_scale(x))
        kernel_y = self.kernel_Y if self.kernel_Y is not None else RBF(length_scale=self._get_length_scale(y))

        Kx, Ky = kernel_x(x), kernel_y(y)
        Kxc, Kyc = self._center_kernel(Kx), self._center_kernel(Ky)
        test_stat = float(np.sum(Kxc * Kyc))

        if self.null_dist == "permutation":
            p_value = self._permutation_pvalue(test_stat, Kxc, y, kernel_y)
        else:
            p_value = self._hsic_gamma_pvalue(test_stat, Kx, Ky)

        self.statistic_ = test_stat
        self.p_value_ = p_value
        return self.statistic_, self.p_value_
