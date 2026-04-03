import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist
from sklearn.gaussian_process.kernels import RBF, Kernel

from ._base import _BaseCITest


class HSIC(_BaseCITest):
    r"""
    Hilbert-Schmidt Independence Criterion (HSIC) test [1][2].

    An unconditional test of marginal independence :math:`X \perp\!\!\!\perp Y`.
    HSIC detects nonlinear dependencies by measuring the distance between the
    joint distribution and the product of marginals in a reproducing kernel
    Hilbert space (RKHS), without explicit density estimation.

    Given *n* i.i.d. observations of continuous variables :math:`X` and :math:`Y`,
    define the centred kernel matrix:

    .. math::
        \tilde{K}_X = H K_X H, \quad
        H = I - \tfrac{1}{n}\mathbf{1}\mathbf{1}^T.

    The HSIC V-statistic (Eq. 4 of [1], Theorem 4 of [2]) is:

    .. math::
        \widehat{\text{HSIC}}(X, Y) = \frac{1}{n^2}\operatorname{Tr}(\tilde{K}_X \tilde{K}_Y).

    **Null distribution.** Two methods are available via ``null_dist``:

    - ``"gamma"`` (default): fits a Gamma distribution to the first two moments
      of the null statistic — fast and analytic (Proposition 6(i) of [2]).
    - ``"permutation"``: builds an empirical null by randomly shuffling the rows
      of :math:`Y` and recomputing the statistic ``n_permutations`` times —
      slower but assumption-free (Section 4 of [1]).

    **Bandwidth selection.** Two heuristics are available via ``bandwidth``:

    - ``"empirical"`` (default): a piecewise rule ported from the KCI Matlab
      reference implementation [3], ``length_scale = w / sqrt(d)``.
    - ``"median"``: the median pairwise-distance heuristic (Section 2.3 of [4]),
      ``length_scale = median(||x_i - x_j||)`` — more adaptive to data scale.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataset containing the variables to test.

    kernel_X : sklearn.gaussian_process.kernels.Kernel or None, default=None
        Kernel for :math:`X`. If ``None``, an RBF kernel is built automatically
        using the heuristic specified by ``bandwidth``.

    kernel_Y : sklearn.gaussian_process.kernels.Kernel or None, default=None
        Kernel for :math:`Y`. Same behaviour as ``kernel_X``.

    bandwidth : {"empirical", "median"}, default="empirical"
        Bandwidth selection heuristic used when ``kernel_X`` or ``kernel_Y``
        is ``None``.

    null_dist : {"gamma", "permutation"}, default="gamma"
        Method used to compute the p-value under :math:`H_0`.

    n_permutations : int, default=500
        Number of permutations. Only used when ``null_dist="permutation"``.

    Attributes
    ----------
    statistic_ : float
        Observed HSIC statistic :math:`T = \operatorname{Tr}(\tilde{K}_X \tilde{K}_Y)`.

    p_value_ : float
        P-value computed via the method specified by ``null_dist``.

    References
    ----------
    .. [1] Gretton, A., Bousquet, O., Smola, A., & Schölkopf, B. (2005).
        Measuring Statistical Dependence with Hilbert-Schmidt Norms.
        ALT 2005. https://doi.org/10.1007/11564089_7
    .. [2] Gretton, A., Fukumizu, K., Teo, C. H., Song, L., Schölkopf, B., & Smola, A. J. (2007).
        A Kernel Statistical Test of Independence.
        NeurIPS 2007. https://papers.nips.cc/paper/3201
    .. [3] Zhang, K., Peters, J., Janzing, D., & Schölkopf, B. (2011).
        Kernel-based Conditional Independence Test and Application in Causal Discovery.
        UAI 2011. https://arxiv.org/abs/1202.3775
    .. [4] Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. (2012).
        A Kernel Two-Sample Test. JMLR, 13, 723–773.

    Examples
    --------
    Test marginal independence between two variables:

    >>> import numpy as np
    >>> import pandas as pd
    >>> from pgmpy.ci_tests import HSIC
    >>> rng = np.random.default_rng(seed=42)
    >>> data = pd.DataFrame(rng.standard_normal((300, 3)), columns=["X", "Y", "Z"])
    >>> test = HSIC(data=data)
    >>> test("X", "Y", [], significance_level=0.05)
    True

    Use the median heuristic for bandwidth:

    >>> test = HSIC(data=data, bandwidth="median")
    >>> test("X", "Y", [], significance_level=0.05)
    True

    Use a permutation-based p-value:

    >>> test = HSIC(data=data, null_dist="permutation", n_permutations=200)
    >>> test("X", "Y", [], significance_level=0.05)
    True

    Pass a custom RBF kernel:

    >>> from sklearn.gaussian_process.kernels import RBF
    >>> test = HSIC(data=data, kernel_X=RBF(length_scale=2.0))
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
    ):
        self.data = data
        self.kernel_X = kernel_X
        self.kernel_Y = kernel_Y
        self.bandwidth = bandwidth
        self.null_dist = null_dist
        self.n_permutations = n_permutations
        super().__init__()

    def _empirical_width(self, X: np.ndarray) -> float:
        """Piecewise RBF length_scale from the KCI Matlab reference [3]. length_scale = w / sqrt(d)."""
        n = X.shape[0]
        if n < 200:
            width = 0.8
        elif n < 1200:
            width = 0.5
        else:
            width = 0.3
        return width / np.sqrt(X.shape[1])

    def _median_width(self, X: np.ndarray) -> float:
        """Median heuristic matching causal-learn: length_scale = sqrt(2) * median(euclidean_dist)."""
        median_dist = np.median(pdist(X, metric="euclidean"))
        return float(np.sqrt(2.0) * median_dist) if median_dist > 0 else 1.0

    def _get_length_scale(self, X: np.ndarray) -> float:
        """Dispatch to the bandwidth heuristic selected by ``self.bandwidth``."""
        if self.bandwidth == "median":
            return self._median_width(X)
        return self._empirical_width(X)

    def _center_kernel(self, K: np.ndarray) -> np.ndarray:
        """Return doubly-centred kernel H @ K @ H, computed in O(n²) without forming H."""
        K_colsums = K.sum(axis=0)
        K_allsum = K_colsums.sum()
        n = K.shape[0]
        return K - (K_colsums[None, :] + K_colsums[:, None]) / n + (K_allsum / n**2)

    def _gamma_pvalue(self, test_stat: float, mean: float, var: float) -> float:
        """P-value via Gamma approximation (Proposition 6(i) of [2]). Returns 1.0 when degenerate."""
        if var <= 0 or mean <= 0:
            return 1.0
        k = mean**2 / var
        theta = var / mean
        return float(1.0 - stats.gamma.cdf(test_stat, a=k, scale=theta))

    def _permutation_pvalue(
        self,
        test_stat: float,
        Kxc: np.ndarray,
        y: np.ndarray,
        kernel_y: Kernel,
    ) -> float:
        """Empirical p-value via row-permutation of Y (Section 4 of [1])."""
        rng = np.random.default_rng()
        n = y.shape[0]
        count = sum(
            float(np.sum(Kxc * self._center_kernel(kernel_y(y[rng.permutation(n)])))) >= test_stat
            for _ in range(self.n_permutations)
        )
        return count / self.n_permutations

    def run_test(
        self,
        X: str,
        Y: str,
        Z: list,
    ):
        r"""
        Compute HSIC statistic and p-value for :math:`X \perp Y`.

        HSIC is a marginal independence test; ``Z`` must be empty.
        For conditional independence testing use :class:`KCI` instead.

        Parameters
        ----------
        X : str
            Column name of the first variable.
        Y : str
            Column name of the second variable.
        Z : list
            Must be an empty list. HSIC does not support conditioning.

        Returns
        -------
        statistic : float
            Observed HSIC statistic :math:`T = \operatorname{Tr}(\tilde{K}_X \tilde{K}_Y)`.
        p_value : float
            P-value from the method specified by ``self.null_dist``.

        Raises
        ------
        ValueError
            If ``Z`` is non-empty.
        """
        if len(Z) > 0:
            raise ValueError(
                "HSIC is a marginal independence test and does not support a "
                "conditioning set Z. Use KCI for conditional independence testing."
            )

        data = self.data
        x = stats.zscore(data.loc[:, X].to_numpy().reshape(-1, 1).astype(float), ddof=1, axis=0)
        y = stats.zscore(data.loc[:, Y].to_numpy().reshape(-1, 1).astype(float), ddof=1, axis=0)
        x = np.nan_to_num(x, nan=0.0)
        y = np.nan_to_num(y, nan=0.0)

        n = x.shape[0]

        kernel_x = self.kernel_X if self.kernel_X is not None else RBF(length_scale=self._get_length_scale(x))
        kernel_y = self.kernel_Y if self.kernel_Y is not None else RBF(length_scale=self._get_length_scale(y))

        Kxc = self._center_kernel(kernel_x(x))
        Kyc = self._center_kernel(kernel_y(y))
        test_stat = float(np.sum(Kxc * Kyc))

        if self.null_dist == "permutation":
            p_value = self._permutation_pvalue(test_stat, Kxc, y, kernel_y)
        else:
            mean = np.trace(Kxc) * np.trace(Kyc) / n
            var = 2.0 * np.sum(Kxc**2) * np.sum(Kyc**2) / n**2
            p_value = self._gamma_pvalue(test_stat, mean, var)

        self.statistic_ = test_stat
        self.p_value_ = p_value
        return self.statistic_, self.p_value_
