import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist
from sklearn.gaussian_process.kernels import RBF, Kernel

from ._base import _BaseCITest


class KCI(_BaseCITest):
    r"""
    Kernel-based Conditional Independence (KCI) test [1].

    When :math:`Z = \emptyset`, this reduces to the HSIC (Hilbert-Schmidt Independence
    Criterion) test for unconditional independence. When :math:`Z \neq \emptyset`, it
    performs a kernel-based conditional independence test (KCIT) by regressing out the
    effect of :math:`Z` from the kernel matrices of :math:`X` and :math:`Y`.

    Both tests compute p-values via a gamma approximation to the null distribution.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset in which to test the independence condition.

    kernel_X : sklearn.gaussian_process.kernels.Kernel or None, default=None
        Kernel for variable X. If ``None``, uses ``RBF`` with bandwidth estimated
        from the data via the median heuristic from Zhang et al. (2011).

    kernel_Y : sklearn.gaussian_process.kernels.Kernel or None, default=None
        Kernel for variable Y. If ``None``, uses ``RBF`` with bandwidth estimated
        from the data.

    kernel_Z : sklearn.gaussian_process.kernels.Kernel or None, default=None
        Kernel for conditioning variables Z. If ``None``, uses ``RBF`` with bandwidth
        estimated from the data.

    Attributes
    ----------
    statistic_ : float
        The test statistic (HSIC V-statistic for unconditional, KCI statistic for
        conditional). Set after calling the test.
    p_value_ : float
        The p-value for the test via gamma approximation. Set after calling the test.

    References
    ----------
    .. [1] Zhang, K., Peters, J., Janzing, D., and Schölkopf, B. (2011).
           Kernel-based Conditional Independence Test and Application in Causal Discovery.
           UAI 2011. https://arxiv.org/abs/1202.3775
    .. [2] Gretton, A., Fukumizu, K., Teo, C., Song, L., Schölkopf, B., and Smola, A.
           (2008). A Kernel Statistical Test of Independence. NIPS 2007.

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
    48.42...
    >>> test.p_value_  # doctest: +SKIP
    0.38...

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
    ):
        self.data = data
        self.kernel_X = kernel_X
        self.kernel_Y = kernel_Y
        self.kernel_Z = kernel_Z
        super().__init__()

    @staticmethod
    def _estimate_bandwidth(X):
        """Estimate RBF kernel bandwidth using the median heuristic.

        Uses the heuristic from Zhang et al. (2011) / causal-learn: the bandwidth is
        set proportional to the median pairwise distance, scaled by a factor that
        depends on sample size.

        Parameters
        ----------
        X : np.ndarray of shape (n, d)
            Input data.

        Returns
        -------
        length_scale : float
            Estimated bandwidth for use as ``RBF(length_scale=...)``.
        """
        n = X.shape[0]
        if n < 200:
            factor = 1.2
        elif n < 1200:
            factor = 0.7
        else:
            factor = 0.4

        d = X.shape[1]
        dists = pdist(X, "euclidean")
        median_dist = np.median(dists)

        if median_dist == 0:
            median_dist = 1.0

        return factor * median_dist / np.sqrt(d)

    def _get_kernel(self, kernel, X):
        """Return the kernel to use, estimating bandwidth if kernel is None."""
        if kernel is not None:
            return kernel
        length_scale = self._estimate_bandwidth(X)
        return RBF(length_scale=length_scale)

    @staticmethod
    def _center_kernel(K):
        """Center a kernel matrix: H @ K @ H where H = I - 1/n.

        Computed in O(n^2) without forming H explicitly.
        """
        n = K.shape[0]
        K_colsums = K.sum(axis=0)
        K_allsum = K_colsums.sum()
        return K - (K_colsums[None, :] + K_colsums[:, None]) / n + (K_allsum / n**2)

    @staticmethod
    def _gamma_pvalue(test_stat, mean, var):
        """Compute p-value using gamma approximation.

        Matches the first two moments of the null distribution to a gamma distribution.
        """
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

        kernel_x = self._get_kernel(self.kernel_X, x)
        kernel_y = self._get_kernel(self.kernel_Y, y)

        Kx = kernel_x(x)
        Ky = kernel_y(y)
        Kxc = self._center_kernel(Kx)
        Kyc = self._center_kernel(Ky)

        # HSIC V-statistic: sum(Kxc * Kyc)
        test_stat = np.sum(Kxc * Kyc)

        # Gamma approximation for p-value
        mean = np.trace(Kxc) * np.trace(Kyc) / n
        var = 2.0 * np.sum(Kxc**2) * np.sum(Kyc**2) / n**2

        return test_stat, self._gamma_pvalue(test_stat, mean, var)

    @staticmethod
    def _get_uu_prod(KxR, KyR, thresh=1e-5):
        """Compute the product matrix whose trace gives null distribution moments."""
        n = KxR.shape[0]

        # Symmetrize to avoid numerical asymmetry issues
        wx, vx = np.linalg.eigh(0.5 * (KxR + KxR.T))
        wy, vy = np.linalg.eigh(0.5 * (KyR + KyR.T))

        # Sort descending
        idx = np.argsort(-wx)
        idy = np.argsort(-wy)
        wx, vx = wx[idx], vx[:, idx]
        wy, vy = wy[idy], vy[:, idy]

        # Keep only significant eigenvalues
        vx = vx[:, wx > np.max(wx) * thresh]
        wx = wx[wx > np.max(wx) * thresh]
        vy = vy[:, wy > np.max(wy) * thresh]
        wy = wy[wy > np.max(wy) * thresh]

        # Scale eigenvectors by sqrt of eigenvalues
        vx = vx @ np.diag(np.sqrt(wx))
        vy = vy @ np.diag(np.sqrt(wy))

        # Form all pairwise element-wise products
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
        epsilon = 1e-3

        kernel_x = self._get_kernel(self.kernel_X, np.hstack([x, 0.5 * z]))
        kernel_y = self._get_kernel(self.kernel_Y, y)
        kernel_z = self._get_kernel(self.kernel_Z, z)

        # Compute kernel matrices
        # Following Zhang et al.: augment X with 0.5*Z for the X kernel
        Kx = kernel_x(np.hstack([x, 0.5 * z]))
        Ky = kernel_y(y)
        Kz = kernel_z(z)

        # Center all kernel matrices
        Kx = self._center_kernel(Kx)
        Ky = self._center_kernel(Ky)
        Kz = self._center_kernel(Kz)

        # Regression-based centering to remove effect of Z
        # Rz = epsilon * (Kz + epsilon * I)^{-1}   (using pseudoinverse for robustness)
        Rz = epsilon * np.linalg.pinv(Kz + epsilon * np.eye(n))

        # Residual kernel matrices
        KxR = Rz @ Kx @ Rz
        KyR = Rz @ Ky @ Rz

        # Test statistic
        test_stat = np.sum(KxR * KyR)

        # Gamma approximation via eigendecomposition
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

        # Z-score normalize
        x = data.loc[:, X].to_numpy().reshape(-1, 1).astype(float)
        y = data.loc[:, Y].to_numpy().reshape(-1, 1).astype(float)

        x = (x - x.mean()) / (x.std(ddof=1) + 1e-10)
        y = (y - y.mean()) / (y.std(ddof=1) + 1e-10)

        if len(Z) == 0:
            self.statistic_, self.p_value_ = self._unconditional_test(x, y)
        else:
            z = data.loc[:, Z].to_numpy().astype(float)
            z = (z - z.mean(axis=0)) / (z.std(axis=0, ddof=1) + 1e-10)
            self.statistic_, self.p_value_ = self._conditional_test(x, y, z)

        return self.statistic_, self.p_value_
