import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform

from ._base import _BaseCITest


class KCI(_BaseCITest):
    r"""
    Kernel-based Conditional Independence (KCI) test [1].

    When :math:`Z = \emptyset`, this reduces to the HSIC (Hilbert-Schmidt Independence
    Criterion) test for unconditional independence. When :math:`Z \neq \emptyset`, it
    performs a kernel-based conditional independence test (KCIT) by regressing out the
    effect of :math:`Z` from the kernel matrices of :math:`X` and :math:`Y`.

    Both tests use the RBF (Gaussian) kernel and compute p-values via a gamma
    approximation to the null distribution.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset in which to test the independence condition.

    kernel_X : str, default="rbf"
        Kernel for variable X. Currently only ``"rbf"`` is supported.

    kernel_Y : str, default="rbf"
        Kernel for variable Y. Currently only ``"rbf"`` is supported.

    kernel_Z : str, default="rbf"
        Kernel for conditioning variables Z. Currently only ``"rbf"`` is supported.

    width_X : float or None, default=None
        Bandwidth (sigma) for the X kernel. If ``None``, estimated from the data
        using a sample-size-based heuristic.

    width_Y : float or None, default=None
        Bandwidth (sigma) for the Y kernel. If ``None``, estimated from the data.

    width_Z : float or None, default=None
        Bandwidth (sigma) for the Z kernel. If ``None``, estimated from the data.

    regularization : float, default=1e-3
        Regularization parameter :math:`\epsilon` used in the conditional test to
        invert the kernel matrix on :math:`Z`.

    Attributes
    ----------
    statistic_ : float
        The test statistic (HSIC V-statistic). Set after calling the test.
    p_value_ : float
        The p-value for the test via gamma approximation. Set after calling the test.

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

    References
    ----------
    .. [1] Zhang, K., Peters, J., Janzing, D., and Schölkopf, B. (2011).
           Kernel-based Conditional Independence Test and Application in Causal Discovery.
           UAI 2011. https://arxiv.org/abs/1202.3775
    .. [2] Gretton, A., Fukumizu, K., Teo, C., Song, L., Schölkopf, B., and Smola, A.
           (2008). A Kernel Statistical Test of Independence. NIPS 2007.
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
        kernel_X: str = "rbf",
        kernel_Y: str = "rbf",
        kernel_Z: str = "rbf",
        width_X: float | None = None,
        width_Y: float | None = None,
        width_Z: float | None = None,
        regularization: float = 1e-3,
    ):
        self.data = data
        self.kernel_X = kernel_X
        self.kernel_Y = kernel_Y
        self.kernel_Z = kernel_Z
        self.width_X = width_X
        self.width_Y = width_Y
        self.width_Z = width_Z
        self.regularization = regularization
        super().__init__()

    @staticmethod
    def _rbf_kernel(X, width):
        """Compute the RBF (Gaussian) kernel matrix.

        Parameters
        ----------
        X : np.ndarray of shape (n, d)
            Input data.
        width : float
            Bandwidth parameter (sigma).

        Returns
        -------
        K : np.ndarray of shape (n, n)
            Kernel matrix where ``K[i, j] = exp(-||X_i - X_j||^2 / (2 * width^2))``.
        """
        dists = squareform(pdist(X, "sqeuclidean"))
        return np.exp(-dists / (2.0 * width**2))

    @staticmethod
    def _estimate_width(X, n):
        """Estimate RBF kernel bandwidth using a sample-size-based heuristic.

        Uses the heuristic from Zhang et al. (2011) / causal-learn: the bandwidth is
        set proportional to the data spread, scaled by a factor that depends on sample
        size.

        Parameters
        ----------
        X : np.ndarray of shape (n, d)
            Input data.
        n : int
            Number of samples.

        Returns
        -------
        width : float
            Estimated bandwidth (sigma).
        """
        if n < 200:
            factor = 1.2
        elif n < 1200:
            factor = 0.7
        else:
            factor = 0.4

        d = X.shape[1]
        # Median of pairwise distances as base, scaled by factor and dimension
        dists = pdist(X, "euclidean")
        median_dist = np.median(dists)

        if median_dist == 0:
            median_dist = 1.0

        return factor * median_dist / np.sqrt(d)

    @staticmethod
    def _center_kernel(K):
        """Center a kernel matrix: H @ K @ H where H = I - 1/n.

        Computed in O(n^2) without forming H explicitly.
        """
        col_mean = K.mean(axis=0)
        total_mean = col_mean.mean()
        return K - col_mean[np.newaxis, :] - col_mean[:, np.newaxis] + total_mean

    @staticmethod
    def _gamma_pvalue(test_stat, mean, var):
        """Compute p-value using gamma approximation.

        Matches the first two moments of the null distribution to a gamma distribution.

        Parameters
        ----------
        test_stat : float
            The observed test statistic.
        mean : float
            Mean of the null distribution.
        var : float
            Variance of the null distribution.

        Returns
        -------
        p_value : float
            The p-value.
        """
        if var <= 0 or mean <= 0:
            return 1.0

        k = mean**2 / var  # shape
        theta = var / mean  # scale
        return 1.0 - stats.gamma.cdf(test_stat, a=k, scale=theta)

    def _unconditional_test(self, x, y):
        """HSIC-based unconditional independence test (X _|_ Y).

        Parameters
        ----------
        x : np.ndarray of shape (n, 1)
            Data for variable X (z-scored).
        y : np.ndarray of shape (n, 1)
            Data for variable Y (z-scored).

        Returns
        -------
        statistic : float
            HSIC V-statistic.
        p_value : float
            P-value via gamma approximation.
        """
        n = x.shape[0]

        # Estimate widths if not provided
        width_x = self.width_X if self.width_X is not None else self._estimate_width(x, n)
        width_y = self.width_Y if self.width_Y is not None else self._estimate_width(y, n)

        # Compute and center kernel matrices
        Kx = self._rbf_kernel(x, width_x)
        Ky = self._rbf_kernel(y, width_y)
        Kxc = self._center_kernel(Kx)
        Kyc = self._center_kernel(Ky)

        # HSIC V-statistic: (1/n^2) * trace(Kxc @ Kyc) = (1/n^2) * sum(Kxc * Kyc)
        test_stat = np.sum(Kxc * Kyc)

        # Gamma approximation for p-value
        mean = np.trace(Kxc) * np.trace(Kyc) / n
        var = 2.0 * np.sum(Kxc**2) * np.sum(Kyc**2) / n**2

        return test_stat, self._gamma_pvalue(test_stat, mean, var)

    def _conditional_test(self, x, y, z):
        """Kernel-based conditional independence test (X _|_ Y | Z).

        Parameters
        ----------
        x : np.ndarray of shape (n, 1)
            Data for variable X (z-scored).
        y : np.ndarray of shape (n, 1)
            Data for variable Y (z-scored).
        z : np.ndarray of shape (n, d_z)
            Data for conditioning variables Z (z-scored).

        Returns
        -------
        statistic : float
            KCI test statistic.
        p_value : float
            P-value via gamma approximation.
        """
        n = x.shape[0]
        epsilon = self.regularization

        # Estimate widths if not provided
        width_x = self.width_X if self.width_X is not None else self._estimate_width(np.hstack([x, 0.5 * z]), n)
        width_y = self.width_Y if self.width_Y is not None else self._estimate_width(y, n)
        width_z = self.width_Z if self.width_Z is not None else self._estimate_width(z, n)

        # Compute kernel matrices
        # Following Zhang et al.: augment X with 0.5*Z for the X kernel
        Kx = self._rbf_kernel(np.hstack([x, 0.5 * z]), width_x)
        Ky = self._rbf_kernel(y, width_y)
        Kz = self._rbf_kernel(z, width_z)

        # Center Kx and Ky
        Kx = self._center_kernel(Kx)
        Ky = self._center_kernel(Ky)
        Kz = self._center_kernel(Kz)

        # Regression-based centering to remove effect of Z
        # Rz = epsilon * (Kz + epsilon * I)^{-1}
        Rz = epsilon * np.linalg.inv(Kz + epsilon * np.eye(n))

        # Residual kernel matrices
        KxR = Rz @ Kx @ Rz
        KyR = Rz @ Ky @ Rz

        # Test statistic
        test_stat = np.sum(KxR * KyR)

        # Eigendecomposition for gamma approximation
        # Product of eigenvectors for null distribution moments
        eigx = np.linalg.eigh(KxR)[1]
        eigy = np.linalg.eigh(KyR)[1]
        uu_prod = (eigx.T @ eigy) ** 2

        mean = np.sum(uu_prod)
        var = 2.0 * np.sum(uu_prod**2)

        return test_stat, self._gamma_pvalue(test_stat, mean, var)

    def run_test(
        self,
        X: str,
        Y: str,
        Z: list,
    ):
        """
        Run the KCI/KCIT independence test.

        Sets ``self.statistic_`` and ``self.p_value_``.

        Parameters
        ----------
        X : str
            The first variable for testing X _|_ Y | Z.
        Y : str
            The second variable for testing X _|_ Y | Z.
        Z : list
            Conditioning variables.

        Returns
        -------
        statistic : float
            The KCI test statistic.
        p_value : float
            The p-value.
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
