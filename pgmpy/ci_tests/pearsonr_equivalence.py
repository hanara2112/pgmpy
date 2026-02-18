from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ._base import _BaseCITest
from .pearsonr import Pearsonr


class PearsonrEquivalence(_BaseCITest):
    """
    Computes a two-sided level-alpha equivalent test using partial correlations.

    Tests the Null Hypothesis that the partial correlation is greater than or
    equal to `delta_threshold` (Dependence). Rejection implies Practical Independence.

    Parameters
    ----------
    X: str
        The first variable for testing the independence condition X _|_ Y | Z

    Y: str
        The second variable for testing the independence condition X _|_ Y | Z

    Z: list/array-like
        A list of conditional variable for testing the condition X _|_ Y | Z

    data: pandas.DataFrame
        The dataset in which to test the independence condition.

    boolean: bool
        If True, returns True (Independent) if p_value < significance_level.

    delta_threshold: float
        The equivalence bound (threshold for practical independence).

    Returns
    -------
    CI Test results: tuple or bool
        If boolean=True, returns True (Independent) if p-value < significance_level.
        If boolean=False, returns (Partial Correlation, p-value).

    References
    ----------
    .. [1] Malinsky, Daniel. "A cautious approach to constraint-based causal model selection." arXiv preprint
            arXiv:2404.18232 (2024).
    """

    _tags = {
        "name": "pearsonr_equivalence",
        "data_types": ("continuous",),
        "default_for": None,
        "requires_data": True,
    }

    def __init__(self, data: pd.DataFrame, delta_threshold: float = 0.1):
        self.delta_threshold = delta_threshold
        self.data = data
        self._pearsonr_test = Pearsonr(data)
        super().__init__()

    def _compute_statistic(
        self,
        X: str,
        Y: str,
        Z: list,
        **kwargs,
    ) -> Tuple[float, float]:
        """
        Compute Pearson equivalence statistic and p-value.

        Parameters
        ----------
        X : str
            The first variable for testing the independence condition X _|_ Y | Z.
        Y : str
            The second variable for testing the independence condition X _|_ Y | Z.
        Z : list
            A list of conditional variables for testing the condition X _|_ Y | Z.
        **kwargs
            Additional arguments.

        Returns
        -------
        tuple
            A tuple of (partial correlation, p-value).
        """
        data = self.data
        # Step 2: Compute Partial Pearson Correlation and clip values to avoid infinities
        rho, _ = self._pearsonr_test.test(X, Y, Z, boolean=False)
        rho = np.clip(rho, -0.999999, 0.999999)

        # Step 3: Fisher Z-Transformation
        coeff = np.arctanh(rho)
        z_delta = np.arctanh(self.delta_threshold)

        n = data.shape[0]
        s = len(Z)  # Number of conditioning variables

        std_error_factor = np.sqrt(n - s - 3)

        # Step 4: TOST (Two One-Sided Tests)
        # Step 4.1: H0: rho <= -delta  vs  H1: rho > -delta
        z_score_lower = std_error_factor * (coeff + z_delta)
        p_value_lower = 1 - stats.norm.cdf(z_score_lower)

        # Step 4.2: H0: rho >= delta   vs  H1: rho < delta
        z_score_upper = std_error_factor * (coeff - z_delta)
        p_value_upper = stats.norm.cdf(z_score_upper)

        p_value = max(p_value_lower, p_value_upper)

        return coeff, p_value
