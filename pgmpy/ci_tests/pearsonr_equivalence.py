from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from .pearsonr import Pearsonr


class PearsonrEquivalence(Pearsonr):
    """
    Computes a two-sided level-alpha equivalent test using partial correlations.

    Tests the Null Hypothesis that the partial correlation is greater than or
    equal to `delta_threshold` (Dependence). Rejection implies Practical Independence.

    Parameters
    ----------
    data: pandas.DataFrame
        The dataset in which to test the independence condition.

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
        super().__init__(data=data)

    def test(
        self,
        X: str,
        Y: str,
        Z: Optional[list] = None,
        boolean: bool = True,
        significance_level: float = 0.05,
        **kwargs,
    ) -> Union[bool, Tuple[float, ...]]:
        """
        Perform the equivalence CI test.

        Note: For equivalence tests, independence is concluded when p_value < significance_level
        (rejecting the null of dependence), which is the OPPOSITE of standard CI tests.
        """
        if not isinstance(Z, (list, tuple)):
            raise ValueError(f"Z must be a list or tuple. Got {type(Z)}.")
        self._validate_inputs(X, Y, Z)

        self._compute_statistic(X=X, Y=Y, Z=Z, **kwargs)

        if boolean:
            # INVERTED: equivalence test returns True when p < alpha
            return self.p_value_ < significance_level
        return self.statistic_, self.p_value_

    def _compute_statistic(
        self,
        X: str,
        Y: str,
        Z: list,
        **kwargs,
    ) -> None:
        """
        Compute Pearson equivalence statistic and p-value.

        Sets ``self.statistic_`` (Fisher z-transformed partial correlation) and ``self.p_value_``.
        """
        # Step 2: Compute Partial Pearson Correlation via parent and clip to avoid infinities
        super()._compute_statistic(X, Y, Z, **kwargs)
        rho = np.clip(self.statistic_, -0.999999, 0.999999)

        # Step 3: Fisher Z-Transformation
        coeff = np.arctanh(rho)
        z_delta = np.arctanh(self.delta_threshold)

        n = self.data.shape[0]
        s = len(Z)  # Number of conditioning variables

        std_error_factor = np.sqrt(n - s - 3)

        # Step 4: TOST (Two One-Sided Tests)
        # Step 4.1: H0: rho <= -delta  vs  H1: rho > -delta
        z_score_lower = std_error_factor * (coeff + z_delta)
        p_value_lower = 1 - stats.norm.cdf(z_score_lower)

        # Step 4.2: H0: rho >= delta   vs  H1: rho < delta
        z_score_upper = std_error_factor * (coeff - z_delta)
        p_value_upper = stats.norm.cdf(z_score_upper)

        self.statistic_ = coeff
        self.p_value_ = max(p_value_lower, p_value_upper)
