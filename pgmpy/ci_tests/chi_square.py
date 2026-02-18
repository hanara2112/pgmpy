import pandas as pd

from .power_divergence import PowerDivergence


class ChiSquare(PowerDivergence):
    """
    Perform Chi-square conditional independence test.

    Tests the null hypothesis that X is independent from Y given Zs.

    Parameters
    ----------
    X: int, string, hashable object
        A variable name contained in the data set

    Y: int, string, hashable object
        A variable name contained in the data set, different from X

    Z: list, array-like
        A list of variable names contained in the data set, different from X and Y.
        This is the separating set that (potentially) makes X and Y independent.
        Default: []

    data: pandas.DataFrame
        The dataset on which to test the independence condition.

    boolean: bool
        If boolean=True, an additional argument `significance_level` must
        be specified. If p_value of the test is greater than equal to
        `significance_level`, returns True. Otherwise returns False.
        If boolean=False, returns the chi2 and p_value of the test.

    Returns
    -------
    result : bool or tuple
        If boolean=False, returns (chi, p_value, dof).
        If boolean=True, returns True if p_value > significance_level.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Chi-squared_test

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> data = pd.DataFrame(
    ...     np.random.randint(0, 2, size=(50000, 4)), columns=list("ABCD")
    ... )
    >>> data["E"] = data["A"] + data["B"] + data["C"]
    >>> chi_square(X="A", Y="C", Z=[], data=data, boolean=True, significance_level=0.05)
    np.True_
    >>> chi_square(
    ...     X="A", Y="B", Z=["D"], data=data, boolean=True, significance_level=0.05
    ... )
    np.True_
    >>> chi_square(
    ...     X="A", Y="B", Z=["D", "E"], data=data, boolean=True, significance_level=0.05
    ... )
    np.False_
    """

    _tags = {
        "name": "chi_square",
        "data_types": ("discrete",),
        "default_for": "discrete",
        "requires_data": True,
    }

    def __init__(self, data: pd.DataFrame):
        super().__init__(data=data, lambda_="pearson")
