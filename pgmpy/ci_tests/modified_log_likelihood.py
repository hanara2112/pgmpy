from .power_divergence import PowerDivergence


class ModifiedLogLikelihood(PowerDivergence):
    """
    Modified log likelihood ratio test for conditional independence.
    Tests the null hypothesis that X is independent of Y given Zs.

    Parameters
    ----------
    X: int, string, hashable object
        A variable name contained in the data set

    Y: int, string, hashable object
        A variable name contained in the data set, different from X

    Z: list (array-like)
        A list of variable names contained in the data set, different from X and Y.
        This is the separating set that (potentially) makes X and Y independent.
        Default: []

    data: pandas.DataFrame
        The dataset on which to test the independence condition.

    boolean: bool
        If boolean=True, an additional argument `significance_level` must be
        specified. If p_value of the test is greater than equal to
        `significance_level`, returns True. Otherwise returns False.
        If boolean=False, returns the chi2 and p_value of the test.

    Returns
    -------
    CI Test Results: tuple or bool
        If boolean = False, Returns a tuple (chi, p_value, dof). `chi` is the
        chi-squared test statistic. The `p_value` for the test, i.e. the
        probability of observing the computed chi-square statistic (or an even
        higher value), given the null hypothesis that X ⊥⊥ Y | Zs is True.
        If boolean = True, returns True if the p_value of the test is greater
        than `significance_level` else returns False.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> data = pd.DataFrame(
    ...     np.random.randint(0, 2, size=(50000, 4)), columns=list("ABCD")
    ... )
    >>> data["E"] = data["A"] + data["B"] + data["C"]
    >>> modified_log_likelihood(
    ...     X="A", Y="C", Z=[], data=data, boolean=True, significance_level=0.05
    ... )
    np.True_
    >>> modified_log_likelihood(
    ...     X="A", Y="B", Z=["D"], data=data, boolean=True, significance_level=0.05
    ... )
    np.True_
    >>> modified_log_likelihood(
    ...     X="A", Y="B", Z=["D", "E"], data=data, boolean=True, significance_level=0.05
    ... )
    np.False_
    """

    _tags = {
        "name": "modified_log_likelihood",
        "data_types": ("discrete",),
        "default_for": None,
        "requires_data": True,
    }

    def __init__(self, data):
        super().__init__(data=data, lambda_="mod-log-likelihood")
