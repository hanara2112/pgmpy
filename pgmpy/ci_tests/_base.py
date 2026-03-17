from typing import Union

import pandas as pd
from skbase.base import BaseObject
from skbase.lookup import all_objects


class _BaseCITest(BaseObject):
    """
    Base class for all Conditional Independence (CI) tests.
    Subclasses must implement `_compute_statistic`.
    """

    _tags = {
        "name": None,
        "data_types": (),
        "default_for": None,
        "requires_data": True,
    }

    def __call__(
        self,
        X: str,
        Y: str,
        Z: Union[list, tuple] = (),
        significance_level: float = 0.05,
        **kwargs,
    ):
        return self.test(
            X=X,
            Y=Y,
            Z=Z,
            significance_level=significance_level,
            **kwargs,
        )

    def test(
        self,
        X: str,
        Y: str,
        Z: Union[list, tuple] = (),
        significance_level: float = 0.05,
        **kwargs,
    ) -> bool:
        """
        Perform the conditional independence test.

        Parameters
        ----------
        X : str
            The first variable for testing the independence condition X ⊥⊥ Y | Z.
        Y : str
            The second variable for testing the independence condition X ⊥⊥ Y | Z.
        Z : list or tuple
            A list of conditional variables for testing the condition X ⊥⊥ Y | Z.
            Default is an empty tuple.
        significance_level : float, default=0.05
            The significance level for the test.
        **kwargs
            Additional arguments specific to the CI test implementation.

        Returns
        -------
        bool
            True if X ⊥⊥ Y | Z (p_value_ >= significance_level), else False.

        Raises
        ------
        ValueError
            If inputs are invalid.

        Notes
        -----
        Always sets ``self.statistic_`` and ``self.p_value_`` as side effects,
        regardless of the return value. Access these attributes to inspect raw results.
        CI test instances are not thread-safe; use a separate instance per thread
        for parallel computation.
        """

        if not isinstance(Z, (list, tuple)):
            raise ValueError(f"Z must be a list or tuple. Got {type(Z)}.")
        self._validate_inputs(X, Y, Z)

        self._compute_statistic(X=X, Y=Y, Z=Z, **kwargs)

        return self.p_value_ >= significance_level

    def _validate_inputs(self, X, Y, Z):
        if X == Y:
            raise ValueError("X and Y must be different variables.")

        if X in Z or Y in Z:
            raise ValueError(
                f"X and Y cannot appear in Z. Found {X if X in Z else Y} in Z."
            )

        if self.get_tag("requires_data"):
            if not hasattr(self, "data") or not isinstance(self.data, pd.DataFrame):
                raise ValueError(
                    f"self.data must be a pandas.DataFrame. Got {type(getattr(self, 'data', None))}."
                )

            missing = ({X, Y} | set(Z)) - set(self.data.columns)
            if missing:
                raise ValueError(f"Missing columns in data: {missing}")


def _instantiate_ci_test(cls, data=None):
    """
    Helper to instantiate a CI test class, respecting whether it requires data.
    """
    requires_data = cls.get_class_tag("requires_data", tag_value_default=True)
    if requires_data:
        if data is None:
            raise ValueError(
                f"CI test '{cls.__name__}' requires data, but data is None."
            )
        return cls(data=data)
    return cls()


def get_ci_test(test=None, data=None):
    """
    Retrieve CI test instance by name or infer default from data.
    """

    from pgmpy.utils import get_dataset_type

    if isinstance(test, _BaseCITest):
        return test

    if callable(test):
        return test

    if test is None:
        if data is None:
            raise ValueError(
                "Cannot determine CI test: both `test` and `data` are None."
            )

        var_type = get_dataset_type(data)

        tests = all_objects(
            object_types=_BaseCITest,
            package_name="pgmpy.ci_tests",
            return_names=False,
            filter_tags={"default_for": var_type},
        )

        if tests:
            return _instantiate_ci_test(tests[0], data=data)

        raise ValueError(f"No default CI test found for data type '{var_type}'.")

    if isinstance(test, str):
        tests = all_objects(
            object_types=_BaseCITest,
            package_name="pgmpy.ci_tests",
            return_names=False,
            filter_tags={"name": test.lower()},
        )

        if tests:
            return _instantiate_ci_test(tests[0], data=data)

        raise ValueError(f"Unknown CI test: {test!r}")

    raise ValueError(f"Invalid `test` argument: {test!r}")
