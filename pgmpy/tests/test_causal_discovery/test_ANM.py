import pandas as pd
import pytest

from pgmpy.causal_discovery import ANM


def test_init():
    est = ANM(random_state=0)
    assert est.get_params() == {"regressor": None, "ci_test": None, "random_state": 0}


def test_fit_not_implemented():
    data = pd.DataFrame({"X": [0.0, 1.0, 2.0], "Y": [1.0, 2.0, 3.0]})
    with pytest.raises(NotImplementedError):
        ANM().fit(data)
