import os

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from pgmpy.ci_tests import PillaiTrace
from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.models import LinearGaussianBayesianNetwork

skip_gh_actions = pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true",
    reason="Skipping residual tests on GitHub Actions.",
)


@pytest.fixture
def pillai_data():
    model_indep = LinearGaussianBayesianNetwork(
        [
            ("Z1", "X"),
            ("Z2", "X"),
            ("Z3", "X"),
            ("Z1", "Y"),
            ("Z2", "Y"),
            ("Z3", "Y"),
        ]
    )
    cpd_z1 = LinearGaussianCPD("Z1", [0], 1)
    cpd_z2 = LinearGaussianCPD("Z2", [0], 1)
    cpd_z3 = LinearGaussianCPD("Z3", [0], 1)
    cpd_x = LinearGaussianCPD("X", [0, 0.5, 0.5, 0.5], 1, ["Z1", "Z2", "Z3"])
    cpd_y_indep = LinearGaussianCPD("Y", [0, 0.5, 0.5, 0.5], 1, ["Z1", "Z2", "Z3"])
    model_indep.add_cpds(cpd_z1, cpd_z2, cpd_z3, cpd_x, cpd_y_indep)
    df_indep = model_indep.simulate(n_samples=1000, seed=42)

    df_indep_cont_cont = df_indep.copy()
    df_indep_cont_cont.Z2 = pd.cut(
        df_indep_cont_cont.Z2,
        bins=4,
        ordered=False,
        labels=["z21", "z22", "z23", "z24"],
    ).cat.codes

    df_indep_cat_cont = df_indep_cont_cont.copy()
    df_indep_cat_cont.X = pd.cut(
        df_indep_cat_cont.X,
        bins=4,
        ordered=False,
        labels=["x1", "x2", "x3", "x4"],
    ).cat.codes

    df_indep_cat_cat = df_indep_cont_cont.copy()
    df_indep_cat_cat.X = pd.cut(
        df_indep_cat_cat.X,
        bins=4,
        ordered=False,
        labels=["x1", "x2", "x3", "x4"],
    ).cat.codes
    df_indep_cat_cat.Y = pd.cut(
        df_indep_cat_cat.Y,
        bins=4,
        ordered=False,
        labels=["y1", "y2", "y3", "y4"],
    ).cat.codes

    df_indep_ord_cont = df_indep_cont_cont.copy()
    df_indep_ord_cont.X = pd.cut(df_indep_ord_cont.X, bins=4).cat.codes

    model_dep = LinearGaussianBayesianNetwork(
        [
            ("Z1", "X"),
            ("Z2", "X"),
            ("Z3", "X"),
            ("Z1", "Y"),
            ("Z2", "Y"),
            ("Z3", "Y"),
            ("X", "Y"),
        ]
    )
    cpd_y_dep = LinearGaussianCPD("Y", [0, 0.5, 0.5, 0.5, 0.5], 1, ["Z1", "Z2", "Z3", "X"])
    model_dep.add_cpds(cpd_z1, cpd_z2, cpd_z3, cpd_x, cpd_y_dep)
    df_dep = model_dep.simulate(n_samples=1000, seed=42)

    df_dep_cont_cont = df_dep.copy()
    df_dep_cont_cont.Z2 = pd.cut(
        df_dep_cont_cont.Z2,
        bins=4,
        ordered=False,
        labels=["z21", "z22", "z23", "z24"],
    ).cat.codes

    df_dep_cat_cont = df_dep_cont_cont.copy()
    df_dep_cat_cont.X = pd.cut(
        df_dep_cat_cont.X,
        bins=4,
        ordered=False,
        labels=["x1", "x2", "x3", "x4"],
    ).cat.codes

    df_dep_cat_cat = df_dep_cont_cont.copy()
    df_dep_cat_cat.X = pd.cut(
        df_dep_cat_cat.X,
        bins=4,
        ordered=False,
        labels=["x1", "x2", "x3", "x4"],
    ).cat.codes
    df_dep_cat_cat.Y = pd.cut(
        df_dep_cat_cat.Y,
        bins=4,
        ordered=False,
        labels=["y1", "y2", "y3", "y4"],
    ).cat.codes

    df_dep_ord_cont = df_dep_cont_cont.copy()
    df_dep_ord_cont.X = pd.cut(df_dep_ord_cont.X, bins=4).cat.codes

    return {
        "indep": [df_indep, df_indep_cont_cont, df_indep_cat_cont, df_indep_cat_cat, df_indep_ord_cont],
        "dep": [df_dep, df_dep_cont_cont, df_dep_cat_cont, df_dep_cat_cat, df_dep_ord_cont],
    }


@skip_gh_actions
def test_pillai_no_cond(pillai_data):
    expected_coefs = [0.1572, 0.1572, 0.1517, 0.1212, 0.1517]
    expected_pvalues = [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]

    computed_coefs = []
    computed_pvalues = []
    for df in pillai_data["indep"]:
        test = PillaiTrace(data=df)
        test("X", "Y", [])
        computed_coefs.append(test.statistic_)
        computed_pvalues.append(test.p_value_)

    assert np.allclose(computed_coefs, expected_coefs, atol=1e-4)
    assert np.allclose(computed_pvalues, expected_pvalues, atol=1e-4)


@skip_gh_actions
def test_pillai_indep(pillai_data):
    expected_coefs = [0.0016, 0.0001, 0.0004, 0.0003, 0.0004]
    expected_pvalues = [0.2125, 0.7537, 0.5118, 0.5879, 0.5118]

    computed_coefs = []
    computed_pvalues = []
    for df in pillai_data["indep"]:
        test = PillaiTrace(data=df)
        test("X", "Y", ["Z1", "Z2", "Z3"])
        computed_coefs.append(test.statistic_)
        computed_pvalues.append(test.p_value_)

    assert np.allclose(computed_coefs, expected_coefs, atol=1e-4)
    assert np.allclose(computed_pvalues, expected_pvalues, atol=1e-4)


@skip_gh_actions
def test_pillai_dependent(pillai_data):
    expected_coefs = [0.1700, 0.2187, 0.1649, 0.1317, 0.1649]
    expected_pvalues = [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]

    computed_coefs = []
    computed_pvalues = []
    for df in pillai_data["dep"]:
        test = PillaiTrace(data=df)
        test("X", "Y", ["Z1", "Z2", "Z3"])
        computed_coefs.append(test.statistic_)
        computed_pvalues.append(test.p_value_)

    assert np.allclose(computed_coefs, expected_coefs, atol=1e-4)
    assert np.allclose(computed_pvalues, expected_pvalues, atol=1e-4)


def test_pillai_tests_approx(pillai_data):
    computed_coefs = []
    computed_pvalues = []
    for df in pillai_data["indep"]:
        test = PillaiTrace(data=df)
        test("X", "Y", [])
        computed_coefs.append(test.statistic_)
        computed_pvalues.append(test.p_value_)

    assert np.all(np.array(computed_coefs) >= 0.1)
    assert np.all(np.array(computed_pvalues) <= 0.05)

    computed_coefs = []
    computed_pvalues = []
    for df in pillai_data["indep"]:
        test = PillaiTrace(data=df)
        test("X", "Y", ["Z1", "Z2", "Z3"])
        computed_coefs.append(test.statistic_)
        computed_pvalues.append(test.p_value_)

    assert np.all(np.array(computed_coefs) <= 0.1)
    assert np.all(np.array(computed_pvalues) >= 0.05)

    computed_coefs = []
    computed_pvalues = []
    for df in pillai_data["dep"]:
        test = PillaiTrace(data=df)
        test("X", "Y", ["Z1", "Z2", "Z3"])
        computed_coefs.append(test.statistic_)
        computed_pvalues.append(test.p_value_)

    assert np.all(np.array(computed_coefs) >= 0.05)
    assert np.all(np.array(computed_pvalues) <= 0.05)


def test_pillai_linear_regression_estimator():
    rng = np.random.default_rng(0)
    n = 2000
    Z = rng.normal(size=(n, 2))
    X = Z @ [0.5, 0.5] + rng.normal(size=n)
    Y_indep = Z @ [0.5, 0.5] + rng.normal(size=n)
    Y_dep = Z @ [0.5, 0.5] + 0.8 * X + rng.normal(size=n)
    df_indep = pd.DataFrame({"X": X, "Y": Y_indep, "Z1": Z[:, 0], "Z2": Z[:, 1]})
    df_dep = pd.DataFrame({"X": X, "Y": Y_dep, "Z1": Z[:, 0], "Z2": Z[:, 1]})

    test_indep = PillaiTrace(data=df_indep, estimator=LinearRegression())
    result_indep = test_indep("X", "Y", ["Z1", "Z2"])
    assert isinstance(result_indep, (bool, np.bool_))
    assert test_indep.p_value_ > 0.05

    test_dep = PillaiTrace(data=df_dep, estimator=LinearRegression())
    result_dep = test_dep("X", "Y", ["Z1", "Z2"])
    assert isinstance(result_dep, (bool, np.bool_))
    assert test_dep.p_value_ < 0.05


def test_pillai_linear_regression_no_predict_proba():
    rng = np.random.default_rng(0)
    n = 200
    Z = rng.normal(size=n)
    X = pd.Categorical(rng.choice(["a", "b", "c"], size=n))
    Y = rng.normal(size=n)
    df = pd.DataFrame({"X": X, "Y": Y, "Z": Z})

    test = PillaiTrace(data=df, estimator=LinearRegression())
    with pytest.raises(ValueError, match="predict_proba"):
        test("X", "Y", ["Z"])
