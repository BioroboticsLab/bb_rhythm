import scipy
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import PolynomialFeatures
from statsmodels import api as sm
import statsmodels
from statsmodels.formula import api as smf
from statsmodels.tsa.stattools import adfuller
import numpy as np
import pandas as pd


def calculate_regression(
    df,
    x_column,
    y_column,
    type="linear",
    printing=True,
    degree=3,
    epsilon=1.35,
    alpha=0.0001,
):
    # define predictor and response variables
    y = df[y_column]
    x = df[x_column]

    # add constant to predictor variables
    X = sm.add_constant(x)

    # fit linear regression model
    if type == "linear":
        fit = sm.OLS(y, X).fit()

    # fit weighted least squares regression model
    elif type == "weighted_linear":
        fit = sm.OLS(y, X).fit()

        # define weights to use
        wt = (
            1
            / smf.ols("fit.resid.abs() ~ fit.fittedvalues", data=df).fit().fittedvalues
            ** 2
        )

        # fit weighted least squares regression model
        fit_poly = sm.WLS(y, X, weights=wt).fit()
        fit = fit_poly

    elif type == "polynomial":
        # define polynomial x values
        polynomial_features = PolynomialFeatures(degree=degree)
        xp = polynomial_features.fit_transform(X)

        # calculate fit
        fit = sm.OLS(y, xp).fit()

    elif type == "poisson":
        fit = sm.GLM(y, X, family=sm.families.Poisson()).fit()

    elif type == "huber":
        fit = HuberRegressor(epsilon=epsilon, alpha=alpha).fit(X, y)

    elif type == "log":
        fit = sm.Logit(y, X).fit()

    else:
        raise NotImplementedError

    # view model summary
    if printing:
        print(fit.summary())

    return fit


def test_normally_distributed_bins(
    df, change_type="vel_change_bee_focal", printing=True
):
    # test for each bin if normally distributed
    group_type1 = "bins_non_focal"
    group_type2 = "bins_focal"
    test_results_normal = {}
    for name, group in df[[group_type1, group_type2, change_type]].groupby(
        [group_type1, group_type2]
    ):
        test_results_normal[name] = test_bin_normally_distributed(
            change_type, group, name, printing
        )
    return test_results_normal


def test_for_equal_variance_in_bins(
    df, change_type="vel_change_bee_focal", printing=True
):
    # get samples from bins
    group_type1 = "bins_non_focal"
    group_type2 = "bins_focal"
    bin_samples = [
        group[change_type].to_numpy()
        for name, group in df[[group_type1, group_type2, change_type]].groupby(
            [group_type1, group_type2]
        )
    ]

    # test if all samples have same variance
    test_result_variance = test_bins_have_equal_variance(printing, bin_samples)
    return test_result_variance


def test_for_comparison_bins(
    df_null,
    df_interaction,
    test_func,
    args=None,
    change_type="vel_change_bee_focal",
    printing=True,
):
    # iterate through bins and test
    group_type1 = "bins_non_focal"
    group_type2 = "bins_focal"
    test_results_comparison = {}
    for (name_null, group_null), (name_interaction, group_interaction) in zip(
        df_null[[group_type1, group_type2, change_type]].groupby(
            [group_type1, group_type2]
        ),
        df_interaction[[group_type1, group_type2, change_type]].groupby(
            [group_type1, group_type2]
        ),
    ):
        samples = [
            group_null[change_type].to_numpy(),
            group_interaction[change_type].to_numpy(),
        ]
        if printing:
            print((name_null, name_interaction))
        test_result = test_func(samples, *args)
        test_results_comparison[(name_null, name_interaction)] = test_result
    return test_results_comparison


def test_bins_have_equal_variance(samples, printing=True, center="mean"):
    test_result_variance = scipy.stats.levene(*samples, center=center)
    if printing:
        print("equal variance test levene: \n")
        print("statistic: %s \n" % str(test_result_variance.statistic))
        print("p_value: %s \n" % str(test_result_variance.pvalue))
    return test_result_variance


def test_bins_have_unequal_mean(samples, printing=True, equal_var=False):
    test_result_t = scipy.stats.ttest_ind(*samples, equal_var=equal_var)
    if printing:
        print("t-test means: \n")
        print("statistic: %s \n" % str(test_result_t.statistic))
        print("p_value: %s \n" % str(test_result_t.pvalue))
    return test_result_t


def test_bin_normally_distributed(change_type, group, name, printing):
    dist_args = scipy.stats.norm.fit(group[change_type].to_numpy())
    norm_distribution = scipy.stats.norm(*dist_args).cdf
    print(norm_distribution)
    test_result_normal = scipy.stats.kstest(
        rvs=group[change_type].to_numpy(), cdf=norm_distribution
    )
    if printing:
        print(name)
        print("\n")
        print(group.describe())
        print("\n")
        print("normally distributed test 'normaltest': \n")
        print("statistic: %s \n" % str(test_result_normal.statistic))
        print("p_value: %s \n" % str(test_result_normal.pvalue))
        print("\n\n")
    return test_result_normal


def time_lagged_cross_correlation(p, q):
    p = p.to_numpy()
    q = q.to_numpy()
    p = (p - np.mean(p)) / (np.std(p) * len(p))
    q = (q - np.mean(q)) / (np.std(q))
    c = scipy.signal.correlate(p, q, "full")
    return c


def apply_time_lagged_cross_correlation_to_df(df, y_variable="velocity"):
    df.velocity.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna(inplace=True)
    if len(df) == 0:
        print("df contains only nans: %s" % str(df))
        return None
    try:
        p_value_velocity = adfuller(df[y_variable])[1]
    except (ValueError, statsmodels.tools.sm_exceptions.MissingDataError):
        p_value_velocity = np.nan
    cross_correlation_dfs = pd.DataFrame(
        columns=["lags", "parameter", "ccr", "adfuller", "adfuller_v"]
    )
    for column in df.drop(columns=[y_variable]).columns:
        cross_correlation_df = pd.DataFrame(
            columns=["lags", "parameter", "ccr", "adfuller", "adfuller_v"]
        )
        df_subset = df[[column, y_variable]]
        try:
            p_value = adfuller(
                df_subset[column].replace({np.inf: np.nan, -np.inf: np.nan}).dropna()
            )[1]
        except (ValueError, statsmodels.tools.sm_exceptions.MissingDataError):
            p_value = np.nan
        try:
            cross_correlation_df["ccr"] = time_lagged_cross_correlation(
                df_subset[y_variable],
                df_subset[column]
                .replace(np.inf, np.nan)
                .replace(-np.inf, np.nan)
                .dropna(),
            )
        except ValueError:
            continue
        lags = scipy.signal.correlation_lags(
            len(df_subset[y_variable]),
            len(
                df_subset[column]
                .replace(np.inf, np.nan)
                .replace(-np.inf, np.nan)
                .dropna()
            ),
        )
        cross_correlation_df["lags"] = lags
        cross_correlation_df["parameter"] = len(lags) * [column]
        cross_correlation_df["adfuller"] = len(lags) * [p_value]
        cross_correlation_df["adfuller_v"] = len(lags) * [p_value_velocity]
        if len(cross_correlation_df) > 0:
            cross_correlation_dfs = pd.concat(
                [cross_correlation_dfs, cross_correlation_df]
            )
    return cross_correlation_dfs
