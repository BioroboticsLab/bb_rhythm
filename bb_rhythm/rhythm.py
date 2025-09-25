import numpy as np
import os
import pandas as pd
import datetime
import pytz
import scipy
import statsmodels.formula.api as smf
import statsmodels.sandbox.stats.runs
import scipy.stats as stats
from statsmodels.regression.linear_model import RegressionResults
import statsmodels.stats.stattools
from statsmodels.tsa.stattools import adfuller
from astropy.timeseries import LombScargle

import bb_behavior.db.base
import bb_circadian.lombscargle
import bb_behavior.db

from . import time, plotting, utils


def fit_cosinor(X, Y, period=24 * 60 * 60, cov_type='HAC'):
    data = pd.DataFrame()
    data["x"] = X
    data["y"] = Y
    data= pd.concat([data] * 3, ignore_index=True)
    frequency = 2.0 * np.pi * 1 / period
    data["beta_x"] = np.cos((data.x / period) * 2.0 * np.pi)
    data["gamma_x"] = np.sin((data.x / period) * 2.0 * np.pi)
    trigonometric_regression_model = smf.glm(formula="y ~ beta_x + gamma_x", data=data, family=sm.genmod.families.family.Gamma())
    if cov_type == 'HAC': # adjust covariance https://www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.OLSResults.get_robustcov_results.html
        try:
            cov_kwds = {"maxlags": int(4 * (len(data) / 100) ** (2 / 9))}
            fit = trigonometric_regression_model.fit(cov_type='HAC', cov_kwds=cov_kwds)
        except Exception:
            fit = trigonometric_regression_model.fit()
    else:
        fit = trigonometric_regression_model.fit()
    return fit

def spectral_analysis(X, Y, min_frequency=1 / 72, max_frequency=1):
    """
    Args:
        X: time series
        Y: velocities
    """
    ts = (X - X[0]) / 3600  # time in hours
    ls = LombScargle(ts, Y)
    frequency, power = ls.autopower(minimum_frequency=min_frequency, maximum_frequency=max_frequency)

    day_frequency = (1 / 24)
    max_power_idx = np.argmax(power)
    max_frequency = frequency[max_power_idx]
    max_power = power[max_power_idx]
    circadian_power = ls.power(day_frequency)

    return dict(max_power=max_power, max_frequency=max_frequency,
                circadian_power=circadian_power, max_frequency_h=1 / max_frequency)


def derive_cosine_parameter_from_cosinor(cosinor_fit):
    """

    :param cosinor_fit:
    :return:
    """
    mesor = cosinor_fit.params[0]
    amplitude = (cosinor_fit.params.beta_x ** 2 + cosinor_fit.params.gamma_x ** 2) ** (
            1 / 2
    )
    # checking sign of beta and gamma and calculate phase accordingly
    # derived by https://rdrr.io/cran/card/src/R/cosinor-fit.R
    sb = np.sign(cosinor_fit.params.beta_x)
    sg = np.sign(cosinor_fit.params.gamma_x)
    theta = np.arctan(np.abs(cosinor_fit.params.gamma_x / cosinor_fit.params.beta_x))
    if ((sb == 1) | (sb == 0)) & (sg == 1):  # both +
        acrophase = -theta
    elif (sb == -1) & ((sg == 1) | (sg == 0)):  # - and +
        acrophase = theta - np.pi
    elif ((sb == -1) | (sb == 0)) & (sg == -1):  # - and -
        acrophase = -theta - np.pi
    else:  # (sb == 1 & (sg == -1 | sg == 0)) # + and -
        acrophase = theta - (2 * np.pi)

    # shift phase to first time interval
    acrophase %= 2 * np.pi
    if acrophase > np.pi:
        acrophase -= 2 * np.pi
    elif acrophase < -np.pi:
        acrophase += 2 * np.pi
    return mesor, amplitude, acrophase


def get_significance_values_cosinor(mesor, amplitude, acrophase, cosinor_fit):
    """

    :param mesor:
    :param amplitude:
    :param acrophase:
    :param cosinor_fit:
    :return:
    """
    # covariance matrix
    # subset for amplitude and acrophase
    indVmat = cosinor_fit.cov_params().loc[["beta_x", "gamma_x"], ["beta_x", "gamma_x"]]

    beta_r = cosinor_fit.params.beta_x
    beta_s = cosinor_fit.params.gamma_x
    a_r = (beta_r ** 2 + beta_s ** 2) ** (-0.5) * beta_r
    a_s = (beta_r ** 2 + beta_s ** 2) ** (-0.5) * beta_s
    b_r = (1 / (1 + (beta_s ** 2 / beta_r ** 2))) * (-beta_s / beta_r ** 2)
    b_s = (1 / (1 + (beta_s ** 2 / beta_r ** 2))) * (1 / beta_r)

    jac = np.array([[a_r, a_s], [b_r, b_s]])
    cov_trans = np.dot(np.dot(jac, indVmat), np.transpose(jac))
    se_trans_only = np.sqrt(np.diag(cov_trans))
    students_t = abs(scipy.stats.norm.ppf(0.05 / 2))

    coef_trans = np.array([mesor, amplitude, acrophase])
    se_trans = np.concatenate(
        (
            np.sqrt(
                np.diag(cosinor_fit.cov_params().loc[["Intercept"], ["Intercept"]])
            ),
            se_trans_only,
        )
    )

    lower_CI_trans = coef_trans - np.abs(students_t * se_trans)
    upper_CI_trans = coef_trans + np.abs(students_t * se_trans)
    p_values_trans = 2 * scipy.stats.norm.cdf(-np.abs(coef_trans / se_trans))
    return zip(lower_CI_trans, upper_CI_trans), p_values_trans


def fit_cosinor_per_bee(timeseries=None, velocities=None, period=24 * 60 * 60):
    """

    :param timeseries:
    :param velocities:
    :param period:
    :return:
    """
    # p_value alpha error correction
    X, Y = timeseries, velocities

    # make linear regression with least squares for cosinor fit
    cosinor_fit = fit_cosinor(X, Y, period=period)

    # get parameter from model
    mesor, amplitude, acrophase = derive_cosine_parameter_from_cosinor(cosinor_fit)

    # statistics according to Cornelissen (eqs (8) - (9))
    # Sum of squares due to regression model = "MSS is the sum of squared differences between the estimated values based on the fitted model and the arithmetic mean."
    # Residual sum of squared = "RSS is the sum of squared differences between the data and the estimated values from the fitted model"

    # F test for good/significant fit/ model -> 9 cornelliesen
    # if p <= alpha -> significant rhythm
    p = cosinor_fit.wald_test("gamma_x = 0, beta_x = 0").pvalue

    # p for amplitude
    # z test
    # Confidence Intervals for parameters
    (ci_mesor, ci_amplitude, ci_acrophase), (
        p_mesor,
        p_amplitude,
        p_acrophase,
    ) = get_significance_values_cosinor(mesor, amplitude, acrophase, cosinor_fit)

    # 1 - statistics of Goodness Of Fit according to Cornelissen (eqs (14) - (15))
    RSS = np.sum(cosinor_fit.resid_pearson ** 2)
    X_periodic = np.round_(X % period, 2)
    X_unique = np.unique(X_periodic)
    m = len(X_unique)
    SSPE = 0
    for x in X_unique:
        Y_i_avg = np.mean(Y[X_periodic == x])
        SSPE += sum((Y[X_periodic == x] - Y_i_avg) ** 2)
    SSLOF = RSS - SSPE

    # statistics of Goodness Of Fit according to Cornelissen (eqs (14) - (15))
    F = (SSLOF / (m - 3)) / (SSPE / (cosinor_fit.nobs - m))
    p_reject = 1 - scipy.stats.f.cdf(F, m - 3, cosinor_fit.nobs - m)

    # 2 - kolgomorov-smirnov test for residuals are normally distributed
    try:
        resid_distribution = scipy.stats.norm.fit(cosinor_fit.resid_pearson.dropna())
        p_ks = scipy.stats.kstest(
            cosinor_fit.resid_pearson.dropna(), cdf=scipy.stats.norm(*resid_distribution).cdf
        ).pvalue
    except (TypeError, ValueError):
        p_ks = np.nan

    # 3 - F = (N - 2p - 2)r² / (1-r²) > F -> variance is homogeneous
    F_hom = (
            cosinor_fit.nobs
            * cosinor_fit.fittedvalues.sum() ** 2
            / (1 - cosinor_fit.fittedvalues.sum() ** 2)
    )
    p_hom = 1 - scipy.stats.f.cdf(F_hom, 1, cosinor_fit.nobs)

    # 4 - independence of residuals -> not treat parameters but their CI -> underestimated
    # -> if violated low-passed filter by averaging or decimation
    # durbine watson statistic
    dw = statsmodels.stats.stattools.durbin_watson(cosinor_fit.resid_pearson, axis=0)

    # runs test
    try:
        p_runs = statsmodels.sandbox.stats.runs.runstest_2samp(
            cosinor_fit.resid_pearson[cosinor_fit.resid_pearson >= 0],
            cosinor_fit.resid_pearson[cosinor_fit.resid_pearson < 0],
        )[1]
    except (TypeError, ValueError):
        p_runs = np.nan

    # r_squared
    r_squared, r_squared_adj = cosinor_fit.pseudo_rsquared(), np.nan

    # lombscargle frequency analysis
    lombscargle = spectral_analysis(X, Y)

    data = {
        "mesor": mesor,
        "amplitude": amplitude,
        "phase": acrophase,
        "p_value": p,
        "p_mesor": p_mesor,
        "p_amplitude": p_amplitude,
        "p_acrophase": p_acrophase,
        "ci_mesor": ci_mesor,
        "ci_amplitude": ci_amplitude,
        "ci_acrophase": ci_acrophase,
        "p_reject": p_reject,
        "r_squared": r_squared,
        "r_squared_adj": r_squared_adj,
        "p_ks": p_ks,
        "p_hom": p_hom,
        "dw": dw,
        "p_runs": p_runs,
        "RSS": RSS,
        "SSPE": SSPE,
        "max_power_ls": lombscargle["max_power"],  # max power (y-value) of spectral analysis
        "max_frequency_ls": lombscargle["max_frequency"],  # frequency of max power (x-value) of spectral analysis
        "circadian_power_ls": lombscargle["circadian_power"],  # power (y-value) of 24h of spectral analysis
        "max_frequency_h_ls": lombscargle["max_frequency_h"],
        # frequency in hours of max power (x-value) of spectral analysis
    }
    return data


# This is copied and modified from bb_circadian.lombscargle
def circadian_cosine(x, amplitude, phase, offset, period=24 * 60 * 60):
    frequency = 2.0 * np.pi * 1 / period
    return np.cos(x * frequency + phase) * amplitude + offset


# This is copied and modified from bb_circadian.lombscargle
def fit_circadian_cosine(X, Y, phase=0):
    """Fits a cosine wave with a circadian frequency to timestamp-value pairs with the timestamps being in second precision.

    Arguments:
        X: np.array
            Timestamps in seconds. Do not have to be sorted.
        Y: np.array
            Values for their respective timestamps.
        fix_minimum: boolean
            Whether to fix offset = amplitude so that min(f(X)) == 0.
    Returns:
        Dictionary with all information about a fit.
    """
    amplitude = 3 * np.std(Y) / (2 ** 0.5)
    phase = phase
    offset = np.mean(Y)
    initial_parameters = [amplitude, phase, offset]
    bounds = [(0, -np.inf, 0), (np.inf, np.inf, np.inf)]
    fit = scipy.optimize.curve_fit(
        circadian_cosine, X, Y, p0=initial_parameters, bounds=bounds
    )
    circadian_cosine_parameters = fit[0]
    y_predicted = circadian_cosine(X, *circadian_cosine_parameters)
    circadian_sse = np.sum((y_predicted - Y) ** 2.0)

    constant_fit, full_data = np.polynomial.polynomial.Polynomial.fit(
        X, Y, deg=0, full=True
    )
    constant_sse = full_data[0][0]

    linear_fit, full_data = np.polynomial.polynomial.Polynomial.fit(
        X, Y, deg=1, full=True
    )
    linear_sse = full_data[0][0]

    r_squared_linear = 1.0 - (circadian_sse / linear_sse)
    r_squared = 1.0 - (circadian_sse / constant_sse)

    return dict(
        parameters=circadian_cosine_parameters,
        jacobian=fit[1],
        circadian_sse=circadian_sse,
        angular_frequency=2.0 * np.pi * 1 / 60 / 60 / 24,
        linear_parameters=linear_fit.convert().coef,
        linear_sse=linear_sse,
        constant_parameters=constant_fit.convert().coef,
        constant_sse=constant_sse,
        r_squared=r_squared,
        r_squared_linear=r_squared_linear,
    )


def collect_fit_data_for_bee_date(
        bee_id, date, velocities=None, delta=datetime.timedelta(days=1, hours=12), phase=0
):
    if "offset" in velocities.columns:
        ts = velocities.offset.values
    else:
        ts = np.array([t.total_seconds() for t in velocities.datetime - date])
    v = velocities.velocity.values
    assert v.shape[0] == ts.shape[0]

    begin_dt = date - delta
    end_dt = date + delta

    bee_date_data = fit_circadian_cosine(ts, v, phase=phase)
    bee_date_data["bee_id"] = bee_id
    bee_date_data["date"] = date

    # Additionally, get night/day velocities.
    try:
        add_velocity_day_night_information(bee_date_data, velocities)
    except:
        raise
    return bee_date_data


def add_velocity_day_night_information(bee_date_data, velocities):
    """

    :param bee_date_data:
    :param velocities:
    :return:
    """
    time_index = pd.DatetimeIndex(velocities.datetime)
    daytime = time_index.indexer_between_time("9:00", "18:00")
    nighttime = time_index.indexer_between_time("21:00", "6:00")
    daytime_velocities = velocities.iloc[daytime].velocity.values
    nighttime_velocities = velocities.iloc[nighttime].velocity.values
    bee_date_data["day_mean"] = np.mean(daytime_velocities)
    bee_date_data["day_std"] = np.std(daytime_velocities)
    bee_date_data["night_mean"] = np.mean(nighttime_velocities)
    bee_date_data["night_std"] = np.std(nighttime_velocities)


def fit_circadianess_fit_per_bee_phase_variation(
        day=None, bee_id=None, from_dt=None, to_dt=None, bee_age=None, phases=None
):
    if bee_age == -1 or bee_age == 0:
        return {None: dict(error="Bee is already dead or new to colony..")}

    # fetch velocities
    velocities = bb_behavior.db.trajectory.get_bee_velocities(
        bee_id, from_dt, to_dt, confidence_threshold=0.1, max_mm_per_second=15.0
    )

    if velocities is None:
        return {None: dict(error="No velocities could be fetched..")}

    try:
        # get right data types
        day = datetime.datetime.fromisoformat(day)
        assert day.tzinfo == datetime.timezone.utc
        day = day.replace(tzinfo=pytz.UTC)

        # remove NaNs and infs
        velocities = velocities[~pd.isnull(velocities.velocity)]
        data_lst = []
        for phase in phases:
            # calculate circadianess
            data = collect_fit_data_for_bee_date(
                bee_id, day, velocities=velocities, phase=phase
            )
            if data:
                # add parameters
                data["age"] = bee_age
                data["phase_default"] = phase
                # parameters for quality of velocities
                add_velocity_quality_params(data, velocities)
                # extract from parameters of fit
                extract_parameters_from_circadian_fit(data)
                data_lst.append(data)
            else:
                assert ValueError
    except (AssertionError, ValueError, IndexError, RuntimeError):
        data_lst = [{None: dict(error="Something went wrong during the fit..")}]
    return data_lst


def fit_circadianess_fit_per_bee(
        day=None, bee_id=None, from_dt=None, to_dt=None, bee_age=None
):
    if bee_age == -1 or bee_age == 0:
        return {None: dict(error="Bee is already dead or new to colony..")}

    # fetch velocities
    velocities = bb_behavior.db.trajectory.get_bee_velocities(
        bee_id, from_dt, to_dt, confidence_threshold=0.1, max_mm_per_second=15.0
    )

    if velocities is None:
        return {None: dict(error="No velocities could be fetched..")}

    try:
        # get right data types
        day = datetime.datetime.fromisoformat(day)
        assert day.tzinfo == datetime.timezone.utc
        day = day.replace(tzinfo=pytz.UTC)

        # remove NaNs and infs
        velocities = velocities[~pd.isnull(velocities.velocity)]

        # calculate circadianess
        data = bb_circadian.lombscargle.collect_circadianess_data_for_bee_date(
            bee_id, day, velocities=velocities, n_workers=0
        )
        if data:
            # add parameters
            data["age"] = bee_age
            # parameters for quality of velocities
            add_velocity_quality_params(data, velocities)
            # extract from parameters of fit
            extract_parameters_from_circadian_fit(data)
        else:
            assert ValueError
    except (AssertionError, ValueError, IndexError, RuntimeError):
        data = {None: dict(error="Something went wrong during the fit..")}
    return data


def fit_cosinor_fit_per_bee(day=None, bee_id=None, velocities=None, bee_age=None):
    """

    :param day:
    :param bee_id:
    :param velocities:
    :param bee_age:
    :return:
    """
    # get right data types
    day = datetime.datetime.fromisoformat(day.isoformat())
    assert day.tzinfo == datetime.timezone.utc
    day = day.replace(tzinfo=pytz.UTC)

    # remove NaNs and infs
    velocities = velocities[~pd.isnull(velocities.velocity)]

    # get timeseries and velocities
    if "offset" in velocities.columns:
        ts = velocities.offset.values
    else:
        ts = np.array([t.total_seconds() for t in velocities.datetime - day])
    v = velocities.velocity.values
    assert v.shape[0] == ts.shape[0]

    # calculate circadianess
    data = fit_cosinor_per_bee(ts, v)
    if data:
        # add parameters
        data["bee_id"] = bee_id
        data["age"] = bee_age
        data["date"] = day
        # parameters for quality of velocities
        add_velocity_quality_params(data, velocities)
        # extract from parameters of fit
        add_velocity_day_night_information(data, velocities)
    return data


def extract_parameters_from_circadian_fit(data):
    data["amplitude"] = data["parameters"][0]
    data["phase"] = data["parameters"][1]
    data["offset"] = data["parameters"][2]


def add_velocity_quality_params(data, velocities):
    """

    :param data:
    :param velocities:
    :return:
    """
    data["n_data_points"] = len(velocities)
    data["data_point_dist_max"] = velocities["time_passed"].max()
    data["data_point_dist_min"] = velocities["time_passed"].min()
    data["data_point_dist_mean"] = velocities["time_passed"].mean()
    data["data_point_dist_median"] = velocities["time_passed"].median()


def create_agg_circadian_df(circadianess_df, column="age_bins", agg_func="mean"):
    return (
        circadianess_df.pivot_table(
            index=["date"],
            columns=column,
            values="well_tested_circadianess",
            aggfunc=agg_func,
        )
        .melt(ignore_index=False)
        .reset_index()
    )


def create_mean_count_circadianess_per_day_df(circadianess_df, column="age_bins"):
    circadianess_df_mean = create_agg_circadian_df(circadianess_df, column=column)
    circadianess_df_count = create_agg_circadian_df(
        circadianess_df, column=column, agg_func="count"
    )
    circadianess_df = circadianess_df_mean.rename(columns={"value": "mean"})
    circadianess_df["count"] = circadianess_df_count["value"]

    # filter counts lower than 0.05 counts out
    circadianess_df = circadianess_df[
        circadianess_df["count"] > circadianess_df["count"].quantile(q=0.05)
        ]
    return circadianess_df


def calculate_well_tested_circadianess(circadianess_df):
    circadianess_df["is_good_fit"] = (circadianess_df.goodness_of_fit > 0.1).astype(
        np.float64
    )
    circadianess_df["is_circadian"] = (circadianess_df.resampled_p_value < 0.05).astype(
        np.float64
    )
    circadianess_df["well_tested_circadianess"] = (
            circadianess_df.is_circadian * circadianess_df.is_good_fit
    )


def calculate_well_tested_circadianess_cosinor(circadianess_df):
    circadianess_df["is_good_fit"] = (
            (circadianess_df.p_reject > 0.05)
            & (circadianess_df.p_ks < 0.05)
            & (circadianess_df.p_hom > 0.05)
            & (circadianess_df.ad_fuller < 0.05)
            & (circadianess_df.dw > 0.5)
    ).astype(np.float64)
    circadianess_df["is_circadian"] = (
            (circadianess_df.p_value < 0.05) & (circadianess_df.amplitude > 0)
    ).astype(np.float64)
    circadianess_df["well_tested_circadianess"] = (
            circadianess_df.is_circadian * circadianess_df.is_good_fit
    )


def extract_fit_parameters(circadianess_df):
    # extract parameters (amplitude, phase, offset) from fit
    amplitude = []
    phase = []
    offset = []
    for p in circadianess_df["parameters"]:
        amplitude.append(p[0])
        phase.append(p[1])
        offset.append(p[2])
    circadianess_df["amplitude"] = amplitude
    circadianess_df["phase"] = phase
    circadianess_df["offset"] = offset
    return circadianess_df


def create_phase_plt_age_df(circadianess_df, phase_shift=12):
    return pd.DataFrame(
        {
            "phase_plt": (
                    (time.map_pi_time_interval_to_24h(circadianess_df["phase"]))
                    + phase_shift
            ).tolist(),
            "Age [days]": circadianess_df["Age [days]"].tolist(),
            "age": circadianess_df["age"].tolist(),
        }
    )


def add_phase_plt_to_df_cosine_fit(
        circadianess_df, fit_type="cosine", time_reference=None
):
    if fit_type == "cosine":
        time_shift = 12
    else:
        time_shift = 0
    if time_reference:
        time_shift = circadianess_df["time_reference"]
    circadianess_df["phase_plt"] = (
                                           time.map_pi_time_interval_to_24h(circadianess_df["phase"]) + time_shift
                                   ) % 24
    return circadianess_df


def add_phase_plt_to_df_cosinor(circadianess_df, period=24):
    circadianess_df["phase_plt"] = (
                                           (-period * circadianess_df["phase"] / (2 * np.pi)) + 12
                                   ) % 24
    return circadianess_df


def create_phase_per_date_df(circadianess_df):
    circadianess_df_plt = plotting.apply_three_group_age_map_for_plotting_phase(
        circadianess_df
    )

    # map time interval of [-pi, pi] to 24h
    df_phase_utc = pd.DataFrame(
        {
            "phase_plt": ((circadianess_df_plt["phase"] * (12 / np.pi)) + 12).tolist(),
            "Age [days]": circadianess_df_plt["Age [days]"].tolist(),
            "age": circadianess_df_plt["age"].tolist(),
            "date": circadianess_df_plt["date"].tolist(),
        }
    )
    # create phase per date norm dist fit
    phase_per_date_df_ls = []
    for date in df_phase_utc["date"].unique():
        date_df = df_phase_utc[df_phase_utc["date"] == date]
        # test for normality
        for age_group in date_df["Age [days]"].unique():
            # get subsets
            df_phase_utc_subset = date_df[date_df["Age [days]"] == age_group]

            # fit normal distribution
            dist_args_utc = scipy.stats.norm.fit(
                df_phase_utc_subset["phase_plt"].to_numpy()
            )
            phase_per_date_df_ls.append(
                {
                    "date": date,
                    "phase_mean": dist_args_utc[0],
                    "phase_std": dist_args_utc[1],
                    "age_group": age_group,
                }
            )
    return pd.DataFrame(phase_per_date_df_ls)


def get_overall_velocity_mean(from_dt, to_dt):
    # get alive bees
    alive_bees = list(bb_behavior.db.get_alive_bees(from_dt, to_dt))

    velocities = pd.DataFrame(columns=["velocity", "datetime"])
    # iterate through all bees
    for bee_id in alive_bees:
        # fetch velocities
        velocities = pd.concat(
            [
                velocities,
                bb_behavior.db.trajectory.get_bee_velocities(
                    bee_id,
                    from_dt,
                    to_dt,
                    confidence_threshold=0.1,
                    max_mm_per_second=15.0,
                )[["velocity", "datetime"]],
            ],
            axis=0,
        )
    velocities = velocities.groupby(["datetime"])["velocity"].mean().reset_index()
    return velocities


def get_normalized_velocities(dt_from, dt_to):
    velocities_mean = get_overall_velocity_mean(
        dt_from - datetime.timedelta(hours=6), dt_to + datetime.timedelta(hours=6)
    )[["velocity", "datetime"]]
    velocities_mean["velocity_normalized"] = (
            velocities_mean.velocity.values
            - velocities_mean.set_index("datetime")
            .rolling("12h")
            .mean()
            .reset_index()["velocity"]
            .values
    )
    return velocities_mean[
        (dt_from <= velocities_mean.datetime) & (velocities_mean.datetime < dt_to)
        ].reset_index()


def get_constant_fit(velocities):
    if "offset" in velocities.columns:
        ts = velocities.offset.values
    else:
        ts = np.array(
            [
                t.total_seconds()
                for t in velocities.datetime
                         - pd.to_datetime(velocities.datetime.dt.date, utc=True)
            ]
        )
    v = velocities.velocity.values
    assert v.shape[0] == ts.shape[0]
    constant_fit = np.polynomial.polynomial.Polynomial.fit(ts, v, deg=0)
    return constant_fit


def get_raw_phase_df(file, velocities_path):
    bee_id = int(file[:-7])
    velocities = pd.read_pickle(os.path.join(velocities_path, file))
    velocities["datetime"] = velocities["datetime"].dt.round("2min")
    velocities = velocities.groupby(["datetime"])["velocity"].mean().reset_index()
    velocities["date"] = velocities.datetime.dt.date
    time_lst = []
    velocity_lst = []
    age_lst = []
    df_max_vel = pd.DataFrame(columns=["bee_id", "datetime", "age", "velocity"])
    for name, group in velocities.groupby(["date"]):
        time_lst.append(group.datetime.iloc[group.velocity.argmax()])
        velocity_lst.append(group.velocity.max())
        age_lst.append((bee_id, name))
    age_lst = [int(age) for _, _, age in bb_behavior.db.metadata.get_bee_ages(age_lst)]
    df_max_vel = pd.concat(
        [
            df_max_vel,
            pd.DataFrame(
                {
                    "bee_id": len(age_lst) * [bee_id],
                    "datetime": time_lst,
                    "age": age_lst,
                    "velocity": velocity_lst,
                }
            ),
        ]
    )
    return df_max_vel


def create_10_min_mean_velocity_df_per_bee(
        bee_id, dt_from, dt_to, velocity_df_path=None, cursor=None
):
    """

    :param bee_id:
    :param dt_from:
    :param dt_to:
    :param velocity_df_path:
    :param cursor:
    :return:
    """
    # set dates
    delta = datetime.timedelta(days=1)
    dates = list(pd.date_range(start=dt_from, end=dt_to, tz=pytz.UTC).to_pydatetime())

    # get velocities
    velocities = utils.fetch_velocities_from_remote_or_db(
        bee_id, dt_to, dt_from, velocity_df_path
    )

    # if empty return None
    if velocities is None or velocities.empty:
        return {
            None: dict(
                error="No velocities could be fetched..%d %s %s"
                      % (int(bee_id), dates[0], dates[-1])
            )
        }
    velocities.drop(columns=["time_passed"], inplace=True)

    bee_age_lst = []

    # iterate through all dates to get bee age
    for current_dt in dates:
        from_dt = current_dt
        to_dt = current_dt + delta
        bee_age = int(
            bb_behavior.db.metadata.get_bee_ages(
                [(bee_id, from_dt.date())], cursor=cursor
            )[0][2]
        )
        # subset velocities
        current_velocities = velocities[
            (velocities["datetime"] >= from_dt) & (velocities["datetime"] < to_dt)
            ]
        bee_age_lst.extend([bee_age] * len(current_velocities))

    # add age
    velocities["age"] = bee_age_lst
    # remove -1 ages
    velocities = velocities[velocities.age != -1]
    # remove NaNs
    velocities = velocities[~pd.isnull(velocities.velocity)]

    # get every ten minutes
    velocities["time"] = velocities["datetime"].dt.round("10min")
    velocities = velocities.drop(columns=["datetime"])
    grouped_velocities = (
        velocities.groupby(["time", "age"])["velocity"].mean().reset_index()
    )
    return grouped_velocities


def create_cosinor_df_per_bee_time_period(
        bee_id, to_dt, from_dt, second=60, velocity_df_path=None
):
    """

    :param bee_id:
    :param to_dt:
    :param from_dt:
    :param second:
    :param velocity_df_path:
    :return:
    """
    # get velocities
    velocities = utils.fetch_velocities_from_remote_or_db(
        bee_id, to_dt, from_dt, velocity_df_path
    )
    if velocities is None:
        return {None: dict(error="No velocities could be fetched..")}

    # get median velocity to reduce noise and increase residual independency
    if second > 0:
        velocities["datetime"] = velocities["datetime"].dt.round("%ss" % second)
        velocities = (
            velocities.groupby(["datetime"])[["velocity", "time_passed"]]
            .median()
            .reset_index()
        )
    velocities.dropna(inplace=True)
    if len(velocities) == 0:
        return {None: dict(error="No velocities could be fetched..")}

    # test for stationarity of velocities
    p_adfuller = adfuller(velocities.velocity, regression="ct")[1]

    # iterate through dates of time interval and calculate cosinor fit
    # per day with a time window of 3 consecutive days
    dates = list(
        pd.date_range(
            start=from_dt,
            end=to_dt,
            tz=pytz.UTC,
        ).to_pydatetime()
    )
    delta = datetime.timedelta(days=1, hours=12)
    data_ls = []
    for current_dt in dates:
        try:
            # get bee age
            bee_age = int(
                bb_behavior.db.metadata.get_bee_ages([(bee_id, current_dt.date())])[0][2]
            )
            # "Bee is already dead or new to colony.."
            if bee_age == -1 or bee_age == 0:
                continue
        except Exception:
            bee_age = np.nan

        # subset velocities
        current_velocities = velocities[
            (velocities.datetime >= (current_dt - delta))
            & (velocities.datetime < (current_dt + delta))
            ]
        if len(current_velocities) == 0:
            continue

        # get circadian fit data
        data = pd.DataFrame(
            fit_cosinor_fit_per_bee(
                day=current_dt,
                bee_id=bee_id,
                velocities=current_velocities,
                bee_age=bee_age,
            )
        )
        data["ad_fuller"] = p_adfuller
        data["fit_type"] = second
        data_ls.append(data)

    # concat cosinor data to dataframe
    if len(data_ls) > 0:
        cosinor_df = pd.concat(data_ls)
    else:
        cosinor_df = {None: dict(error="No velocities could be fetched or bee is dead")}
    return cosinor_df


def create_grid_from_df(df, var, aggfunc):
    return pd.pivot_table(
        df, index="y", aggfunc=aggfunc, columns="x", values=var
    ).to_numpy()


def fit_cosinor_from_df(bee_id, day, cosinor_df, velocity_df):
    """

    :param bee_id:
    :param day:
    :param cosinor_df:
    :param velocity_df:
    :return:
    """
    cosinor_df_subset = cosinor_df[(cosinor_df.bee_id == bee_id) & (cosinor_df.date == day)]
    A = cosinor_df_subset.amplitude.values[0]
    P = cosinor_df_subset.phase.values[0]
    M = cosinor_df_subset.mesor.values[0]
    X = np.array([t.total_seconds() for t in velocity_df.datetime - day])
    Y = A * np.cos(X - P) + M
    return X, Y


def min_max_scaling(timeseries):
    """
    Scales a timeseries with range [r_min, r_max] to range [0,1].
    """

    # Get min and max of original range.
    r_min = np.nanmin(timeseries)
    r_max = np.nanmax(timeseries)

    if (r_max - r_min) != 0:
        return (timeseries - r_min) / (r_max - r_min)

    # In case of division by zero return array of zeros.
    else:
        return np.zeros(timeseries.shape)