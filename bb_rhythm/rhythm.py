import numpy as np
import pandas as pd
import datetime
import pytz
import scipy
import statsmodels.formula.api as smf
import statsmodels.sandbox.stats.runs
import scipy.stats as stats

import bb_circadian.lombscargle
import bb_behavior.db
from statsmodels.regression.linear_model import RegressionResults
import statsmodels.stats.stattools
from statsmodels.tsa.stattools import adfuller

from . import time, plotting


def fit_cosinor(X, Y):
    data = pd.DataFrame()
    data["x"] = X
    data["y"] = Y
    frequency = 2.0 * np.pi * 1 / 60 / 60 / 24
    data["beta_x"] = np.sin((data.x / (24 * 60 * 60)) * 2.0 * np.pi)
    data["gamma_x"] = np.cos((data.x / (24 * 60 * 60)) * 2.0 * np.pi)
    trigonometric_regression_model = smf.ols("y ~ beta_x + gamma_x", data)
    fit: RegressionResults = trigonometric_regression_model.fit()
    return fit


def derive_cosine_parameter_from_cosinor(cosinor_fit):
    mesor = cosinor_fit.params[0]
    amplitude = (cosinor_fit.params.beta_x**2 + cosinor_fit.params.gamma_x**2) ** (
        1 / 2
    )
    acrophase = np.arctan(-cosinor_fit.params.beta_x / cosinor_fit.params.gamma_x)
    return mesor, amplitude, acrophase


def get_confidence_intervals_cosinor(mesor, amplitude, acrophase, cosinor_fit):
    # covariance matrix
    # subset for amplitude and acrophase
    indVmat = cosinor_fit.cov_params().loc[["beta_x", "gamma_x"], ["beta_x", "gamma_x"]]

    beta_r = cosinor_fit.params.beta_x
    beta_s = cosinor_fit.params.gamma_x
    a_r = (beta_r**2 + beta_s**2) ** (-0.5) * beta_r
    a_s = (beta_r**2 + beta_s**2) ** (-0.5) * beta_s
    b_r = (1 / (1 + (beta_s**2 / beta_r**2))) * (-beta_s / beta_r**2)
    b_s = (1 / (1 + (beta_s**2 / beta_r**2))) * (1 / beta_r)

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
    return zip(lower_CI_trans, upper_CI_trans)


def fit_cosinor_per_bee(timeseries=None, velocities=None, period=24 * 60 * 60):
    # p_value alpha error correction
    X, Y = timeseries, velocities

    # make linear regression with least squares for cosinor fit
    cosinor_fit = fit_cosinor(X, Y)

    # get parameter from model
    mesor, amplitude, acrophase = derive_cosine_parameter_from_cosinor(cosinor_fit)

    # statistics according to Cornelissen (eqs (8) - (9))
    # Sum of squares due to regression model = "MSS is the sum of squared differences between the estimated values based on the fitted model and the arithmetic mean."
    # Residual sum of squared = "RSS is the sum of squared differences between the data and the estimated values from the fitted model"

    # F test for good/significant fit/ model -> 9 cornelliesen
    # if p <= alpha -> significant rhythm
    p = cosinor_fit.f_pvalue

    # p for amplitude
    # z test
    p_mesor, p_amplitude, p_acrophase = cosinor_fit.pvalues

    # Confidence Intervals for parameters
    ci_mesor, ci_amplitude, ci_acrophase = get_confidence_intervals_cosinor(
        mesor, amplitude, acrophase, cosinor_fit
    )

    # 1 - statistics of Goodness Of Fit according to Cornelissen (eqs (14) - (15))
    RSS = cosinor_fit.ssr
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

    # 2 - chi_square test for goodness of fit -> residuals are normally distributed
    chi_square_test_statistic, chi_p_value = scipy.stats.chisquare(
        Y, cosinor_fit.fittedvalues
    )

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
    dw = statsmodels.stats.stattools.durbin_watson(cosinor_fit.resid, axis=0)

    # test for stationarity
    p_adfuller = adfuller(Y)[1]

    # r_squared
    r_squared, r_squared_adj = cosinor_fit.rsquared, cosinor_fit.rsquared_adj

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
        "chi_p_value": chi_p_value,
        "p_hom": p_hom,
        "dw": dw,
        "p_adfuller": p_adfuller,
    }
    return data


# This is copied and modified from bb_circadian.lombscargle
def circadian_cosine(x, amplitude, phase, offset):
    frequency = 2.0 * np.pi * 1 / 60 / 60 / 24
    return np.cos(x * frequency - phase) * amplitude + offset


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
    amplitude = 3 * np.std(Y) / (2**0.5)
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


def fit_cosinor_fit_per_bee(
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

    # get right data types
    day = datetime.datetime.fromisoformat(day)
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


def add_phase_plt_to_df(circadianess_df, fit_type="cosine", time_reference=None):
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
