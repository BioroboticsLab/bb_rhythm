import numpy as np
import pandas as pd
import datetime
import pytz
import scipy

import bb_circadian.lombscargle
import bb_behavior.db

from . import time, plotting


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

    try:
        bee_date_data = fit_circadian_cosine(ts, v, phase=phase)
    except (RuntimeError, TypeError):
        return None

    bee_date_data["bee_id"] = bee_id
    bee_date_data["date"] = date

    # Additionally, get night/day velocities.
    try:
        time_index = pd.DatetimeIndex(velocities.datetime)
        daytime = time_index.indexer_between_time("9:00", "18:00")
        nighttime = time_index.indexer_between_time("21:00", "6:00")
        daytime_velocities = velocities.iloc[daytime].velocity.values
        nighttime_velocities = velocities.iloc[nighttime].velocity.values
        bee_date_data["day_mean"] = np.mean(daytime_velocities)
        bee_date_data["day_std"] = np.std(daytime_velocities)
        bee_date_data["night_mean"] = np.mean(nighttime_velocities)
        bee_date_data["night_std"] = np.std(nighttime_velocities)
    except:
        raise
    return bee_date_data


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
            else:
                assert ValueError
    except (AssertionError, ValueError, IndexError, RuntimeError):
        data = {None: dict(error="Something went wrong during the fit..")}
    return data


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
    )
    circadianess_df["phase_plt"] = circadianess_df["phase_plt"] + np.where(
        circadianess_df["phase_plt"] < 0, 24, 0
    )
    if time_reference:
        circadianess_df = circadianess_df[
            circadianess_df["phase_plt"] >= (circadianess_df["time_reference"] - 12)
        ]
        circadianess_df = circadianess_df[
            circadianess_df["phase_plt"] < (circadianess_df["time_reference"] + 12)
        ]
    else:
        circadianess_df = circadianess_df[circadianess_df["phase_plt"] >= 0]
        circadianess_df = circadianess_df[circadianess_df["phase_plt"] < 24]
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
