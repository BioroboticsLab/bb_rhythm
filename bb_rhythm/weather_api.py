import datetime
from wetterdienst.provider.dwd.observation import (
    DwdObservationRequest,
    DwdObservationResolution,
)
from wetterdienst import Settings
from functools import reduce
import pandas as pd
import numpy as np
import pytz

import bb_behavior.db

from . import statistics


def get_weather_parameter_df(
    dt_from, dt_to, parameter, station_name="Berlin-Tempelhof"
):
    settings = Settings(tidy=True, si_units=False, humanize=True)
    parameter = [(param) for param in parameter]
    stations = DwdObservationRequest(
        parameter=parameter,
        resolution=DwdObservationResolution.MINUTE_10,
        start_date=dt_from,
        end_date=dt_to,
        settings=settings,
    ).filter_by_name(name=station_name)
    weather_lst = []
    for res in stations.values.query():
        weather_lst.append(res)
    return weather_lst[0].df


def combine_weather_frames(df):
    dfs = []
    for name, group in df.groupby(["parameter"]):
        group.rename(columns={"value": name}, inplace=True)
        group.drop(columns=["parameter"], inplace=True)
        dfs.append(group)
    df_merged = reduce(
        lambda left, right: pd.merge(
            left,
            right,
            left_on=["date", "station_id"],
            right_on=["date", "station_id"],
            how="outer",
        ),
        dfs,
    )
    return df_merged


def get_weather_frame(
    dt_from,
    dt_to,
    station_name="Berlin-Tempelhof",
    weather_params=["wind_speed", "temperature_air_mean_200"],
):
    df = get_weather_parameter_df(
        dt_from, dt_to, parameter=weather_params, station_name=station_name
    )
    df.drop(columns=["quality", "dataset"], inplace=True)
    weather_df = combine_weather_frames(df)
    return weather_df


def combine_weather_velocity_dfs(velocity_df, weather_df):
    """

    :param velocity_df:
    :param weather_df:
    :return:
    """
    velocity_df.rename(columns={"datetime": "date"}, inplace=True)
    velocity_df.drop(columns=["time_passed"], inplace=True)
    velocity_df = velocity_df[
        ~(pd.isnull(velocity_df.velocity) | np.isinf(velocity_df.velocity))
    ]
    velocity_df["date"] = velocity_df.date.dt.round("10min")
    velocity_df = velocity_df.groupby(["date"])["velocity"].mean().reset_index()
    weather_df.drop(columns=["station_id"], inplace=True)
    velocity_weather_df = weather_df.set_index("date").join(
        velocity_df.set_index("date"), on="date", how="inner"
    )
    return velocity_weather_df


def get_max_min_ccr_agg(x):
    try:
        columns = {
            "max_corr": x.ccr.max(),
            "max_lag": x.lags.iloc[x.ccr.argmax()] * 10 / 60,
            "min_corr": x.ccr.min(),
            "min_lag": x.lags.iloc[x.ccr.argmin()] * 10 / 60,
        }
    except TypeError:
        columns = {
            "max_corr": np.nan,
            "max_lag": np.nan,
            "min_corr": np.nan,
            "min_lag": np.nan,
        }
    return pd.Series(columns, index=["max_corr", "max_lag", "min_corr", "min_lag"])


def create_min_max_ccr_df_per_bee(bee_id, cc_df):
    """

    :param bee_id:
    :param cc_df:
    :return:
    """
    df_corr = (
        cc_df.groupby(["bee_id", "date", "age", "parameter"])
        .apply(get_max_min_ccr_agg)
        .reset_index()
    )
    return df_corr


def create_ccr_df_per_bee_from_period(
    bee_id, dt_from, dt_to, velocity_weather_df, delta=datetime.timedelta(days=1)
):
    """

    :param bee_id:
    :param dt_from:
    :param dt_to:
    :param velocity_weather_df:
    :param delta:
    :return:
    """
    dates = list(pd.date_range(start=dt_from, end=dt_to, tz=pytz.UTC).to_pydatetime())
    cross_correlations_dfs = []
    for current_dt in dates:
        bee_age = int(
            bb_behavior.db.metadata.get_bee_ages([(bee_id, current_dt.date())])[0][2]
        )
        if bee_age < 1:
            print("Bee age: %d" % bee_age)
            continue
        velocity_weather_df_day = velocity_weather_df[
            (velocity_weather_df.index >= (current_dt))
            & (velocity_weather_df.index < (current_dt + delta))
        ]
        if len(velocity_weather_df_day) == 0:
            print("No velocities for %s" % str(current_dt))
            continue
        cross_correlation_df = statistics.apply_time_lagged_cross_correlation_to_df(
            velocity_weather_df_day
        )
        if cross_correlation_df is None:
            print("df: %s" % str(velocity_weather_df_day))
            continue
        cross_correlation_df["date"] = len(cross_correlation_df) * [current_dt]
        cross_correlation_df["bee_id"] = len(cross_correlation_df) * [bee_id]
        cross_correlation_df["age"] = len(cross_correlation_df) * [bee_age]
        cross_correlations_dfs.append(cross_correlation_df)
    if len(cross_correlations_dfs) == 0:
        return None
    cc_df = pd.concat(cross_correlations_dfs)
    return cc_df


def calculate_weather_activity_cross_correlation(
    bee_id, dt_from, dt_to, weather_df_path=None, velocity_df_path=None, cc_path=None
):
    """

    :param bee_id:
    :param dt_from:
    :param dt_to:
    :param weather_df_path:
    :param velocity_df_path:
    :param cc_path:
    :return:
    """
    # fetch weather df
    weather_df = pd.read_pickle(weather_df_path)
    weather_df.drop(columns=["lat", "long"], inplace=True)

    # fetch velocities
    velocity_df = utils.fetch_velocities_from_remote_or_db(
        bee_id, dt_to, dt_from, velocity_df_path
    )
    if velocity_df is None:
        return {None: "No velocities could be fetched"}
    velocity_df.dropna(inplace=True)
    if len(velocity_df) == 0:
        return {None: "No velocities could be fetched"}

    # combine weather and velocity df
    velocity_weather_df = combine_weather_velocity_dfs(velocity_df, weather_df)

    # per bee_id full cross correlation velocity and weather per day
    cc_df = create_ccr_df_per_bee_from_period(
        bee_id, dt_from, dt_to, velocity_weather_df
    )
    if cc_df is None:
        return {None: "Bee is dead"}
    cc_df.to_pickle(os.path.join(cc_path, "%s.pkl" % bee_id))

    # get per bee, per day, per weather param max, min ccr
    df_corr = weather_api.create_min_max_ccr_df_per_bee(bee_id, cc_df)
    return df_corr
