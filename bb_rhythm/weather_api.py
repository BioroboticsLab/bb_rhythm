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
    velocity_df.rename(columns={"datetime": "date"}, inplace=True)
    velocity_df.drop(columns=["time_passed"], inplace=True)
    velocity_df = velocity_df[
        ~(pd.isnull(velocity_df.velocity) |
            np.isinf(velocity_df.velocity))
    ]
    velocity_df["date"] = velocity_df.date.dt.round("10min")
    velocity_df = (
        velocity_df
        .groupby(["date"])["velocity"]
        .mean()
        .reset_index()
    )
    weather_df.drop(columns=["station_id"], inplace=True)
    velocity_weather_df = weather_df.set_index("date").join(
        velocity_df.set_index("date"), on="date", how="inner"
    )
    return velocity_weather_df


def create_min_max_ccr_df_per_bee(bee_id, cc_df):
    df_ccf_ls = []
    for parameter in cc_df.parameter.unique():
        df_plot = cc_df[cc_df.parameter == parameter].groupby(["lags"]).agg("mean").reset_index()
        for date in cc_df[cc_df.parameter == parameter].date.unique():
            df_plot_day = cc_df[cc_df.parameter == parameter][cc_df[cc_df.parameter == parameter].date == date]
            max_corr = df_plot_day.ccf.max()
            max_leg = df_plot.lags.iloc[df_plot_day.ccf.argmax()] * 10 / 60
            min_corr = df_plot_day.ccf.min()
            min_leg = df_plot.lags.iloc[df_plot_day.ccf.argmin()] * 10 / 60
            df_ccf_ls.append(
                {"parameter": parameter, "max_corr": max_corr, "max_corr_time": max_leg, "min_corr": min_corr,
                 "min_corr_time": min_leg, "date": date, "bee_id": bee_id, "age": cc_df["age"].iloc[0]})
    df_corr = pd.DataFrame(df_ccf_ls)
    return df_corr


def create_ccr_df_per_bee_from_period(bee_id, dt_from, dt_to, velocity_weather_df, delta=datetime.timedelta(days=1)):
    dates  = list(
        pd.date_range(
            start=dt_from,
            end=dt_to,
            tz=pytz.UTC,
        ).to_pydatetime()
    )
    cross_correlations_dfs = []
    for current_dt in dates:
        bee_age = int(
            bb_behavior.db.metadata.get_bee_ages([(bee_id, current_dt.date())])[0][
                2
            ]
        )
        if bee_age < 1:
            continue
        velocity_weather_df_day = velocity_weather_df[
            (velocity_weather_df.date >= (current_dt)) &
            (velocity_weather_df.date < (current_dt + delta))
            ]
        cross_correlation_df = statistics.apply_time_lagged_cross_correlation_to_df(velocity_weather_df_day)
        cross_correlation_df["date"] = len(cross_correlation_df) * [current_dt]
        cross_correlation_df["bee_id"] = len(cross_correlation_df) * [bee_id]
        cross_correlation_df["age"] = len(cross_correlation_df) * [bee_age]
        cross_correlations_dfs.append(cross_correlation_df)
    if len(cross_correlations_dfs) == 0:
        return None
    cc_df = pd.concat(cross_correlations_dfs)
    return cc_df
