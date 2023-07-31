from wetterdienst.provider.dwd.observation import (
    DwdObservationRequest,
    DwdObservationResolution,
)
from wetterdienst import Settings


def get_weather_parameter_df(dt_from, dt_to, parameter, station_name="Berlin-Tempelhof"):
    settings = Settings(tidy=True, si_units=False, humanize=True)
    stations = DwdObservationRequest(
        parameter=[(parameter)],
        resolution=DwdObservationResolution.MINUTE_10,
        start_date=dt_from,
        end_date=dt_to,
        settings=settings,
    ).filter_by_name(name=station_name)
    weather_lst = []
    for res in stations.values.query():
        weather_lst.append(res)
    return weather_lst[0].df


def get_air_temperature_frame(dt_from, dt_to, station_name="Berlin-Tempelhof"):
    settings = Settings(tidy=True, si_units=False, humanize=True)
    stations = DwdObservationRequest(
        parameter=[("temperature_air_mean_200")],
        resolution=DwdObservationResolution.MINUTE_10,
        start_date=dt_from,
        end_date=dt_to,
        settings=settings,
    ).filter_by_name(name=station_name)
    temp_lst = []
    for res in stations.values.query():
        temp_lst.append(res)
    return temp_lst[0].df


def get_wind_speed_frame(dt_from, dt_to, station_name="Berlin-Tempelhof"):
    settings = Settings(tidy=True, si_units=False, humanize=True)
    stations = DwdObservationRequest(
        parameter=[("wind_speed")],
        resolution=DwdObservationResolution.MINUTE_10,
        start_date=dt_from,
        end_date=dt_to,
        settings=settings,
    ).filter_by_name(name=station_name)
    wind_lst = []
    for res in stations.values.query():
        wind_lst.append(res)
    return wind_lst[0].df


def combine_temp_wind_frames(temp_frame, wind_frame):
    frame = temp_frame.merge(
        wind_frame,
        left_on=["date", "station_id"],
        right_on=["date", "station_id"],
        suffixes=("_temp", "_wind"),
    )
    frame.drop(
        columns=[
            "dataset_temp",
            "parameter_temp",
            "quality_temp",
            "dataset_wind",
            "parameter_wind",
            "quality_wind",
        ],
        inplace=True,
    )
    frame.rename(
        columns={
            "value_temp": "temperature_air_mean_200",
            "value_wind": "wind_speed_mean",
        },
        inplace=True,
    )
    return frame


def get_weather_frame(dt_from, dt_to, station_name="Berlin-Tempelhof"):
    temp_frame = get_air_temperature_frame(dt_from, dt_to, station_name)
    wind_frame = get_wind_speed_frame(dt_from, dt_to, station_name)
    weather_frame = combine_temp_wind_frames(temp_frame, wind_frame)
    return weather_frame
