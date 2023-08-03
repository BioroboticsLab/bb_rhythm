from suncalc import get_times
import datetime
import pandas as pd
import numpy as np

from . import rhythm


class SolarTimeConverter:
    def __init__(self, latitude, longitude):
        self.latitude = latitude
        self.longitude = longitude

    def convert_utc_to_local_mean_time(self, time_utc):
        # get local time when sun stays in zenith
        solar_reference = self.get_solar_reference(time_utc, reference="solar_noon")
        # get difference to utc time
        reference_shift = (
            pd.Timestamp(datetime.datetime(*time_utc.timetuple()[:3]))
            + datetime.timedelta(hours=12)
            - solar_reference
        )
        # add difference to utc time
        local_mean_time = time_utc - reference_shift
        return local_mean_time.tz_convert(None)

    def get_solar_reference(self, time_utc, reference="solar_noon"):
        solar_reference = get_times(time_utc, self.longitude, self.latitude)[reference]
        # get difference to utc time
        return solar_reference

    def get_time_shift_relative_to_solar_reference(
        self, time_utc, reference="solar_noon"
    ):
        solar_reference = self.get_solar_reference(time_utc, reference=reference)
        time_shift = pd.to_timedelta(
            solar_reference - pd.Timestamp(datetime.datetime(*time_utc.timetuple()[:3]))
        ) / pd.offsets.Hour(1)
        return time_shift


def get_coordinates_berlin():
    latitude = 52.5
    longitude = 13.4
    return latitude, longitude


def apply_time_wrapper_berlin(df, reference="sunrise"):
    # create time converter
    latitude, longitude = get_coordinates_berlin()
    berlin_time_converter = SolarTimeConverter(latitude, longitude)
    # get reference time for sunset
    df["time_reference"] = df.date.apply(
        berlin_time_converter.get_time_shift_relative_to_solar_reference,
        reference=reference,
    )


def map_pi_time_interval_to_24h(pi_time):
    return pi_time * (12 / np.pi)


def phase_to_utc_time(phase_df, fit_type="cosine"):
    rhythm.add_phase_plt_to_df(phase_df)
    if fit_type == "cosine":
        phase = phase_df["phase_plt"] - 12
    else:
        phase = phase_df["phase_plt"]
    phase_df["phase_utc"] = phase_df.date + (
        np.array([datetime.timedelta(hours=p) for p in phase])
    )
