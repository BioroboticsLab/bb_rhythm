from suncalc import get_position, get_times
import datetime
import pandas as pd


class SolarTimeConverter:
    def __init__(self, latitude, longitude):
        self.latitude = latitude
        self.longitude = longitude

    def convert_utc_to_local_mean_time(self, time_utc, reference="solar_noon"):
        # get local time when sun stays in zenith
        solar_noon = get_times(time_utc, self.longitude, self.latitude)[reference]
        # get difference to utc time
        noon_shift = (
            pd.Timestamp(datetime.datetime(*time_utc.timetuple()[:3]))
            + datetime.timedelta(hours=12)
            - solar_noon
        )
        # add difference to utc time
        local_mean_time = time_utc + noon_shift
        return local_mean_time.tz_convert(None)

    def convert_local_mean_time_to_utc(self):
        pass

    def convert_utc_to_solar_time(self):
        pass

    def convert_solar_time_to_utc(self):
        pass

    def convert_local_mean_time_to_solar_time(self):
        pass

    def convert_solar_time_to_local_mean_time(self):
        pass


def get_coordinates_berlin():
    latitude = 52.5
    longitude = 13.4
    return latitude, longitude
