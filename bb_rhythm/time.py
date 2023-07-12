from suncalc import get_times
import datetime
import pandas as pd


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

    def get_time_shift_relative_to_solar_reference(self, time_utc, reference="solar_noon"):
        solar_reference = self.get_solar_reference(time_utc, reference=reference)
        time_shift = pd.to_timedelta(solar_reference - pd.Timestamp(datetime.datetime(*time_utc.timetuple()[:3]))) / pd.offsets.Hour(1)
        return time_shift

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
