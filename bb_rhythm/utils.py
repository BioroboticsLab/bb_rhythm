import numpy as np
import pandas as pd


class Binning:
    def __init__(self, bin_name, bin_parameter):
        self.bin_name = bin_name
        self.bin_parameter = bin_parameter
        self.bin_n = None
        self.bin_max_n = None
        self.step_size=None
        self.bin_map = None
        self.bins = None
        self.bin_max_value=None
        self.remove_none = True

    def set_bin_n(self, bin_n):
        self.bin_n = bin_n

    def set_bin_max_n(self, bin_max_n):
        self.bin_max_n = bin_max_n

    def set_step_size(self, step_size):
        self.step_size = step_size

    def set_step_size(self, step_size):
        self.step_size = step_size

    def set_bin_map(self, bin_map):
        self.bin_map = bin_map

    def set_bins(self, bins):
        self.bins = bins
    def set_remove_none(self, remove_none):
            self.remove_none = remove_none

    def set_bin_name(self, bin_name):
        self.remove_none = bin_name
    def _replace_bin_identifier_by_bin_map_identifier(self, df):
        df[self.bin_name] = [
            self.bin_map[str(item)] for item in pd.cut(x=df[bin_parameter], bins=self.bins)
        ]

    def _create_bin_map(self):
        if self.bin_parameter == "age":
            self._create_age_bin_map()
        else:
            self._create_float_bin_map()

    def _create_bin(self, df=None):
        """
        Creates self.bins according step size and bin number limit.
        """
        if self.n_bin is None:
            self.bin_n = int(np.ceil(self.bin_max_value / self.step_size))
            self.bins = []
            i_out = 0
            for i in range(self.n_bin):
                if self.bin_max_n is not None:
                    if i > self.bin_max_n:
                        break
                self.bins.append(i * self.step_size)
                i_out = i
            self.bins.append(int(self.bin_max_value))
        else:
            self.bins = pd.qcut(
            x=df[self.bin_parameter], q=self.bin_n
        )

    def _create_age_bin_map(self):
        """
        Creates human nicely readable age map.

        :return:
        """
        self.bin_map = {}
        for i in range(len(self.bins) - 1):
            self.bin_map[
                ("(%s, %s]" % (str(float(self.bins[i])), str(float(self.bins[i + 1]))))
            ] = ("0%s+" % str(self.bins[i]))[-3:]
        self.bin_map[
            ("(%s, %s]" % (str(float(self.bins[len(self.bins) - 1])), str(float(self.bin_max_value))))
        ] = ("0%s+" % str(self.bins[len(self.bins) - 1]))[-3:]
        self.bin_map["nan"] = "Nan"

    def _create_float_bin_map(self):
        """
        Creates human nicely readable age map.

        :return:
        """
        self.bin_map = {}
        for i in range(len(self.bins) - 1):
            self.bin_map[
                ("(%s, %s]" % (str(float(self.bins[i].left)), str(float(self.bins[i].right))))
            ] = ("(%s, %s]" % (str(float(self.bins[i].left.round(0.01))), str(float(self.bins[i].right.round(0.01)))))
        self.bin_map["nan"] = "Nan"

    def add_bins_to_df(self, df, step_size=5, n_bins=6, bin_max_n=None, remove_none=True):
        self.bin_max_value = df[self.bin_parameter].max()
        if self.bins is None:
            self.set_bin_n(n_bins)
            self.set_bin_max_n(bin_max_n)
            self._create_bin(df=df)
        if self.bin_map is None:
            self.create_bin_map()
        self._replace_bin_identifier_by_bin_map_identifier(df)
        self.set_remove_none(remove_none)
        if self.remove_none:
            df = df[df[self.bin_name] != "Nan"]
        return df