import os

import bb_behavior.db
import numpy as np
import pandas as pd

class Binning:
    def __init__(self, bin_name, bin_parameter):
        self.bin_name = bin_name
        self.bin_parameter = bin_parameter
        self.bin_n = None
        self.bin_max_n = None
        self.step_size = None
        self.bin_labels = None
        self.bins = None
        self.bin_max_value = None
        self.remove_none = True
        self.is_categorical = None

    def replace_bin_identifier_by_bin_map_identifier(self, df):
        df[self.bin_name] = [self.bin_labels[item] for item in self.bins]

    def create_bin(self, df=None, bins=None):
        """
        Creates self.bins according step size and bin number limit.
        """
        if self.is_categorical:
            self.bins = df[self.bin_parameter]
        if (bins is not None) and (not self.is_categorical):
            bins.append(self.bin_max_value)
            self.bins = pd.cut(
                x=df[self.bin_parameter], bins=pd.IntervalIndex.from_breaks(bins)
            )
        if (self.step_size is None) and (self.bins is None):
            self.bins = pd.qcut(x=df[self.bin_parameter], q=self.bin_n)
        if (self.step_size is not None) and (self.bins is None):
            bins = []
            for i in range(int(self.bin_n)):
                bins.append(i * self.step_size)
            self.bins = pd.cut(
                x=df[self.bin_parameter], bins=pd.IntervalIndex.from_breaks(bins)
            )
        if self.bins is None:
            assert ValueError

    def create_bin_labels(self, bin_labels=None):
        """
        Creates human nicely readable age map.

        :return:
        """
        bins_unique = self.bins.unique()
        for i in range(len(self.bins)):
            try:
                if np.isnan(self.bins.iloc[i]):
                    self.bins.iloc[i] = np.nan
            except TypeError:
                continue
        #self.bins.replace({None: np.nan}, inplace=True)
        self.bin_labels = {}
        if (np.isnan(b) for b in bins_unique):
            if self.remove_none:
                self.bin_labels[np.nan] = np.nan
            else:
                self.bin_labels[np.nan] = "Nan"
        bins_unique = bins_unique[~pd.isna(bins_unique)]
        i = 0
        for b in sorted(bins_unique):
            if bin_labels is None:
                if not "age" in self.bin_parameter:
                    label = "(%s, %s]" % (
                        str(b.left.round(2)),
                        str(b.right.round(2)),
                    )
                else:
                    if b.left > 9:
                        label = "%d+" % int(b.left)
                    else:
                        label = "0%d+" % int(b.left)
            else:
                label = bin_labels[i]
            self.bin_labels[b] = label
            i += 1

    def add_bins_to_df(
        self,
        df,
        step_size=5,
        n_bins=6,
        bin_max_n=None,
        remove_none=True,
        bins=None,
        bin_labels=None,
        is_categorical=None
    ):
        self.remove_none = remove_none
        if is_categorical or (df[self.bin_parameter].dtype == pd.Categorical) or (df[self.bin_parameter].dtype == bool):
            self.is_categorical = True
        else:
            self.bin_max_value = df[self.bin_parameter].max()
            # case if not custom
            if self.bins is None:
                self.bin_max_n = bin_max_n
                self.step_size = step_size
                # case step size given
                if self.step_size:
                    self.bin_n = np.floor(self.bin_max_value / self.step_size)
                    # case step size and max bin given
                    if self.bin_max_n:
                        if self.bin_max_n < self.bin_n:
                            self.bin_n = self.bin_max_n
                # case number of bins given
                else:
                    self.bin_n = n_bins
        self.create_bin(df=df, bins=bins)
        self.create_bin_labels(bin_labels)
        self.replace_bin_identifier_by_bin_map_identifier(df)
        df.dropna(subset=[self.bin_name], inplace=True)
        return df

def fetch_velocities_from_remote_or_db(
    bee_id, dt_after, dt_before, velocities_path, max_mm_per_second=15.0
):
    if type(bee_id) == np.int64:
        bee_id = bee_id.item()
    try:
        # fetch velocities
        if velocities_path is not None:
            velocities = pd.read_pickle(
                os.path.join(velocities_path, "%d.pickle" % bee_id)
            )
            print(bee_id)
            velocities.velocity[velocities.velocity > max_mm_per_second] = np.nan
        else:
            assert FileNotFoundError
    except FileNotFoundError:
        print(bee_id)
        # fetch velocities
        velocities = bb_behavior.db.trajectory.get_bee_velocities(
            bee_id,
            dt_before,
            dt_after,
            confidence_threshold=0.1,
            max_mm_per_second=max_mm_per_second,
        )
    return velocities
