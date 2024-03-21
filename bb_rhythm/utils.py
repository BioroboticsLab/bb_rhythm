import os
import bb_behavior.db
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


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
        self.label_type = str

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
                x=df[self.bin_parameter].replace({np.inf: np.nan, -np.inf: np.nan}),
                bins=pd.IntervalIndex.from_breaks(bins),
            )
        if (self.step_size is None) and (self.bins is None):
            self.bins = pd.qcut(
                x=df[self.bin_parameter].replace({np.inf: np.nan, -np.inf: np.nan}),
                q=self.bin_n,
                duplicates="drop",
            )
        if (self.step_size is not None) and (self.bins is None):
            bins = []
            for i in range(int(self.bin_n)):
                bins.append(i * self.step_size)
            bins.append(self.bin_max_value)
            self.bins = pd.cut(
                x=df[self.bin_parameter].replace({np.inf: np.nan, -np.inf: np.nan}),
                bins=pd.IntervalIndex.from_breaks(bins),
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
        # self.bins.replace({None: np.nan}, inplace=True)
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
                    if self.label_type == str:
                        label = "(%s, %s]" % (
                            str(b.left.round(2)),
                            str(b.right.round(2)),
                        )
                    else:
                        label = pd.Interval(b.left.round(2), b.right.round(2))
                else:
                    if b.left > 8:
                        label = "%d+" % int(b.left + 1)
                    else:
                        label = "0%d+" % int(b.left + 1)
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
        is_categorical=None,
        label_type=str,
    ):
        self.remove_none = remove_none
        self.label_type = label_type
        if (
            is_categorical
            or (df[self.bin_parameter].dtype == pd.Categorical)
            or (df[self.bin_parameter].dtype == bool)
        ):
            self.is_categorical = True
        else:
            self.bin_max_value = df[self.bin_parameter].max()
            # case if not custom
            if self.bins is None:
                self.bin_max_n = bin_max_n
                self.step_size = step_size
                # case step size given
                if self.step_size:
                    self.bin_n = np.ceil(self.bin_max_value / self.step_size)
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
    """

    :param bee_id:
    :param dt_after:
    :param dt_before:
    :param velocities_path:
    :param max_mm_per_second:
    :return:
    """
    if type(bee_id) == np.int64:
        bee_id = bee_id.item()
    try:
        # fetch velocities
        if velocities_path is not None:
            velocities = pd.read_pickle(
                os.path.join(velocities_path, "%d.pickle" % bee_id)
            )
            velocities.velocity[velocities.velocity > max_mm_per_second] = np.nan
        else:
            assert FileNotFoundError
    except FileNotFoundError:
        # fetch velocities
        velocities = bb_behavior.db.trajectory.get_bee_velocities(
            bee_id,
            dt_before,
            dt_after,
            confidence_threshold=0.1,
            max_mm_per_second=max_mm_per_second,
        )
    return velocities


def split_ci_lower_upper(df, variables):
    df_plt = df.copy()
    df_plt = df_plt[df_plt["index"] == 0]
    df_plt.drop(columns="index", inplace=True)
    for var in variables:
        ci_var_lower = df[df["index"] == 0]["ci_%s" % var]
        ci_var_upper = df[df["index"] == 1]["ci_%s" % var]
        df_plt.drop(columns="ci_%s" % var, inplace=True)
        df_plt["ci_%s_lower" % var] = ci_var_lower.values
        df_plt["ci_%s_upper" % var] = ci_var_upper.values
    df_plt.reset_index(inplace=True)
    df_plt.drop(columns=["index"], inplace=True)
    return df_plt


def nan_tolerant_gaussian_filtering(U, sigma):
    V = U.copy()
    V[np.isnan(U)] = 0
    VV = gaussian_filter(V, sigma=sigma)

    W = 0 * U.copy()+1
    W[np.isnan(U)] = 0
    WW = gaussian_filter(W, sigma=sigma)
    return VV/WW