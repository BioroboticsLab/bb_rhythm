import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
import statsmodels.formula.api as smf


def add_age_bins(velocity_df, step_size=5, age_bins=None):
    # subgroup per age group in steps of five
    max_age = velocity_df.age.max()
    n_age_bins = int(np.ceil(max_age / step_size))
    if age_bins is not None:
        age_bins, age_map = create_age_map_custom(age_bins, max_age)
    else:
        age_bins, age_map = create_age_map_bin(max_age, n_age_bins, step_size)
    velocity_df["age_bins"] = [
        age_map[str(item)] for item in pd.cut(x=velocity_df["age"], bins=age_bins)
    ]


def create_age_map_custom(age_bins, max_age):
    age_map = {}
    for i in range(len(age_bins) - 1):
        age_map[
            ("(%s, %s]" % (str(float(age_bins[i])), str(float(age_bins[i + 1]))))
        ] = ("0%s+" % str(age_bins[i]))[-3:]
    age_map[
        ("(%s, %s]" % (str(float(age_bins[len(age_bins) - 1])), str(float(max_age))))
    ] = ("0%s+" % str(age_bins[len(age_bins) - 1]))[-3:]
    age_map["nan"] = "Nan"
    age_bins.append(int(max_age))
    return age_bins, age_map


def add_euqal_n_duration_bins(n_bins, vel_change_df):
    vel_change_df.sort_values("duration", inplace=True)
    vel_change_df["bins_bee_focal"] = pd.qcut(x=vel_change_df["duration"], q=n_bins)
    vel_change_df["bins_bee_non_focal"] = vel_change_df["bins_bee_focal"]


def add_equal_n_age_bins(n_bins, vel_change_df):
    vel_change_df.sort_values("age_focal", inplace=True)
    vel_change_df["bins_bee_focal"] = pd.qcut(x=vel_change_df["age_focal"], q=n_bins)
    vel_change_df.sort_values("age_non_focal", inplace=True)
    vel_change_df["bins_bee_non_focal"] = pd.qcut(
        x=vel_change_df["age_non_focal"], q=n_bins
    )


def add_equal_n_circ_bins(n_bins, vel_change_df):
    vel_change_df.sort_values("circadianess_focal", inplace=True)
    vel_change_df["bins_bee_focal"] = pd.qcut(
        x=vel_change_df["circadianess_focal"], q=n_bins
    )
    vel_change_df.sort_values("circadianess_non_focal", inplace=True)
    vel_change_df["bins_bee_non_focal"] = pd.qcut(
        x=vel_change_df["circadianess_non_focal"], q=n_bins
    )


def add_amplitude_bins(n_bins, vel_change_df):
    vel_change_df["bins_bee_focal"] = pd.qcut(
        x=vel_change_df["amplitude_focal"], q=n_bins
    )
    vel_change_df["bins_bee_non_focal"] = pd.qcut(
        x=vel_change_df["amplitude_non_focal"], q=n_bins
    )


def add_equal_dist_circ_bins(n_bins, vel_change_df):
    bins_bee_0 = bin_equ_dist_circ(n_bins, vel_change_df, "circadianess_bee0")
    bins_bee_1 = bin_equ_dist_circ(n_bins, vel_change_df, "circadianess_bee1")
    vel_change_df["bins_bee_focal"] = pd.cut(
        x=vel_change_df["circadianess_focal"], bins=bins_bee_0
    )
    vel_change_df["bins_bee_non_focal"] = pd.cut(
        x=vel_change_df["circadianess_non_focal"], bins=bins_bee_1
    )


def bin_equ_dist_circ(n_bins, vel_change_df, circadianess):
    # subgroup per circadianess
    max_circadian = vel_change_df[circadianess].max()
    min_circadian = vel_change_df[circadianess].min()
    bin_range = (max_circadian - min_circadian) / n_bins
    bins = [i * bin_range + min_circadian for i in range(n_bins)]
    bins.append(max_circadian)
    return bins


def add_circadian_bins(vel_change_df, metric="equal_circ_distance", n_bins=7):
    if metric == "equal_circ_distance":
        add_equal_dist_circ_bins(n_bins, vel_change_df)
    if metric == "equal_bin_size":
        add_equal_n_circ_bins(n_bins, vel_change_df)
    if metric == "age":
        add_equal_n_age_bins(n_bins, vel_change_df)
    if metric == "age_5":
        add_5_step_age_bins(
            6, vel_change_df
        )  # int(np.ceil(vel_change_df.age_focal.max() / 5))
    if metric == "duration":
        add_euqal_n_duration_bins(n_bins, vel_change_df)
    if metric == "start_vel":
        add_equal_n_start_vel_bins(n_bins, vel_change_df)
    if metric == "new_equal_bin_size":
        new_add_equal_n_circ_bins(n_bins, vel_change_df)
    if metric == "amplitude":
        add_amplitude_bins(n_bins, vel_change_df)


def new_add_equal_n_circ_bins(n_bins, vel_change_df):
    vel_change_df["bins_bee_focal"] = pd.qcut(
        x=vel_change_df["circadianess_focal"], q=n_bins
    )
    bins = pd.IntervalIndex(vel_change_df["bins_bee_focal"].values.unique())
    vel_change_df["bins_bee_non_focal"] = pd.cut(
        x=vel_change_df["circadianess_non_focal"], bins=bins
    )


def add_equal_n_start_vel_bins(n_bins, vel_change_df):
    vel_change_df.sort_values("vel_start_focal", inplace=True)
    vel_change_df["bins_bee_focal"] = pd.qcut(
        x=vel_change_df["vel_start_focal"], q=n_bins
    )
    vel_change_df.sort_values("vel_start_non_focal", inplace=True)
    vel_change_df["bins_bee_non_focal"] = pd.qcut(
        x=vel_change_df["vel_start_non_focal"], q=n_bins
    )


def add_5_step_age_bins(n_age_bins, velocity_df, step_size=5):
    # subgroup per age group in steps of five
    max_age = velocity_df.age_focal.max()
    age_bins, age_map = create_age_map_bin(max_age, n_age_bins, step_size)
    velocity_df["bins_bee_focal"] = [
        age_map[str(item)] for item in pd.cut(x=velocity_df["age_focal"], bins=age_bins)
    ]
    velocity_df["bins_bee_non_focal"] = [
        age_map[str(item)]
        for item in pd.cut(x=velocity_df["age_non_focal"], bins=age_bins)
    ]


def create_age_map_bin(max_age, n_age_bins, step_size):
    age_map = {}
    age_bins = []
    i_out = 0
    for i in range(n_age_bins):
        if i > 8:
            break
        age_map[
            (
                "(%s, %s]"
                % (str(float(i * step_size)), str(float(i * step_size + step_size)))
            )
        ] = ("0%s+" % str(i * step_size))[-3:]
        age_bins.append(i * step_size)
        i_out = i
    age_map[("(%s, %s]" % (str(float(i_out * step_size)), str(float(max_age))))] = (
        "0%s+" % str(i_out * step_size)
    )[-3:]
    age_map["nan"] = "Nan"
    age_bins.append(int(max_age))
    return age_bins, age_map


def calculate_regression(
    df, x_column, y_column, type="linear", printing=True, degree=3
):
    # define predictor and response variables
    y = df[y_column]
    x = df[x_column]

    # add constant to predictor variables
    X = sm.add_constant(x)

    # fit linear regression model
    if type == "linear":
        fit = sm.OLS(y, X).fit()

    # fit weighted least squares regression model
    elif type == "weighted_linear":
        # define weights to use
        wt = (
            1
            / smf.ols("fit.resid.abs() ~ fit.fittedvalues", data=df).fit().fittedvalues
            ** 2
        )

        # fit weighted least squares regression model
        fit = sm.WLS(y, X, weights=wt).fit()

    elif type == "polynomial":
        # define polynomial x values
        polynomial_features = PolynomialFeatures(degree=degree)
        xp = polynomial_features.fit_transform(X)

        # calculate fit
        fit = sm.OLS(y, xp).fit()

    else:
        raise NotImplementedError

    # view model summary
    if printing:
        print(fit.summary())

    return fit
