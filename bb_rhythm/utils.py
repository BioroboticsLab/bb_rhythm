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


def calculate_regression(df, x_column, y_column, type="linear", printing=True, degree=3):
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
        wt = 1 / smf.ols('fit.resid.abs() ~ fit.fittedvalues', data=df).fit().fittedvalues ** 2

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
