import numpy as np
import pandas as pd
import datetime
import pytz

import bb_circadian.lombscargle
import bb_behavior.db

from . import time

def fit_circadianess_fit_per_bee(server=None, day=None, bee_id=None, from_dt=None, to_dt=None, bee_age=None):
    if bee_age == -1 or bee_age == 0:
        return {None: dict(error="Bee is already dead or new to colony..")}

    with server:
        # fetch velocities
        velocities = bb_behavior.db.trajectory.get_bee_velocities(
            bee_id, from_dt, to_dt, confidence_threshold=0.1, max_mm_per_second=15.0
        )

        if velocities is None:
            return {None: dict(error="No velocities could be fetched..")}

        try:
            # get right data types
            day = datetime.datetime.fromisoformat(day)
            assert day.tzinfo == datetime.timezone.utc
            day = day.replace(tzinfo=pytz.UTC)

            # remove NaNs and infs
            velocities = velocities[~pd.isnull(velocities.velocity)]

            # calculate circadianess
            data = bb_circadian.lombscargle.collect_circadianess_data_for_bee_date(
                bee_id, day, velocities=velocities, n_workers=0
            )
            if data:
                # add parameters
                data["age"] = bee_age
                # parameters for quality of velocities
                add_velocity_quality_params(data, velocities)
                # extract from parameters of fit
                extract_parameters_from_circadian_fit(data)
            else:
                assert ValueError
        except (AssertionError, ValueError, IndexError, RuntimeError):
            data = {None: dict(error="Something went wrong during the fit..")}
        return data


def extract_parameters_from_circadian_fit(data):
    data["amplitude"] = data["parameters"][0]
    data["phase"] = data["parameters"][1]
    data["offset"] = data["parameters"][2]


def add_velocity_quality_params(data, velocities):
    data["n_data_points"] = len(velocities)
    data["data_point_dist_max"] = velocities["time_passed"].max()
    data["data_point_dist_min"] = velocities["time_passed"].min()
    data["data_point_dist_mean"] = velocities["time_passed"].mean()
    data["data_point_dist_median"] = velocities["time_passed"].median()


def create_agg_circadian_df(circadianess_df, column="age_bins", agg_func="mean"):
    return (
        circadianess_df.pivot_table(
            index=["date"],
            columns=column,
            values="well_tested_circadianess",
            aggfunc=agg_func,
        )
        .melt(ignore_index=False)
        .reset_index()
    )


def create_mean_count_circadianess_per_day_df(circadianess_df, column="age_bins"):
    circadianess_df_mean = create_agg_circadian_df(circadianess_df, column=column)
    circadianess_df_count = create_agg_circadian_df(
        circadianess_df, column=column, agg_func="count"
    )
    circadianess_df = circadianess_df_mean.rename(columns={"value": "mean"})
    circadianess_df["count"] = circadianess_df_count["value"]

    # filter counts lower than 0.05 counts out
    circadianess_df = circadianess_df[
        circadianess_df["count"] > circadianess_df["count"].quantile(q=0.05)
    ]
    return circadianess_df


def calculate_well_tested_circadianess(circadianess_df):
    circadianess_df["is_good_fit"] = (circadianess_df.goodness_of_fit > 0.1).astype(
        np.float64
    )
    circadianess_df["is_circadian"] = (circadianess_df.resampled_p_value < 0.05).astype(
        np.float64
    )
    circadianess_df["well_tested_circadianess"] = (
        circadianess_df.is_circadian * circadianess_df.is_good_fit
    )


def extract_fit_parameters(circadianess_df):
    # extract parameters (amplitude, phase, offset) from fit
    amplitude = []
    phase = []
    offset = []
    for p in circadianess_df["parameters"]:
        amplitude.append(p[0])
        phase.append(p[1])
        offset.append(p[2])
    circadianess_df["amplitude"] = amplitude
    circadianess_df["phase"] = phase
    circadianess_df["offset"] = offset
    return circadianess_df


def create_phase_plt_age_df(circadianess_df, phase_shift=12):
    return pd.DataFrame(
        {
            "phase_plt": ((time.map_pi_time_interval_to_24h(circadianess_df["phase"])) + phase_shift).tolist(),
            "Age [days]": circadianess_df["Age [days]"].tolist(),
            "age": circadianess_df["age"].tolist(),
        }
    )


def add_phase_plt_to_df(circadianess_df, fit_type="cosine", time_reference=None):
    if fit_type == "cosine":
        time_shift = 12
    else:
        time_shift = 0
    if time_reference:
        time_shift = circadianess_df["time_reference"]
    circadianess_df["phase_plt"] = time.map_pi_time_interval_to_24h(circadianess_df["phase"]) + time_shift
    circadianess_df["phase_plt"] = circadianess_df["phase_plt"] + np.where(circadianess_df["phase_plt"] < 0, 24, 0)
    if time_reference:
        circadianess_df = circadianess_df[circadianess_df["phase_plt"] >= (circadianess_df["time_reference"] - 12)]
        circadianess_df = circadianess_df[circadianess_df["phase_plt"] < (circadianess_df["time_reference"] + 12)]
    else:
        circadianess_df = circadianess_df[circadianess_df["phase_plt"] >= 0]
        circadianess_df = circadianess_df[circadianess_df["phase_plt"] < 24]
    return circadianess_df
