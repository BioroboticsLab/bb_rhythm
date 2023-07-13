import numpy as np
from . import time


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


def create_phase_plt_age_df(phase_shift=12):
    return pd.DataFrame(
        {
            "phase_plt": ((time.map_pi_time_interval_to_24h(circadianess_df_plt["phase"])) + phase_shift).tolist(),
            "Age [days]": circadianess_df_plt["Age [days]"].tolist(),
            "age": circadianess_df_plt["age"].tolist(),
        }
    )


def add_phase_plt_to_df(circadianess_df, fit_type="cosine", time_reference=None):
    if fit_type == "cosine":
        time_shift = 12
    else:
        time_shift = 0
    if time_reference:
        time_shift = circadianess_df["time_reference"]
    circadianess_df["phase_plt"] = time.map_pi_time_interval_to_24h(circadianess_df_plt["phase"]) + time_shift
    if time_reference:
        circadianess_df = circadianess_df[circadianess_df["phase_plt"] >= (circadianess_df["time_reference"] - 12)]
        circadianess_df = circadianess_df[circadianess_df["phase_plt"] < (circadianess_df["time_reference"] + 12)]
    else:
        circadianess_df = circadianess_df[circadianess_df["phase_plt"] >= 0]
        circadianess_df = circadianess_df[circadianess_df["phase_plt"] < 24]
    return circadianess_df
