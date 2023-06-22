import numpy as np


def create_agg_circadian_df(circadianess_df, column="age_bins", agg_func='mean'):
    return circadianess_df.pivot_table(index=["date"], columns=column, values="well_tested_circadianess",
                                       aggfunc=agg_func).melt(ignore_index=False).reset_index()


def create_mean_count_circadianess_per_day_df(circadianess_df, column="age_bins"):
    circadianess_df_mean = create_agg_circadian_df(circadianess_df, column=column)
    circadianess_df_count = create_agg_circadian_df(circadianess_df, column=column, agg_func='count')
    circadianess_df = circadianess_df_mean.rename(columns={"value": "mean"})
    circadianess_df["count"] = circadianess_df_count["value"]

    # filter counts lower than 0.05 counts out
    circadianess_df = circadianess_df[circadianess_df["count"] > circadianess_df["count"].quantile(q=0.05)]
    return circadianess_df


def calculate_well_tested_circadianess(circadianess_df):
    circadianess_df["is_good_fit"] = (circadianess_df.goodness_of_fit > 0.1).astype(np.float64)
    circadianess_df["is_circadian"] = (circadianess_df.resampled_p_value < 0.05).astype(np.float64)
    circadianess_df["well_tested_circadianess"] = (circadianess_df.is_circadian * circadianess_df.is_good_fit)
