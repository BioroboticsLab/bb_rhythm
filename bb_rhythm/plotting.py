from matplotlib import rcParams
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import datetime
from scipy.ndimage import gaussian_filter1d

from . import utils


def plot_body_location_of_interactions(vel_change_matrix_df, plot_dir=None, imshow=False, annotate=False, ax=None, title_extra=None):
    # plot settings
    rcParams.update({'figure.autolayout': True})
    plt.rcParams['axes.facecolor'] = 'white'
    plt.tight_layout()

    # create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 12))

    # plot
    sns.scatterplot(data=vel_change_matrix_df, x="x", y="y",
                    hue="vel_change_bee_focal", palette="viridis",
                    hue_norm=(vel_change_matrix_df["vel_change_bee_focal"][
                                  vel_change_matrix_df["vel_change_bee_focal"] != 0.].quantile(0.05),
                              vel_change_matrix_df["vel_change_bee_focal"][
                                  vel_change_matrix_df["vel_change_bee_focal"] != 0.].quantile(0.95)),
                    size="count",
                    sizes=(0, 500),
                    ax=ax
                    )

    # add annotations one by one with a loop
    if annotate:
        vel_change_matrix_df.vel_change_bee_focal = vel_change_matrix_df.vel_change_bee_focal.round(2)
        for line in range(0, vel_change_matrix_df.shape[0]):
            ax.text(vel_change_matrix_df.x[line], vel_change_matrix_df.y[line],
                    vel_change_matrix_df.vel_change_bee_focal[line], horizontalalignment='center',
                    color='black')

    # legend settings
    handles, labels = ax.get_legend_handles_labels()
    labels = ["Velocity change" if item == "vel_change_bee_focal" else item for item in labels]
    labels = ["Count" if item == "count" else item for item in labels]
    ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1))

    # label plot
    plt.xlabel('x position')
    plt.ylabel('y position')

    if title_extra is None:
        ax.set_title('Velocity change per body location')
    else:
        ax.set_title('Velocity change per body location\n%s' % str(title_extra))
    if imshow:
        plt.imshow()
    if plot_dir:
        plt.savefig(plot_dir)


def transform_interaction_df_to_vel_change_matrix_df(vel_change_df_trans):
    # group velocity changes by coordinates
    vel_change_matrix_df = vel_change_df_trans.groupby(['focal0_x_trans', 'focal0_y_trans'])[
        "vel_change_bee_focal"].agg(
        [('count', 'count'), ('vel_change_bee_focal', 'median')]).reset_index()
    vel_change_matrix_df.rename(columns={'focal0_x_trans': "x", 'focal0_y_trans': "y"}, inplace=True)
    return vel_change_matrix_df


def transform_interaction_matrix_to_df(vel_change_matrix, count_matrix):
    vel_change_matrix_df = pd.DataFrame(vel_change_matrix).stack().rename_axis(['y', 'x']).reset_index(
        name='vel_change_bee_focal')
    count_matrix_df = pd.DataFrame(count_matrix).stack().rename_axis(['y', 'x']).reset_index(
        name='count')
    return pd.merge(vel_change_matrix_df, count_matrix_df, on=["y", "x"], how="outer")


def plot_velocity_over_time_with_weather(velocity_df, weather_df, dt_from, dt_to, age_map=True,
                                         age_map_step_size=5, smoothing=False, rounded=False, round_time="60min",
                                         age_bins=None, imshow=False, image_path=None):
    # create figure
    fig, axs = plt.subplots(3, 1, figsize=(16, 10), sharex=True, gridspec_kw={'height_ratios': [8, 1, 1]})
    fig.suptitle("Mean movement speed over time")
    fig.tight_layout()

    # plot velocities
    plot_velocity_per_age_group(velocity_df, axs[0], dt_from=dt_from, dt_to=dt_to, age_map=age_map,
                                age_map_step_size=age_map_step_size, age_bins=age_bins, smoothing=smoothing,
                                rounded=rounded, round_time=round_time)

    # plot weather
    plot_weather_data(weather_df, axs, dt_from, dt_to)

    # plot settings
    plt.xticks(rotation=90)
    plt.xlabel("Time")

    # show and save plot
    if imshow:
        plt.show()

    if image_path is not None:
        plt.savefig(image_path)


def plot_weather_data(weather_df, axs, dt_from, dt_to):
    # subset time
    weather_df = weather_df[(weather_df["date"] >= dt_from) & (weather_df["date"] < dt_to)]

    # plot temperature
    plot_weather_subplot(axs[1], dt_from, dt_to, 'Temp [C°]', "temperature_air_mean_200", weather_df)

    # plot wind speed
    plot_weather_subplot(axs[2], dt_from, dt_to, 'Wind [m/s]', "wind_speed_mean", weather_df)


def plot_weather_subplot(ax, dt_from, dt_to, y_label, column, weather_df):
    ax.plot(weather_df["date"], weather_df[column])
    ax.set_ylabel(y_label)
    ax.grid(True)

    # add grey bars for nighttime
    add_grey_nighttime_bars(ax, weather_df)
    ax.set_xlim(xmin=dt_from, xmax=dt_to)


def add_grey_nighttime_bars(ax, df):
    for day in np.unique([day.date() for day in df.date]):
        day = datetime.datetime.combine(day, df.date.iloc[0].to_pydatetime().time())
        ax.axvspan(day, day + datetime.timedelta(hours=6), facecolor='lightgrey', alpha=0.5)
        ax.axvspan(day + datetime.timedelta(hours=18), day + datetime.timedelta(hours=24), facecolor='lightgrey',
                   alpha=0.5)


def plot_velocity_per_age_group(time_age_velocity_df, ax, dt_from, dt_to, age_map=False, age_bins=None,
                                age_map_step_size=5, smoothing=False, rounded=False, round_time="60min"):
    # remove NaNs
    time_age_velocity_df = time_age_velocity_df[~pd.isnull(time_age_velocity_df.age)]

    # subset time
    time_age_velocity_df = time_age_velocity_df[
        (time_age_velocity_df["date"] >= dt_from) & (time_age_velocity_df["date"] < dt_to)]

    if rounded:
        time_age_velocity_df = round_time_age_velocity_df(round_time, time_age_velocity_df)

    # add human-readable age bins to df
    sorted_by = "age"
    if age_map:
        sorted_by = "age_bins"
        utils.add_age_bins(time_age_velocity_df, age_map_step_size, age_bins)
        time_age_velocity_df = time_age_velocity_df[time_age_velocity_df["age_bins"] != "Nan"]

    # create color palette
    palette = create_age_color_palette(age_bins, sorted_by, time_age_velocity_df)

    # smooth lines per group, but still plot transparently 95% confidence interval for non-smoothed lines
    if smoothing:
        plot_smoothed_age_velocity_over_time(ax, palette, sorted_by, time_age_velocity_df)

    # plot non-smoothed lines
    else:
        plot_non_smoothed_age_velocity_over_time(ax, palette, sorted_by, time_age_velocity_df)

    # add grey bars for nighttime
    add_grey_nighttime_bars(ax, time_age_velocity_df)
    """ for day in np.unique([day.date() for day in time_age_velocity_df.time]):  # TODO: start always at 0:00:00
        day = datetime.datetime.combine(day, time_age_velocity_df.time.iloc[0].to_pydatetime().time())
        ax.axvspan(day, day + datetime.timedelta(hours=6), facecolor='lightgrey', alpha=0.5)
        ax.axvspan(day + datetime.timedelta(hours=18), day + datetime.timedelta(hours=24), facecolor='lightgrey',
                   alpha=0.5)"""
    # axis settings
    ax.set_ylabel('Mean movement speed [cm/s]')
    ax.set_xlim(xmin=dt_from, xmax=dt_to)
    ax.legend(loc="upper right", title="Age [days]")


def plot_non_smoothed_age_velocity_over_time(ax, palette, sorted_by, time_age_velocity_df):
    sns.lineplot(data=time_age_velocity_df, x="date", y="velocity", hue=sorted_by, ax=ax, palette=palette)


def plot_smoothed_age_velocity_over_time(ax, palette, sorted_by, time_age_velocity_df):
    # get smoothed velocities
    get_smoothed_velocities(sorted_by, time_age_velocity_df)

    # plot
    sns.lineplot(data=time_age_velocity_df, x="date", y="velocity", hue=sorted_by, linewidth=0, legend=False, ax=ax,
                 palette=palette)
    sns.lineplot(data=time_age_velocity_df, x="date", y="velocity_smoothed", errorbar=None, hue=sorted_by, ax=ax,
                 palette=palette)


def get_smoothed_velocities(sorted_by, time_age_velocity_df):
    time_age_velocity_df["velocity_smoothed"] = time_age_velocity_df["velocity"]
    for age_bin in time_age_velocity_df[sorted_by].unique():
        time_age_velocity_df["velocity_smoothed"][time_age_velocity_df[sorted_by] == age_bin] = gaussian_filter1d(
            time_age_velocity_df["velocity"][time_age_velocity_df[sorted_by] == age_bin], sigma=4)


def create_age_color_palette(age_map, sorted_by, time_age_velocity_df):
    if age_map:
        palette = sns.color_palette("viridis", len(time_age_velocity_df[sorted_by].unique()) * 4)
        palette = [palette[i * 4] for i in range(len(time_age_velocity_df[sorted_by].unique()))]
    else:
        palette = sns.color_palette("viridis", len(time_age_velocity_df[sorted_by].unique()))
    return palette


def round_time_age_velocity_df(round_time, time_age_velocity_df):
    time_age_velocity_df["date"] = time_age_velocity_df['date'].dt.round(round_time)
    time_age_velocity_df = time_age_velocity_df.groupby(["date", "age"])['velocity'].mean().reset_index()
    return time_age_velocity_df
