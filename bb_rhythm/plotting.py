from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns
import numpy as np
import numpy.ma as ma
import datetime
from scipy.ndimage import gaussian_filter1d

from . import utils
from . import rhythm


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
                                         age_bins=None, imshow=False, image_path=None, y_lim=None):
    # create figure
    fig, axs = plt.subplots(3, 1, figsize=(16, 10), sharex=True, gridspec_kw={'height_ratios': [8, 1, 1]})
    fig.suptitle("Mean movement speed over time")
    fig.tight_layout()

    # plot velocities
    plot_velocity_per_age_group(velocity_df, axs[0], dt_from=dt_from, dt_to=dt_to, age_map=age_map,
                                age_map_step_size=age_map_step_size, age_bins=age_bins, smoothing=smoothing,
                                rounded=rounded, round_time=round_time, y_lim=y_lim)

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
                                age_map_step_size=5, smoothing=False, rounded=False, round_time="60min", y_lim=None):
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
    if y_lim is not None:
        ax.set_ylim(ymin=y_lim[0], ymax=y_lim[1])
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


def plot_boxplot(circadianess_df, ax, x="age_bins"):
    sns.boxplot(circadianess_df,
                ax=ax,
                x=x,
                y="mean",
                width=0.5,
                flierprops={"marker": "o", "color": (0, 0, 0, 0)},
                boxprops={"facecolor": (0, 0, 0, 0), "edgecolor": "blue"},
                whiskerprops={"color": "blue"},
                medianprops={"color": "green", "linewidth": 1},
                showfliers=True,
                capprops={"color": "blue"}
            )


def set_fig_props_circadianess_per_age_plot(fig):
    fig.supxlabel('Age [days]')
    fig.supylabel('')
    fig.suptitle('Fraction of bees sig. different from shuffled data\n and the shuffled data was chi²')


def plot_raincloudplot(circadianess_df, ax, x="age_bins", hue_norm=None):
    # Create violin plots without mini-boxplots inside.
    sns.violinplot(data=circadianess_df, x=x, y="mean", color='mediumslateblue', cut=0, inner=None, ax=ax)

    # Clip the lower half of each violin.
    for item in ax.collections:
        x0, y0, width, height = item.get_paths()[0].get_extents().bounds
        item.set_clip_path(plt.Rectangle((x0, y0), width/2, height,#/2,
                           transform=ax.transData))

    # Create strip plots with partially transparent points of different colors depending on the group.
    num_items = len(ax.collections)
    sns.stripplot(data=circadianess_df, x=x, y="mean", hue="count", size=7, ax=ax, hue_norm=hue_norm)

    # Shift each strip plot strictly below the correponding volin.
    for item in ax.collections[num_items:]:
       for pos in item.get_offsets():
          pos[0] = pos[0] + 0.125

    # Create narrow boxplots on top of the corresponding violin and strip plots, with thick lines, the mean values, without the outliers.
    sns.boxplot(data=circadianess_df, x=x, y="mean", width=0.25,
                showfliers=False, showmeans=True,
                meanprops=dict(markerfacecolor='lightgreen',
                               markersize=5, zorder=3),
                boxprops=dict(facecolor=(0,0,0,0),
                              linewidth=3, zorder=3),
                whiskerprops=dict(linewidth=3),
                capprops=dict(linewidth=3),
                medianprops=dict(linewidth=3), ax=ax)


def plot_violin_swarm_plot(circadianess_df, ax, x="age_bins", size_norm=None, date_ann=False, count_ann=False):
    # get coordinates from swarm plot
    sns.swarmplot(data=circadianess_df, x=x, y="mean", ax=ax, size=7.)
    coordinates = [ma.getdata(ax.collections[i].get_offsets()) for i in range(len(ax.collections))]
    ax.clear()

    # plot violins
    sns.violinplot(data=circadianess_df, x=x, y="mean", palette=["lightgrey"], cut=0, saturation=0.5, scale="width", orient='v', ax=ax, inner="quartile", alpha=0.5)
    circadianess_df['count'][circadianess_df['count'] == "nan"] = 0.

    # plot scatter
    if date_ann:
        sns.scatterplot(data=circadianess_df, x=x, y="mean", size="count", hue="date", palette="viridis", size_norm=size_norm, ax=ax)
    else:
        sns.scatterplot(data=circadianess_df, x=x, y="mean", size="count", size_norm=size_norm, ax=ax)

    # annotate number of counts
    if count_ann:
        for x, y, count in zip(circadianess_df[x], circadianess_df["mean"], circadianess_df["count"]):
            ax.annotate(count, xy=(x, y))

    # use coordinates from swarm plot to spread scatters
    ax_index = find_ax_collection_for_plotting(ax)
    ax.collections[ax_index].set_offsets([item for sublist in coordinates for item in sublist])


def find_ax_collection_for_plotting(ax):
    ax_index = 0
    for i in range(len(ax.collections)):
        if ax.collections[i].get_offsets().size != 0:
            ax_index = i
    return ax_index


def plot_circadianess_per_age_group(circadianess_df, plot_type='boxplot', young=False, file_path=None, age_map_step_size=5, age_bins=None):
    # calculate well tested circadianess
    rhythm.calculate_well_tested_circadianess(circadianess_df)

    # create age map and bins
    utils.add_age_bins(circadianess_df, age_map_step_size, age_bins)
    circadianess_df = circadianess_df[circadianess_df["age_bins"] != "Nan"]

    circadianess_df_young = None
    if young:
        # get subset of young bees
        circadianess_df_young = circadianess_df[circadianess_df["age"] <= 5]

        # Create dataframe with aggregated mean and count of well tested circadianess per day
        circadianess_df_young = rhythm.create_mean_count_circadianess_per_day_df(circadianess_df_young, column="age")

    # Create dataframe with aggregated mean and count of well tested circadianess per day
    circadianess_df = rhythm.create_mean_count_circadianess_per_day_df(circadianess_df)

    # get count norm for shared legend
    min_count = circadianess_df["count"].min()
    max_count = circadianess_df["count"].max()

    plot_agg_well_tested_circadianess_per_bee_age(circadianess_df, circadianess_df_young, max_count, min_count,
                                                  plot_type, young, file_path)


def plot_agg_well_tested_circadianess_per_bee_age(circadianess_df, circadianess_df_young, max_count=None, min_count=None, plot_type="boxplot", young=False, file_path=None):
    if young:
        # create figure and subplots
        fig, (ax0, ax1) = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [3, 7]}, figsize=(16, 8))

        # plot
        plot_agg_circadianess_according_plot_type(ax=ax0, circadianess_df=circadianess_df_young, norm=(min_count, max_count), plot_type=plot_type, x="age")

        # set axis properties
        if ax0.get_legend():
            ax0.legend().set_visible(False)
        set_ax_props_circadianess_per_age_group_plot(ax0)

    else:
        # create figure and subplots
        fig, ax1 = plt.subplots(1, 1, figsize=(15, 10))

    # plot
    plot_agg_circadianess_according_plot_type(ax=ax1, circadianess_df=circadianess_df, norm=(min_count, max_count), plot_type=plot_type)

    # set axis properties
    if ax1.get_legend():
        ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title='Samples per day')
    set_ax_props_circadianess_per_age_group_plot(ax1)

    # set label properties
    set_fig_props_circadianess_per_age_plot(fig)

    # set plt properties
    plt.ylim(0., 1.1)
    plt.tight_layout()
    if file_path:
        plt.savefig(file_path)


def plot_agg_circadianess_according_plot_type(ax=None, circadianess_df=None, norm=None, plot_type="boxplot", x="age_bins", date_ann=False, count_ann=False):
    if plot_type == "boxplot":
        plot_boxplot(circadianess_df, x=x, ax=ax)
    elif plot_type == "violin_swarm":
        plot_violin_swarm_plot(circadianess_df, ax=ax, x=x, size_norm=norm, date_ann=date_ann, count_ann=count_ann)
    elif plot_type == "raincloud":
        plot_raincloudplot(circadianess_df, ax=ax, x=x, hue_norm=norm)
    else:
        raise ValueError("Incorrect plot type. Possible types: ['boxplot', 'violin_swarm', 'raincloud']")


def set_ax_props_circadianess_per_age_group_plot(ax):
    # set step size ax ticks
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))

    # set ax labels
    ax.set_xlabel('')
    ax.set_ylabel('')
