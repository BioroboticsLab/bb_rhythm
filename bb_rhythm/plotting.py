from matplotlib import rcParams
import matplotlib.pyplot as plt
import re
import pandas as pd
from . import interactions


def plot_body_location_of_interactions(vel_change_matrix_df, plot_dir=None, imshow=False, annotate=False, ax=None):
    # plot settings
    rcParams.update({'figure.autolayout': True})
    plt.rcParams['axes.facecolor'] = 'white'

    # create figure
    if ax is None:
        fig, ax = plt.figure(figsize=(12, 12))

    # plot
    # set variables
    scale_factor = vel_change_matrix_df["count"].max()/1500
    scatter = plt.scatter(x=vel_change_matrix_df.x, y=vel_change_matrix_df.y,
                          c=vel_change_matrix_df.vel_change_bee_focal, cmap="viridis",
                          s=vel_change_matrix_df["count"] / scale_factor, vmin=-0.02, vmax=0.16)

    # add annotations one by one with a loop
    if annotate:
        vel_change_matrix_df.vel_change_bee_focal = vel_change_matrix_df.vel_change_bee_focal.round(2)
        for line in range(0, vel_change_matrix_df.shape[0]):
            plt.text(vel_change_matrix_df.x[line], vel_change_matrix_df.y[line],
                     vel_change_matrix_df.vel_change_bee_focal[line], horizontalalignment='center', color='black')

    # produce a legend with the unique colors from the scatter
    legend1 = ax.legend(*scatter.legend_elements(),
                        loc="upper right", title="velocity change", prop={'size': 13})
    ax.add_artist(legend1)
    ax.legend()

    # produce a legend with a cross-section of sizes from the scatter
    handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6, num=4)
    for i in range(len(labels)):
        res = re.split('{|}', labels[i])
        labels[i] = "%s{%d}%s" % (res[0], int(res[1]) * scale_factor, res[2])
        legend2 = ax.legend(handles, labels, loc="lower right", title="count", prop={'size': 13})

    # label plot
    plt.xlabel('x position')
    plt.ylabel('y position')

    ax.set_title('Velocity change per body location')
    if imshow:
        plt.imshow()
    if plot_dir:
        plt.savefig(plot_dir)


def transform_interaction_df_to_vel_change_matrix_df(vel_change_df_trans):
    # group velocity changes by coordinates
    vel_change_matrix_df = vel_change_df_trans.groupby(['focal0_x_trans', 'focal0_y_trans'])["vel_change_bee_focal"].agg(
        [('count', 'count'), ('vel_change_bee_focal', 'median')]).reset_index()
    vel_change_matrix_df.rename(columns={'focal0_x_trans': "x", 'focal0_y_trans': "y"})
    return vel_change_matrix_df


def transform_interaction_matrix_to_df(vel_change_matrix, count_matrix):
    vel_change_matrix_df = pd.DataFrame(vel_change_matrix).stack().rename_axis(['y', 'x']).reset_index(name='vel_change_bee_focal')
    count_matrix_df = pd.DataFrame(count_matrix).stack().rename_axis(['y', 'x']).reset_index(
        name='count')
    return pd.merge(vel_change_matrix_df, count_matrix_df, on=["y", "x"], how="outer")
    