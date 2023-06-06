from matplotlib import rcParams
import matplotlib.pyplot as plt
import re

import interactions


def plot_body_location_of_interaction(df, trans=True, plot_dir=None, imshow=False, annotate=False, ax=None):
    # plot settings
    rcParams.update({'figure.autolayout': True})
    plt.rcParams['axes.facecolor'] = 'white'

    # create figure
    if ax is None:
        fig, ax = plt.figure(figsize=(12, 12))

    # transform coordinates
    if not trans:
        df = interactions.apply_transformation(df)

    # group velocity changes by coordinates
    df_median_focal = df.groupby(['focal0_x_trans', 'focal0_y_trans'])["vel_change_bee_focal"].agg(
        [('count', 'count'), ('vel_change_bee_focal', 'median')]).reset_index()

    # plot
    # set variables
    scale_factor = 80
    scatter = plt.scatter(x=df_median_focal.focal0_x_trans, y=df_median_focal.focal0_y_trans,
                          c=df_median_focal.vel_change_bee_focal, cmap="viridis",
                          s=df_median_focal["count"] / scale_factor, vmin=-0.02, vmax=0.16)

    # add annotations one by one with a loop
    if annotate:
        df_median_focal.vel_change_bee_focal = df_median_focal.vel_change_bee_focal.round(2)
        for line in range(0, df_median_focal.shape[0]):
            plt.text(df_median_focal.focal0_x_trans[line], df_median_focal.focal0_y_trans[line],
                     df_median_focal.vel_change_bee_focal[line], horizontalalignment='center', color='black')

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
        