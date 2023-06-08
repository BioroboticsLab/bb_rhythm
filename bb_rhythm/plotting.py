from matplotlib import rcParams
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_body_location_of_interactions(vel_change_matrix_df, plot_dir=None, imshow=False, annotate=False, ax=None):
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

    ax.set_title('Velocity change per body location')
    if imshow:
        plt.imshow()
    if plot_dir:
        plt.savefig(plot_dir)


def transform_interaction_df_to_vel_change_matrix_df(vel_change_df_trans):
    # group velocity changes by coordinates
    vel_change_matrix_df = vel_change_df_trans.groupby(['focal0_x_trans', 'focal0_y_trans'])[
        "vel_change_bee_focal"].agg(
        [('count', 'count'), ('vel_change_bee_focal', 'median')]).reset_index()
    vel_change_matrix_df.rename(columns={'focal0_x_trans': "x", 'focal0_y_trans': "y"})
    return vel_change_matrix_df


def transform_interaction_matrix_to_df(vel_change_matrix, count_matrix):
    vel_change_matrix_df = pd.DataFrame(vel_change_matrix).stack().rename_axis(['y', 'x']).reset_index(
        name='vel_change_bee_focal')
    count_matrix_df = pd.DataFrame(count_matrix).stack().rename_axis(['y', 'x']).reset_index(
        name='count')
    return pd.merge(vel_change_matrix_df, count_matrix_df, on=["y", "x"], how="outer")
