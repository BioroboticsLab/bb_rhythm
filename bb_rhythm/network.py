import datetime
import pandas as pd
from treelib import Tree
import treelib


class Interaction(object):
    #TODO: make uniform series or dataframe as input
    def __init__(self, interaction_df_row):
        if type(interaction_df_row) == pd.DataFrame:
            self.phase = interaction_df_row["phase"].iloc[0]
            self.datetime = interaction_df_row["datetime"].iloc[0]
            self.x_pos = interaction_df_row["x_pos"].iloc[0]
            self.y_pos = interaction_df_row["y_pos"].iloc[0]
            self.vel_change_parent = interaction_df_row["vel_change_parent"].iloc[0]
            self.r_squared = interaction_df_row["r_squared"].iloc[0]
            self.age = interaction_df_row["age"].iloc[0]
        else:
            self.phase = interaction_df_row["phase"]
            self.datetime = interaction_df_row["datetime"]
            self.x_pos = interaction_df_row["x_pos"]
            self.y_pos = interaction_df_row["y_pos"]
            self.vel_change_parent = interaction_df_row["vel_change_parent"]
            self.r_squared = interaction_df_row["r_squared"]
            self.age = interaction_df_row["age"]


def add_root(
    tree, source_bee_ids, interaction_df, time_threshold, vel_change_threshold
):
    # create root of interaction_tree
    root_df = source_bee_ids[
        [
            "bee_id_focal",
            "phase_focal",
            "age_focal",
            "r_squared_focal",
            "interaction_start",
            "x_pos_start_focal",
            "y_pos_start_focal",
            "vel_change_bee_focal",
        ]
    ]
    root_df.rename(
        columns={
            "phase_focal": "phase",
            "age_focal": "age",
            "r_squared_focal": "r_squared",
            "interaction_start": "datetime",
            "x_pos_start_focal": "x_pos",
            "y_pos_start_focal": "y_pos",
            "vel_change_bee_focal": "vel_change_parent",
        },
        inplace=True,
    )
    tree.create_node(
        int(source_bee_ids.bee_id_focal),
        "%d_%s"
        % (
            int(source_bee_ids.bee_id_focal),
            str(source_bee_ids.interaction_start),
        ),
        data=Interaction(root_df.head(1)),
    )

    # get child nodes of root
    # also parents should be filtered out as then the child's velocity change should be negative
    root_child_df = interaction_df[
        (interaction_df["bee_id_focal"] == source_bee_ids.bee_id_focal.iloc[0])
        & (
            interaction_df["interaction_start"]
            <= source_bee_ids["interaction_start"].iloc[0]
        )
        & (
            interaction_df["interaction_start"]
            > (source_bee_ids["interaction_start"].iloc[0] - time_threshold)
        )
        & (interaction_df["vel_change_bee_focal"] > vel_change_threshold)
    ][
        [
            "bee_id_non_focal",
            "bee_id_focal",
            "phase_non_focal",
            "age_non_focal",
            "r_squared_non_focal",
            "interaction_start",
            "x_pos_start_non_focal",
            "y_pos_start_non_focal",
            "vel_change_bee_focal",
        ]
    ]
    root_child_df.rename(
        columns={
            "bee_id_focal": "parent_bee_id",
            "bee_id_non_focal": "child_bee_id",
            "phase_non_focal": "phase",
            "age_non_focal": "age",
            "r_squared_non_focal": "r_squared",
            "interaction_start": "datetime",
            "x_pos_start_non_focal": "x_pos",
            "y_pos_start_non_focal": "y_pos",
            "vel_change_bee_focal": "vel_change_parent",
        },
        inplace=True,
    )
    for index, row in root_child_df.iterrows():
        tree.create_node(
            int(row["child_bee_id"]),
            "%d_%s" % (int(row["child_bee_id"]), str(row["datetime"])),
            parent="%d_%s"
            % (
                int(source_bee_ids.bee_id_focal),
                str(source_bee_ids.interaction_start),
            ),
            data=Interaction(row),
        )


def add_children(tree, parent, interaction_df, time_threshold, vel_change_threshold):
    # get child nodes of root
    # also parents should be filtered out as then the child's velocity change should be negative
    root_child_df = interaction_df[
        (interaction_df["bee_id_focal"] == parent.tag)
        & (interaction_df["interaction_start"] < parent.data.datetime)
        & (
            interaction_df["interaction_start"]
            >= (parent.data.datetime - time_threshold)
        )
        & (interaction_df["vel_change_bee_focal"] > vel_change_threshold)
    ][
        [
            "bee_id_non_focal",
            "phase_non_focal",
            "age_non_focal",
            "r_squared_non_focal",
            "interaction_start",
            "x_pos_start_non_focal",
            "y_pos_start_non_focal",
            "vel_change_bee_focal",
        ]
    ]
    root_child_df.rename(
        columns={
            "phase_non_focal": "phase",
            "age_non_focal": "age",
            "r_squared_non_focal": "r_squared",
            "interaction_start": "datetime",
            "x_pos_start_non_focal": "x_pos",
            "y_pos_start_non_focal": "y_pos",
            "vel_change_bee_focal": "vel_change_parent",
        },
        inplace=True,
    )
    for index, row in root_child_df.iterrows():
        try:
            tree.create_node(
                int(row["bee_id_non_focal"]),
                "%d_%s" % (int(row["bee_id_non_focal"]), str(row["datetime"])),
                parent=parent.identifier,
                data=Interaction(row),
            )
        except treelib.exceptions.DuplicatedNodeIdError:
            tree.create_node(
                int(row["bee_id_non_focal"]),
                "%d_%s_%d" % (int(row["bee_id_non_focal"]), str(row["datetime"]), 1),
                parent=parent.identifier,
                data=Interaction(row),
            )


def construct_interaction_tree_recursion(
    tree, interaction_df, time_threshold, vel_change_threshold, time_stop
):
    for node in [node for node in tree.leaves() if node.data.datetime > time_stop]:
        add_children(tree, node, interaction_df, time_threshold, vel_change_threshold)
        if not node.is_leaf():
            construct_interaction_tree_recursion(
                tree, interaction_df, time_threshold, vel_change_threshold, time_stop
            )


def create_interaction_tree(
    source_bee_ids, interaction_df, time_threshold, vel_change_threshold, time_stop
):
    tree = Tree()
    add_root(tree, source_bee_ids, interaction_df, time_threshold, vel_change_threshold)
    construct_interaction_tree_recursion(
        tree, interaction_df, time_threshold, vel_change_threshold, time_stop
    )
    return tree
