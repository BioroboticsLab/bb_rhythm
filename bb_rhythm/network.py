import datetime
import pandas as pd
from treelib import Tree
import treelib
from slurmjob import SLURMJob

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
        not_labeled = True
        ms = 0
        while not_labeled:
            try:
                tree.create_node(
                    int(row["bee_id_non_focal"]),
                    "%d_%s" % (int(row["bee_id_non_focal"]), str(row["datetime"] + datetime.timedelta(microseconds=ms))),
                    parent=parent.identifier,
                    data=Interaction(row),
                )
                not_labeled = False
            except treelib.exceptions.DuplicatedNodeIdError:
                ms += 1


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


def tree_to_path_df(slurm_job: SLURMJob) -> pd.DataFrame:
    path_df = pd.DataFrame()
    tree_id = 0
    for kwarg, result in slurm_job.items(ignore_open_jobs=True):
        path_id = 0
        for path in result.paths_to_leaves():
            i = 0
            is_root = True
            for node in path:
                if i > 0:
                    is_root = False
                node = result.get_node(node)
                if not is_root:
                    parent = [result.get_node(node.predecessor(result.identifier)).tag]
                    time_gap = [result.get_node(node.predecessor(result.identifier)).data.datetime - node.data.datetime]
                else:
                    parent = [None]
                    time_gap = [datetime.timedelta(minutes=0)]
                path_df = pd.concat([path_df, pd.DataFrame({
                    "bee_id": [node.tag],
                    "datetime": [node.data.datetime],
                    "phase": [node.data.phase],
                    "x_pos": [node.data.x_pos],
                    "y_pos": [node.data.y_pos],
                    "vel_change_parent": [node.data.vel_change_parent],
                    "r_squared": [node.data.r_squared],
                    "age": [node.data.age],
                    "is_root": [is_root],
                    "depth": [i],
                    "is_leaf": [node.is_leaf()],
                    "n_children": [len(node.successors(result.identifier))],
                    "parent": parent,
                    "tree_id": [tree_id],
                    "time_gap": time_gap,
                    "path_id": [path_id],
                })])
                i += 1
            path_id += 1
        tree_id += 1
    return path_df
