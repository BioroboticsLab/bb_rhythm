import datetime

import pandas as pd
from bigtree import dataframe_to_tree_by_relation, inorder_iter

def random_sample_bee(interaction_df, n, query=None):
    random_samples = interaction_df.query(query).reset_index().sample(n=n)
    return random_samples


# get interaction_df
interaction_df = pd.read_pickle(path)

# filter overlap
# interaction_df = bb_rhythm.interactions.filter_overlap(interaction_df)

# combine df so all bees are considered as focal
interaction_df = bb_rhythm.interactions.combine_bees_from_interaction_df_to_be_all_focal(interaction_df)

# map phase to 24h
interaction_df["phase_focal"] = ((- 24 * interaction_df["phase_focal"] / (2 * np.pi)) + 12) % 24
interaction_df["phase_non_focal"] = ((- 24 * interaction_df["phase_non_focal"] / (2 * np.pi)) + 12) % 24

# sample random bees
query = '(age_focal < 5) & (phase_focal > 12) & (p_value_focal < 0.05) & (age_focal > 0)'
source_bee_ids = random_sample_bee(interaction_df[
                                       (interaction_df.interaction_start.dt.time < datetime.time(hour=13,
                                                                                                 minute=55)) &
                                       (interaction_df.interaction_start.dt.time >= datetime.time(hour=13,
                                                                                                  minute=45))],
                                   n=1,
                                   query=query,
                                   )
time_threshold = datetime.timedelta(minutes=1)
vel_change_threshold = 0


def create_root_df(source_bee_ids, interaction_df, time_threshold, vel_change_threshold):
    # create root of interaction_tree
    root_df = source_bee_ids[
        ["bee_id_focal", "phase_focal", "age_focal", "r_squared_focal", "interaction_start", "x_pos_focal",
         "y_pos_focal", "vel_change_bee_focal"]]
    root_df.rename(colums={
        "bee_id_focal": "child_bee_id",
        "phase_focal": "phase",
        "age_focal": "age",
        "r_squared_focal": "r_squared",
        "interaction_start": "datetime",
        "x_pos_focal": "x_pos",
        "y_pos_focal": "y_pos",
        "vel_change_bee_focal": "vel_change_parent"
    }, inplace=True)
    root_df["parent_bee_id"] = [None]
    # get child nodes of root
    # also parents should be filtered out as then the child's velocity change should be negative
    root_child_df = interaction_df[
        (interaction_df["bee_id_focal"] == source_bee_ids.bee_id_focal.values[0]) &
        (interaction_df["interaction_start"] <= source_bee_ids["interaction_start"].values[0]) &
        (interaction_df["interaction_start"] > (source_bee_ids["interaction_start"].values[0] + time_threshold)) &
        (interaction_df["vel_change_bee_focal"] > vel_change_threshold)
        ][["bee_id_non_focal", "bee_id_focal", "phase_non_focal", "age_non_focal", "r_squared_non_focal",
           "interaction_start", "x_pos_non_focal", "y_pos_non_focal", "vel_change_bee_focal"]]
    root_child_df.rename(colums={
        "bee_id_focal": "parent_bee_id",
        "bee_id_non_focal": "child_bee_id",
        "phase_non_focal": "phase",
        "age_non_focal": "age",
        "r_squared_non_focal": "r_squared",
        "interaction_start": "datetime",
        "x_pos_non_focal": "x_pos",
        "y_pos_non_focal": "y_pos",
        "vel_change_bee_focal": "vel_change_parent"
    }, inplace=True)
    tree_df = pd.concat([root_df, root_child_df])
    return tree_df

def create_child_df(child_bee_ids, interaction_df, time_threshold, vel_change_threshold):
    # get child nodes of root
    # also parents should be filtered out as then the child's velocity change should be negative
    root_child_df = interaction_df[
        (interaction_df["bee_id_focal"] == child_bee_ids.node_name) &
        (interaction_df["interaction_start"] <= child_bee_ids["datetime"]) &
        (interaction_df["interaction_start"] > (child_bee_ids["datetime"] + time_threshold)) &
        (interaction_df["vel_change_bee_focal"] > vel_change_threshold)
        ][["bee_id_non_focal", "bee_id_focal", "phase_non_focal", "age_non_focal", "r_squared_non_focal",
           "interaction_start", "x_pos_non_focal", "y_pos_non_focal", "vel_change_bee_focal"]]
    root_child_df.rename(colums={
        "bee_id_focal": "parent_bee_id",
        "bee_id_non_focal": "child_bee_id",
        "phase_non_focal": "phase",
        "age_non_focal": "age",
        "r_squared_non_focal": "r_squared",
        "interaction_start": "datetime",
        "x_pos_non_focal": "x_pos",
        "y_pos_non_focal": "y_pos",
        "vel_change_bee_focal": "vel_change_parent"
    }, inplace=True)
    return root_child_df

tree_df = create_root_df(source_bee_ids, interaction_df, time_threshold, vel_change_threshold)
tree = dataframe_to_tree_by_relation(tree_df, child_col="child_bee_id", parent_col="parent_bee_id", attribute_cols=["phase", "age", "r_squared", "datetime", "x_pos", "y_pos", "vel_change_parent"])

for child in tree.root.children:
    subtree_df = create_child_df(child, interaction_df, time_threshold, vel_change_threshold)