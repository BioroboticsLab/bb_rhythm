# imports for run job
import datetime
import numpy as np
import itertools
import pandas as pd
from collections import defaultdict
import cv2

import bb_behavior.db


def cluster_interactions_over_time(iterable, min_gap_size=datetime.timedelta(seconds=1), min_event_duration=None,
                                   y="y", time="time", color="color", loc_info_0="loc_info_0",
                                   loc_info_1="loc_info_1", fill_gaps=True):
    """Slightly modified version of bb_behavior.plot.time.plot_timeline()"""
    # y=pair_string, time=timestamp, color="interaction", loc_info_0=location_parameters_bee0,
    #                                loc_info_1=location_parameters_bee1

    last_x_for_y = defaultdict(list)

    for timepoint in iterable:
        dt, y_value, category, location_info_0, location_info_1 = timepoint[time], timepoint[y], timepoint[color], \
            timepoint[loc_info_0], timepoint[loc_info_1]
        last_x = last_x_for_y[y_value]

        def should_overwrite_last_event():
            if not last_x_for_y[y_value]:
                return False
            last = last_x_for_y[y_value][-1]
            return min_event_duration and (last["dt_end"] - last["dt_start"]) < min_event_duration

        def push():
            new_event_data = dict(
                Task=y_value,
                Resource=category,
                dt_start=dt,
                dt_end=dt,
                location_info_0=location_info_0,
                location_info_1=location_info_1,
                location_info_0_end=location_info_0,
                location_info_1_end=location_info_1
            )
            if should_overwrite_last_event():
                # Overwrite.
                last_x_for_y[y_value][-1] = new_event_data
            else:
                last_x_for_y[y_value].append(new_event_data)

        if not last_x or (category != last_x[-1]["Resource"]):
            # case no end point
            push()
            continue
        last_x = last_x[-1]

        delay = dt - last_x["dt_end"]
        if min_gap_size is not None and delay > min_gap_size:
            push()
            continue

        last_x["dt_end"] = dt
        last_x["location_info_0_end"] = location_info_0
        last_x["location_info_1_end"] = location_info_1

    for _, sessions in last_x_for_y.items():
        if len(sessions) == 0:
            continue
        # Check if last session doesn't meet min length criteria.
        last = sessions[-1]
        if min_event_duration and (last["dt_end"] - last["dt_start"]) < min_event_duration:
            del sessions[-1]

    df = list(itertools.chain(*list(s for s in last_x_for_y.values() if s is not None)))

    if len(df) == 0:
        return None
    return last_x_for_y


def extract_parameters_from_events(event):
    bee_id0, bee_id1 = event["Task"].split("_")
    dt_start = event["dt_start"]
    dt_end = event["dt_end"]
    bee_id0_x_pos_start = event["location_info_0"][0]
    bee_id0_y_pos_start = event["location_info_0"][1]
    bee_id0_theta_start = event["location_info_0"][2]
    bee_id1_x_pos_start = event["location_info_1"][0]
    bee_id1_y_pos_start = event["location_info_1"][1]
    bee_id1_theta_start = event["location_info_1"][2]
    bee_id0_x_pos_end = event["location_info_0_end"][0]
    bee_id0_y_pos_end = event["location_info_0_end"][1]
    bee_id0_theta_end = event["location_info_0_end"][2]
    bee_id1_x_pos_end = event["location_info_1_end"][0]
    bee_id1_y_pos_end = event["location_info_1_end"][1]
    bee_id1_theta_end = event["location_info_1_end"][2]
    return {"bee_id0": int(bee_id0),
            "bee_id1": int(bee_id1),
            "interaction_start": dt_start,
            "interaction_end": dt_end,
            "bee_id0_x_pos_start": bee_id0_x_pos_start,
            "bee_id0_y_pos_start": bee_id0_y_pos_start,
            "bee_id0_theta_start": bee_id0_theta_start,
            "bee_id1_x_pos_start": bee_id1_x_pos_start,
            "bee_id1_y_pos_start": bee_id1_y_pos_start,
            "bee_id1_theta_start": bee_id1_theta_start,
            "bee_id0_x_pos_end": bee_id0_x_pos_end,
            "bee_id0_y_pos_end": bee_id0_y_pos_end,
            "bee_id0_theta_end": bee_id0_theta_end,
            "bee_id1_x_pos_end": bee_id1_x_pos_end,
            "bee_id1_y_pos_end": bee_id1_y_pos_end,
            "bee_id1_theta_end": bee_id1_theta_end
            }


def get_all_interactions_over_time(interaction_generator):
    def generator():
        for interaction in interaction_generator:
            bee_id0, bee_id1, timestamp, location_parameters_bee0, location_parameters_bee1 = interaction[
                "bee_id0"], interaction["bee_id1"], interaction["timestamp"], interaction["location_info_bee0"], \
                interaction["location_info_bee1"]
            # Make sure that the bee_id0 always refers to the one with the lower ID here.
            if bee_id0 < bee_id1:
                pair_string = "{}_{}".format(bee_id0, bee_id1)
                yield dict(y=pair_string, time=timestamp, color="interaction", loc_info_0=location_parameters_bee0,
                           loc_info_1=location_parameters_bee1)
            else:
                pair_string = "{}_{}".format(bee_id1, bee_id0)
                yield dict(y=pair_string, time=timestamp, color="interaction", loc_info_0=location_parameters_bee1,
                           loc_info_1=location_parameters_bee0)

    clustered_interactions = cluster_interactions_over_time(generator(), fill_gaps=False)
    return clustered_interactions


def get_velocity_change_per_bee(bee_id, interaction_start, interaction_end):
    delta_t = datetime.timedelta(0, 30)
    dt_before, dt_after = interaction_start - delta_t, interaction_end + delta_t

    if type(bee_id) == np.int64:
        bee_id = bee_id.item()

    # fetch velocities
    velocities = bb_behavior.db.trajectory.get_bee_velocities(bee_id, dt_before, dt_after)

    if velocities is None:
        return None, None

    vel_before = np.mean(
        velocities[(velocities['datetime'] >= dt_before) & (velocities['datetime'] < interaction_start)][
            'velocity'])
    vel_after = np.mean(
        velocities[(velocities['datetime'] > interaction_end) & (velocities['datetime'] <= dt_after)]['velocity'])

    vel_change = vel_after - vel_before
    percent_change = (vel_change / vel_before) * 100

    return vel_change, percent_change


def swap_focal_bee_to_be_low_circadian(df):
    new_frame_dict = {
        "circadianess_focal": [],
        "x_pos_start_focal": [],
        "y_pos_start_focal": [],
        "theta_start_focal": [],
        "vel_change_bee_focal": [],
        "relative_change_bee_focal": [],
        "circadianess_non_focal": [],
        "x_pos_start_non_focal": [],
        "y_pos_start_non_focal": [],
        "theta_start_non_focal": [],
        "vel_change_bee_non_focal": [],
        "relative_change_bee_non_focal": [],
    }
    parameters = ["circadianess", "x_pos_start", "y_pos_start", "theta_start", "vel_change_bee", "relative_change_bee"]
    for index, interaction in df.iterrows():
        if interaction["circadianess_focal"] > interaction["circadianess_non_focal"]:
            for parameter in parameters:
                focal = interaction["%s_focal" % parameter]
                non_focal = interaction["%s_non_focal" % parameter]
                new_frame_dict["%s_focal" % parameter].extend([non_focal])
                new_frame_dict["%s_non_focal" % parameter].extend([focal])
        else:
            for parameter in parameters:
                new_frame_dict["%s_focal" % parameter].extend([interaction["%s_focal" % parameter]])
                new_frame_dict["%s_non_focal" % parameter].extend([interaction["%s_non_focal" % parameter]])
    for parameter in parameters:
        df["%s_focal" % parameter] = new_frame_dict["%s_focal" % parameter]
        df["%s_non_focal" % parameter] = new_frame_dict["%s_non_focal" % parameter]
    return df


# transform coordinates
def transform_coordinates(interaction, focal_bee=0):
    bee0_x = interaction['x_pos_start_focal']
    bee0_y = interaction['y_pos_start_focal']
    bee0_theta = interaction['theta_start_focal']
    bee1_x = interaction['x_pos_start_non_focal']
    bee1_y = interaction['y_pos_start_non_focal']
    bee1_theta = interaction['theta_start_non_focal']

    if focal_bee == 0:
        # translation to make bee0 coordinates the origin
        x_prime = bee1_x - bee0_x
        y_prime = bee1_y - bee0_y
        bee1_coords = np.array([x_prime, y_prime])

        # rotation to make bee0 angle zero
        rot_mat = np.array([[np.cos(-bee0_theta), -np.sin(-bee0_theta)], [np.sin(-bee0_theta), np.cos(-bee0_theta)]])
        transformed = rot_mat @ bee1_coords

        return transformed[0], transformed[1]#, interaction['theta_end_focal'] - bee0_theta
    else:
        # translation to make bee1 coordinates the origin
        x_prime = bee0_x - bee1_x
        y_prime = bee0_y - bee1_y
        bee0_coords = np.array([x_prime, y_prime])

        # rotation to make bee1 angle zero
        rot_mat = np.array([[np.cos(-bee1_theta), -np.sin(-bee1_theta)], [np.sin(-bee1_theta), np.cos(-bee1_theta)]])
        transformed = rot_mat @ bee0_coords

        return transformed[0], transformed[1]#, interaction['theta_end_non_focal'] - bee1_theta


def apply_transformation(interaction_df):
    # apply transformation of coordinates
    transformed_coords_focal0 = interaction_df.apply(transform_coordinates, axis=1, result_type='expand', focal_bee=0)
    transformed_coords_focal1 = interaction_df.apply(transform_coordinates, axis=1, result_type='expand', focal_bee=1)

    # round coordinates to discretize positions
    transformed_coords_focal0 = transformed_coords_focal0.apply(round)
    transformed_coords_focal1 = transformed_coords_focal1.apply(round)

    # append columns to original vel_change_matrix_df
    interaction_df[['focal0_x_trans', 'focal0_y_trans']] = transformed_coords_focal0
    interaction_df[['focal1_x_trans', 'focal1_y_trans']] = transformed_coords_focal1

    return interaction_df


def get_dist(row):
    return np.linalg.norm([row['x'], row['y']])


def get_dist_special_coord(row):
    return np.linalg.norm([row['x_1'], row['y_1']])


def get_duration(df):
    df["duration"] = (df['interaction_end'] - df['interaction_start']).dt.total_seconds()


def concat_duration_time(combined_df, df):
    combined_df["duration"] = pd.concat([df['duration'], df['duration']])
    combined_df["hour"] = pd.concat([df['interaction_start'].dt.hour, df['interaction_start'].dt.hour])


def concat_ages(combined_df, df):
    combined_df['age_focal'] = pd.concat(
        [df['age_0'], df['age_1']])
    combined_df['age_non_focal'] = pd.concat(
        [df['age_1'], df['age_0']])


def concat_circ(combined_df, df):
    combined_df['circadianess_focal'] = pd.concat(
        [df['circadianess_bee0'], df['circadianess_bee1']])
    combined_df['circadianess_non_focal'] = pd.concat(
        [df['circadianess_bee1'], df['circadianess_bee0']])
    return combined_df


def combine_bees_from_interaction_df_to_be_all_focal(df, trans=False):
    combined_df = pd.DataFrame(
        columns=['circadianess_focal', 'circadianess_non_focal', 'age_focal', 'age_non_focal', 'vel_change_bee_non_focal',
                 'relative_change_bee_focal', 'relative_change_bee_non_focal', "duration", "hour"])
    concat_circ(combined_df, df)
    concat_ages(combined_df, df)
    concat_velocity_changes(combined_df, df)
    concat_duration_time(combined_df, df)
    concat_position(combined_df, df, trans=trans)
    return combined_df


def concat_velocity_changes(combined_df, df):
    combined_df['vel_change_bee_focal'] = pd.concat(
        [df['vel_change_bee_0'], df['vel_change_bee_1']])
    combined_df['vel_change_bee_non_focal'] = pd.concat(
        [df['vel_change_bee_1'], df['vel_change_bee_0']])
    combined_df['relative_change_bee_focal'] = pd.concat(
        [df['relative_change_bee_0'], df['relative_change_bee_1']])
    combined_df['relative_change_bee_non_focal'] = pd.concat(
        [df['relative_change_bee_1'], df['relative_change_bee_0']])


def concat_position(combined_df, df, trans=False):
    if trans:
        combined_df['x_pos_start_focal'] = pd.concat(
            [df['focal0_x_trans'], df['focal1_x_trans']])
        combined_df['x_pos_start_non_focal'] = pd.concat(
            [df['focal1_x_trans'], df['focal0_x_trans']])
        combined_df['y_pos_start_focal'] = pd.concat(
            [df['focal0_y_trans'], df['focal1_y_trans']])
        combined_df['y_pos_start_non_focal'] = pd.concat(
            [df['focal1_y_trans'], df['focal0_y_trans']])
        combined_df['theta_start_focal'] = pd.concat(
            [df['focal0_theta'], df['focal1_theta']])
        combined_df['theta_start_non_focal'] = pd.concat(
            [df['focal1_theta'], df['focal0_theta']])
    else:
        combined_df['x_pos_start_focal'] = pd.concat(
            [df['bee_id0_x_pos_start'], df['bee_id1_x_pos_start']])
        combined_df['x_pos_start_non_focal'] = pd.concat(
            [df['bee_id1_x_pos_start'], df['bee_id0_x_pos_start']])
        combined_df['y_pos_start_focal'] = pd.concat(
            [df['bee_id0_y_pos_start'], df['bee_id1_y_pos_start']])
        combined_df['y_pos_start_non_focal'] = pd.concat(
            [df['bee_id1_y_pos_start'], df['bee_id0_y_pos_start']])
        combined_df['theta_start_focal'] = pd.concat(
            [df['bee_id0_theta_start'], df['bee_id1_theta_start']])
        combined_df['theta_start_non_focal'] = pd.concat(
            [df['bee_id1_theta_start'], df['bee_id0_theta_start']])


def clean_interaction_df(interaction_df, column_subset=None):
    if column_subset is None:
        column_subset = interaction_df.columns
    for column in interaction_df[column_subset].columns:
        interaction_df = interaction_df[~interaction_df[column].isnull()]
        interaction_df = interaction_df[~np.isinf(interaction_df[column])]
    return interaction_df


def get_interactions(dt_from=None, dt_to=None, db_connection=None):
    with db_connection as db:
        cursor = db.cursor()

        # get frames
        frame_data = bb_behavior.db.get_frames(0, dt_from, dt_to, cursor=cursor)
        if not frame_data:
            return {None: dict(error="No frames fetched..")}

        interactions_lst = []
        # for each frame id
        for dt, frame_id, cam_id in frame_data:
            # get interactions
            interactions_detected = bb_behavior.db.find_interactions_in_frame(frame_id=frame_id, cursor=cursor,
                                                                              max_distance=14.0)
            for i in interactions_detected:
                interactions_lst.append({
                    "bee_id0": i[1],
                    "bee_id1": i[2],
                    "timestamp": dt,
                    "location_info_bee0": (i[5], i[6], i[7]),
                    "location_info_bee1": (i[8], i[9], i[10])
                })

        # cluster interactions per time in events
        events = get_all_interactions_over_time(interactions_lst)
        if not events:
            return {None: dict(error="No events found..")}

        # get interactions and velocity changes
        extracted_interactions_lst = []
        for key in events:
            event_dict = events[key]
            if events[key]:
                # extract events parameters as interactions
                interaction_dict = extract_parameters_from_events(event_dict[0])
                # get velocity changes
                # "focal" bee
                interaction_dict['vel_change_bee_0'], interaction_dict[
                    'relative_change_bee_0'] = get_velocity_change_per_bee(
                    bee_id=interaction_dict['bee_id0'],
                    interaction_start=interaction_dict['interaction_start'],
                    interaction_end=interaction_dict['interaction_end']
                )
                # "non-focal" bee
                interaction_dict['vel_change_bee_1'], interaction_dict[
                    'relative_change_bee_1'] = get_velocity_change_per_bee(
                    bee_id=interaction_dict['bee_id1'],
                    interaction_start=interaction_dict['interaction_start'],
                    interaction_end=interaction_dict['interaction_end']
                )
                extracted_interactions_lst.append(interaction_dict)
        return extracted_interactions_lst


def rotate(theta, vec):
    """
    Rotates a 2d vector anti-clockwise by an angle theta (in rad).
    """
    rot_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return rot_mat @ vec


def get_non_focal_bee_mask(x, y, theta):
    # create empty result frame
    non_focal_bee = np.zeros((29, 29))

    # get standard edge points of bee
    a = np.array((-7, -2)).astype(float)
    b = np.array((-7, 2)).astype(float)
    c = np.array((4, 2)).astype(float)
    d = np.array((4, -2)).astype(float)

    # rotate all edge points by theta
    if theta != 0:
        a = rotate(theta, a)
        b = rotate(theta, b)
        c = rotate(theta, c)
        d = rotate(theta, d)

    # add offset and move origin to (14,14) to correspond to
    # matrix coordinates
    a += [x + 14, y + 14]
    b += [x + 14, y + 14]
    c += [x + 14, y + 14]
    d += [x + 14, y + 14]

    points = np.array([a, b, c, d]).round().astype(int)

    # draw rectangle based on edge points
    return cv2.fillPoly(non_focal_bee, [points], 1)


def compute_interaction_points(interaction_df, overlap_dict):
    # create empty matrix for interaction counts
    counts = np.zeros((29, 29))

    # create empty matrix for accumulating velocities
    vels = np.zeros((29, 29))

    for i in range(len(interaction_df)):
        overlap = overlap_dict[interaction_df.index[i]]

        # add count and velocity values
        if np.sum(overlap) >= 1:
            counts += overlap
            vel = overlap * interaction_df.iloc[i][['vel_change_bee_focal']][0]
            vels += vel

    avg_vel = np.divide(vels, counts, out=np.zeros_like(vels), where=counts != 0)
    return counts, avg_vel


def create_overlap_dict(interaction_df):
    overlap_dict = {}
    for index, row in interaction_df.iterrows():
        overlap_dict[index] = get_bee_body_overlap(row)
    return overlap_dict


def get_bee_body_overlap(interaction_df_row):
    # create mask for focal bee
    focal_bee = np.zeros((29, 29))
    focal_bee[11:18, 6:20] = 1
    # get coordinates of interacting bee
    x, y, theta = interaction_df_row[['x_pos_start_focal', 'y_pos_start_focal', 'theta_start_focal']]
    # create mask for non-focal bee
    non_focal_bee = get_non_focal_bee_mask(x, y, theta)
    # get overlap
    overlap = np.logical_and(focal_bee, non_focal_bee)
    return overlap
