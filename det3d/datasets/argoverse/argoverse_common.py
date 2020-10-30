import copy
import json
import os
from typing import Dict, Tuple, List, Union

import numpy as np
import pyntcloud
import torch
from argoverse.data_loading.object_label_record import json_label_dict_to_obj_record
from argoverse.data_loading.pose_loader import get_city_SE3_egovehicle_at_sensor_t
from argoverse.utils import se3
from scipy import interpolate
from scipy.spatial.transform import Rotation, RotationSpline

_PathLike = Union[str, "os.PathLike[str]"]


class Label:
    """
    Class to store the bounding boxes and label class of an occupancy grid
    """

    def __init__(self, record, label):
        self.record = record
        self.boxes = None
        self.options = {"BACKGROUND": 0, "VEHICLE": 1}
        self.label_class = label
        self.label_num = self.options[self.label_class]


def get_rotation_matrix(degrees):
    return np.array([
        [np.cos(degrees), -np.sin(degrees), 0],
        [np.sin(degrees), np.cos(degrees), 0],
        [0, 0, 0],
    ])


def load_ply_xyzir(ply_fpath: _PathLike) -> np.ndarray:
    """ Load a point cloud file from a filepath.
    Edit 09/17/20: Include intensity and reflectance.
    Args:
        ply_fpath: Path to a PLY file
    Returns:
        arr: Array of shape (N, 3) - point cloud
        arr: Array of shape (N, 2) - intensity and reflectance data
    """

    data = pyntcloud.PyntCloud.from_file(os.fspath(ply_fpath))

    # Original point cloud data
    x = np.array(data.points.x)[:, np.newaxis]
    y = np.array(data.points.y)[:, np.newaxis]
    z = np.array(data.points.z)[:, np.newaxis]

    # Additional intensity and ring data
    i = np.array(data.points.intensity)[:, np.newaxis]
    r = np.array(data.points.laser_number)[:, np.newaxis]

    return np.concatenate((x, y, z), axis=1), np.concatenate((i, r), axis=1)


def load_all_clouds(dataset: str, log_id: str, timestamp_list: List[int]) -> Dict:
    """Loads all point clouds from the Argoverse Tracking set

    Args:
        dataset: (str) A string pointing to the path of the Argoverse dataset
        log_id: (str) The ID of the data collected
        timestamp_list: (list[int]) List of timestamps to load point clouds from
    Returns:
        cloud_dict: (dict) A dictionary with the timestamp as key and point cloud as value
    """
    cloud_dict = {}
    ir_dict = {}
    file_path = os.path.join(dataset, log_id, "lidar")
    for timestamp in timestamp_list:
        point_cloud, ir_data = load_ply_xyzir(
            os.path.join(file_path, f"PC_{timestamp}.ply")
        )
        ir_dict[timestamp] = ir_data
        cloud_dict[timestamp] = point_cloud
    return cloud_dict, ir_dict


def aggregate_points_to_city_frame(cloud_dict: Dict, dataset: str, log_id: str) -> Dict:
    """Performs the SE3 transformation on all point clouds

    Args:
        cloud_dict: (dict) A dictionary with the timestamp as key and point cloud as value
        dataset: (str) A string pointing to the path of the Argoverse dataset
        log_id: (str) The ID of the data collected
    Returns:
        cloud_dict: (dict) A dictionary with the timestamp as key and transformed point cloud as value
    """
    for k, v in cloud_dict.items():
        SE3 = get_city_SE3_egovehicle_at_sensor_t(int(k), dataset, log_id)
        cloud_dict[k] = SE3.transform_point_cloud(v)
    return cloud_dict


def grids_group_and_SE3(cloud_dict: Dict, dataset: str, log_id: str, n_t: int) -> Dict:
    """Groups five point clouds and performs the necessary SE3 transformation

    Args:
        cloud_dict: (dict) A dictionary with the timestamp as key and transformed point cloud as value
        dataset: (str) A string pointing to the path of the Argoverse dataset
        log_id: (str) The ID of the data collected
        n_t: (int) Number of timestamps per group
    Returns:
        cloud_dict: (dict) A dictionary with the timestamp as key and five consecutive point clouds as value
    """
    timestamps = []
    for key in sorted(cloud_dict.keys()):
        timestamps.append(key)
        if len(timestamps) == n_t:
            t0_to_map_SE3 = get_city_SE3_egovehicle_at_sensor_t(
                int(timestamps[0]), dataset, log_id
            )
            map_to_t0_SE3 = t0_to_map_SE3.inverse()
            rotation = get_rotation_matrix(np.pi / 2)
            translation = np.array([0, 0, 0])
            t0_to_occ_SE3 = se3.SE3(rotation, translation)
            grids = []
            for ts in timestamps:
                pc = cloud_dict[ts]
                pc = map_to_t0_SE3.transform_point_cloud(pc)
                pc = t0_to_occ_SE3.transform_point_cloud(pc)
                grids.append(pc)
            cloud_dict[timestamps[0]] = grids
            timestamps = timestamps[1:]
    for ts in timestamps:
        del cloud_dict[ts]
    return cloud_dict


def load_all_boxes(dataset: str, log_id: str, timestamp_list: List[int]) -> Dict:
    """Loads all bounding boxes from the Argoverse Tracking set

    Args:
        dataset: (str) A string pointing to the path of the Argoverse dataset
        log_id: (str) The ID of the data collected
        timestamp_list: (list[int]) List of timestamps to load point clouds from
    Returns:
        data_dict: (dict) A dictionary with the timestamp as key and all bounding boxes as value

    Note: This bounding box representation can be found at
        https://github.com/argoai/argoverse-api/blob/master/argoverse/data_loading/object_label_record.py#L232
    """
    file_path = os.path.join(dataset, log_id, "per_sweep_annotations_amodal")
    data_dict = {}
    for timestamp in timestamp_list:
        with open(
                os.path.join(file_path, f"tracked_object_labels_{timestamp}.json")
        ) as jsonfile:
            arr = json.loads(jsonfile.read())
            for hashmap in arr:
                if hashmap["label_class"] == "VEHICLE":
                    if timestamp not in data_dict.keys():
                        data_dict[timestamp] = []
                    data_dict[timestamp].append(
                        Label(json_label_dict_to_obj_record(hashmap), "VEHICLE")
                    )
    return data_dict


def convert_to_boundingbox(data_dict: Dict) -> Dict:
    """Converts all bounding boxes to numpy arrays of coordinates and performs respective SE3

    Args:
        data_dict: (dict) A dictionary with the timestamp as key and all bounding boxes as value
    Returns:
        bbox_dict: (dict) A dictionary with the timestamp as key and all bounding boxes in coordinate form as value
    """
    bbox_dict = copy.deepcopy(data_dict)
    for timestamp, objects in bbox_dict.items():
        for label in objects:
            label.boxes = label.record.as_2d_bbox()
    return bbox_dict


def box_group_and_SE3(box_dict: Dict, dataset: str, log_id: str, n_t: int) -> Dict:
    """Groups bounding boxes from five timestamps and performs the necessary SE3 transformation

    Args:
        box_dict: (dict) A dictionary with the timestamp as key and all bounding boxes in coordinate form as value
        dataset: (str) A string pointing to the path of the Argoverse dataset
        log_id: (str) The ID of the data collected
        n_t: (int) Number of timestamps per group
    Returns:
        box_dict: (dict) A dictionary with the timestamp as key and five sets of bounding boxes from consecutive timestamps as value
    """
    curr_timestamps = []
    for key in sorted(box_dict.keys()):
        curr_timestamps.append(key)
        if len(curr_timestamps) == n_t:
            t0_to_map_SE3 = get_city_SE3_egovehicle_at_sensor_t(
                int(curr_timestamps[0]), dataset, log_id
            )
            map_to_t0_SE3 = t0_to_map_SE3.inverse()
            rotation = get_rotation_matrix(np.pi / 2)
            translation = np.array([0, 0, 0])
            t0_to_occ_SE3 = se3.SE3(rotation, translation)
            new_labels = []
            for ts in curr_timestamps:
                labels_for_ts = box_dict[ts]
                new_labels_for_ts = []
                for label in labels_for_ts:
                    new_label = Label(label.record, label.label_class)
                    ego_to_map_SE3 = get_city_SE3_egovehicle_at_sensor_t(
                        ts, dataset, log_id
                    )
                    new_label.boxes = ego_to_map_SE3.transform_point_cloud(label.boxes)
                    new_label.boxes = map_to_t0_SE3.transform_point_cloud(
                        new_label.boxes
                    )
                    new_label.boxes = t0_to_occ_SE3.transform_point_cloud(
                        new_label.boxes
                    )
                    new_labels_for_ts.append(new_label)
                new_labels.append(new_labels_for_ts)
            box_dict[curr_timestamps[0]] = new_labels
            curr_timestamps = curr_timestamps[1:]
    for ts in curr_timestamps:
        del box_dict[ts]
    return box_dict


def box_group_to_track_group(box_dict):
    """Converts bounding box dictionary to list of track groups (boxes with the same track_id)
    """
    track_groups = {}
    # k: timestamp; v: list of 5 lists, each being objects in that timestamp
    for k, v in box_dict.items():
        group_dict = {}
        for idx, timestamp in enumerate(v):
            for label in timestamp:
                if label.record.track_id not in group_dict:
                    group_dict[label.record.track_id] = {}
                group_dict[label.record.track_id][idx] = label
        group_list = [group_dict[k] for k in group_dict]
        track_groups[k] = group_list
    return track_groups

def label_to_tensor(track_group_dict: Dict, timespan_per_sample: int) -> Tuple[Dict, List]:
    """
    Converts all the sets of labels into five tensors for each timestamp
    Each bounding box is in the form of [x, y, z, l, w, h, velocity_x, velocity_y, rot, label]

    Args:
        track_group_dict: (Dict) A dictionary with the timestamp as key and a list of tracks containing object
                                 bounding boxes across five timestamps as value
    Returns:
        box_dict: (Dict) A dictionary with the timestamp as key and five sets of bounding boxes from consecutive
                         timestamps as value
    """
    tensors = {}
    for timestamp, tracks in track_group_dict.items():
        num_tracks = len(tracks)
        tensors[timestamp] = torch.zeros((num_tracks, 9))  # 9 values per bounding box
        for track_idx, track in enumerate(tracks):
            tensors[timestamp][track_idx] = convert_track_and_infer_velocity(track, timespan_per_sample)
    return tensors

def convert_track_and_infer_velocity(track, interval_between_timestamps):
    timestamps = list(iter(track))

    boxes = torch.tensor([track[t].boxes for t in timestamps], dtype=torch.float32)
    boxes_tensor = convert_track_to_tensor(boxes, track, interval_between_timestamps)

    # Only take the bounding boxes from the last timestamp
    return boxes_tensor[-1]

def interpolate_track(track: Dict, n_t: int) -> Tuple[torch.tensor, List]:
    """Interpolate a track to fill in gaps in which we do not have poses for the tracked object.

    Args:
        track: (Dict) A dictionary with timestamps (0, 1, 2, ...) as keys and Label as values
        n_t: (int) Number of timestamps in total

    Returns:
        interpolated: (Dict) Same as input track, but with gaps filled
        interpolated_timestamps: (List) List of timestamps of interpolated poses
    """
    # TODO: more intelligent interpolation. E.g. we know the orientation of the car and it can't
    # move along the width axis, so even if there are some small jittery in translation along width axis, shouldn't
    # keep moving the vehicle along that direction
    timestamps = list(iter(track))
    to_interpolate = [t for t in range(n_t) if t not in timestamps]

    existing_boxes = torch.tensor([track[t].boxes for t in timestamps], dtype=torch.float32)
    existing_boxes_tensor = convert_track_to_tensor(existing_boxes, track, interval_between_timestamps=0.1)

    if len(to_interpolate) == 0:
        # Nothing to interpolate
        return existing_boxes_tensor, []
    elif len(timestamps) == 1:
        # Not enough to interpolate. Repeat the tensor n_t times
        return existing_boxes_tensor.repeat((n_t, 1)), to_interpolate

    # Fit rotations
    rots = Rotation.from_euler('x', existing_boxes_tensor[:, 6])
    spline = RotationSpline(timestamps, rots)
    interpolated_rots = spline(to_interpolate)

    # Fit x, y and z of translation separately
    # UnivariateSpline works the best practically in our case, but it doesn't work with <= 3 points
    interp_cls = interpolate.UnivariateSpline if len(timestamps) > 3 else interpolate.KroghInterpolator
    func_x = interp_cls(timestamps, existing_boxes_tensor[:, 0])
    func_y = interp_cls(timestamps, existing_boxes_tensor[:, 1])
    func_z = interp_cls(timestamps, existing_boxes_tensor[:, 2])

    interpolated_x = func_x(to_interpolate)
    interpolated_y = func_y(to_interpolate)
    interpolated_z = func_z(to_interpolate)

    interpolated = torch.zeros((n_t, 9))  # 9 values per bounding box

    interpolated[timestamps, :] = existing_boxes_tensor

    first_label = track[list(iter(track))[0]]

    interpolated[to_interpolate, 0] = torch.tensor(interpolated_x, dtype=torch.float32)
    interpolated[to_interpolate, 1] = torch.tensor(interpolated_y, dtype=torch.float32)
    interpolated[to_interpolate, 2] = torch.tensor(interpolated_z, dtype=torch.float32)

    # Length, width, height and label are set to the same value as the first given label
    interpolated[to_interpolate, 3] = first_label.record.length
    interpolated[to_interpolate, 4] = first_label.record.width
    interpolated[to_interpolate, 5] = first_label.record.height

    interpolated[to_interpolate, 8] = torch.tensor(interpolated_rots.as_euler('xyz')[:, 0], dtype=torch.float32)

    return interpolated, to_interpolate

def convert_track_to_tensor(bboxes, track, interval_between_timestamps):
    center_and_yaws = find_center_and_rotation(bboxes)
    w_l_h = torch.tensor([
        [label.record.width, label.record.length, label.record.height]
        for _, label in track.items()
    ], dtype=torch.float32)

    if w_l_h.shape[0] == 1:
        velocity = torch.zeros((1, 2))
    else:
        velocity = torch.zeros((w_l_h.shape[0], 2))
        velocity[:-1] = (center_and_yaws[1:, :2] - center_and_yaws[:-1, :2]) / interval_between_timestamps
        # Duplicate velocity from the second last timestamp to the last
        velocity[-1] = velocity[-2]

    return torch.cat((
        center_and_yaws[:, 0].unsqueeze(1),
        center_and_yaws[:, 1].unsqueeze(1),
        bboxes[:, 0, 2].unsqueeze(1),
        w_l_h,
        velocity,
        center_and_yaws[:, 2].unsqueeze(1),
    ), dim=1)

def find_center_and_rotation(boxes: torch.Tensor) -> torch.Tensor:
    """
    Finds the (x,y) coordinates of the bounding boxes' centers, as well as their yaws

    Args:
        boxes: (torch.Tensor) A list of bounding boxes, each bounding box has four corners (x, y, z)
    Returns:
        centers_and_yaws: (torch.Tensor) Center x, y and yaw for each bounding box
    """
    dxdy = boxes[:, 0, :2] - boxes[:, 2, :2]
    dx = dxdy[:, 0]
    dy = dxdy[:, 1]
    yaws = -torch.atan2(dy, dx).unsqueeze(1) - np.pi / 2  # Conforming to the way nusc_common handles yaw
    corners = boxes[:, [3, 0], :2]
    center = (corners[:, 0, :] + corners[:, 1, :]) / 2
    centers_and_yaws = torch.cat((center, yaws), dim=1)
    return centers_and_yaws
