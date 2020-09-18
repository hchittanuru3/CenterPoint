from det3d.datasets.argoverse.argoverse_common import *
from det3d.datasets.custom import PointCloudDataset
from det3d.datasets.registry import DATASETS


@DATASETS.register_module
class ArgoverseDataset(PointCloudDataset):

    def __init__(self, root_path, pipeline=None, test_mode=False):
        self._log_id_timestamps_list = ArgoverseDataset.load_logs(root_path, 1)
        # Not using info_path but loading ground truth directly for now
        super(ArgoverseDataset, self).__init__(
            root_path, None, pipeline, test_mode=test_mode
        )
        #self._num_point_features = 3  # x, y, z
        self._num_point_features = 5  # x, y, z, i, r

    def __getitem__(self, index):
        return self.get_sensor_data(index)

    def __len__(self):
        return len(self._log_id_timestamps_list)

    def get_sensor_data(self, index):
        log_id, timestamps = self._log_id_timestamps_list[index]

        # Load point cloud/intensity and reflectance data separately
        clouds, ir = load_all_clouds(self._root_path, log_id, timestamps) 

        clouds = perform_SE3(clouds, self._root_path, log_id)
        clouds = grids_group_and_SE3(clouds, self._root_path, log_id, 1)
        #points = np.float32(list(clouds.values())[0][0])

        # Concatenate transformed point cloud with intensity and reflection data
        points = np.concatenate(
            [np.float32(list(clouds.values())[0][0]), np.float32(list(ir.values())[0])], 
            axis=1)

        data_dict = load_all_boxes(self._root_path, log_id, timestamps)
        bbox_dict = convert_to_boundingbox(data_dict)
        box_dict = box_group_and_SE3(bbox_dict, self._root_path, log_id, 1)
        track_group = box_group_to_track_group(box_dict)
        track_group_tensors_dict, _ = label_to_tensor(track_group, 1)
        gt_boxes = list(track_group_tensors_dict.values())[0]
        gt_boxes = gt_boxes.squeeze(1).numpy()

        info = {}
        res = {
            "lidar": {
                "type": "lidar",
                "points": points,
                "annotations": {
                    "boxes": gt_boxes,
                    "names": np.array(["vehicle"] * gt_boxes.shape[0])
                },
            },
            "metadata": {
                "image_prefix": self._root_path,
                "num_point_features": self._num_point_features,
                "token": None,
            },
            "calib": None,
            "cam": {},
            "mode": "val" if self.test_mode else "train",
            "type": "ArgoverseDataset",
        }

        data, _ = self.pipeline(res, info)

        return data

    @staticmethod
    def load_logs(dataset: str, n_t: int) -> List[Tuple[str, List[int]]]:
        """Loads a list of timestamps in all logs and their corresponding log ids.

        Args:
            dataset: (str) Path to dataset
            n_t: (int) Number of timestamps per training example

        Returns:
            tuple_list: (List[Tuple[str, List[int]]]) A list of Tuples in which the first element is the log id of
                                                      a training example, and the second element is the list of
                                                      timestamps
        """
        tuple_list = []
        for log_id in os.listdir(dataset):
            timestamps_list = ArgoverseDataset.load_timestamps_in_log(dataset, log_id, n_t)
            tuple_list.extend([(log_id, lst) for lst in timestamps_list])
        return tuple_list

    @staticmethod
    def load_timestamps_in_log(dataset: str, log_id: str, n_t: int) -> List[List[int]]:
        """Loads list of timestamps using the point cloud files in the given dataset and log id

        Args:
            dataset: (str) Path to dataset
            log_id: (str) Log id
            n_t: (int) Number of timestamps per training example

        Returns:
            timestamps_list: (List[List[int]]) A list in which every element is a list of size n_t containing
                                               timestamps for that training example
        """
        timestamps = []
        dir_path = os.path.join(dataset, log_id, "lidar")
        for file in os.listdir(dir_path):
            timestamps.append(int(file[3:-4]))
        timestamps.sort()

        return [timestamps[start:(start + n_t)] for start in range(len(timestamps) - n_t + 1)]


if __name__ == '__main__':
    dataset = ArgoverseDataset('data/argoverse/sample')
    dataset[0]
