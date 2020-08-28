from .nuscenes import NuScenesDataset
from .waymo import WaymoDataset
from .argoverse import ArgoverseDataset

dataset_factory = {
    "NUSC": NuScenesDataset,
    "WAYMO": WaymoDataset,
    "ARGOVERSE": ArgoverseDataset,
}


def get_dataset(dataset_name):
    return dataset_factory[dataset_name]
