from configs.centerpoint.argoverse_centerpoint_pp_02voxel_circle_nms import *

data['samples_per_gpu'] = 1
data['workers_per_gpu'] = 1
device_ids = [0]
data_root = 'data/argoverse/sample'
