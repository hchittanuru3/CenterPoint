from configs.centerpoint.argoverse_centerpoint_pp_02voxel_circle_nms import *

data_root = 'data/argoverse/sample'
device_ids = [0]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    val=dict(
        type=dataset_type,
        root_path=data_root,
        test_mode=True,
        pipeline=test_pipeline,
    ),
)
