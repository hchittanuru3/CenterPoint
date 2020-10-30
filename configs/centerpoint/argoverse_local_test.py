from configs.centerpoint.argoverse_centerpoint_pp_02voxel_circle_nms import *

device_ids = [0]
data_root = 'data/argoverse/sample'

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        root_path=data_root,
        n_sweeps_per_sample=n_sweeps_per_sample,
        timespan_per_sample=timespan_per_sample,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        root_path=data_root,
        n_sweeps_per_sample=n_sweeps_per_sample,
        timespan_per_sample=timespan_per_sample,
        test_mode=True,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        root_path=data_root,
        n_sweeps_per_sample=n_sweeps_per_sample,
        timespan_per_sample=timespan_per_sample,
        pipeline=test_pipeline,
    ),
)
