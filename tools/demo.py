import argparse
import os

try:
    import apex
except:
    print("No APEX!")
import numpy as np
import torch
from det3d.datasets import build_dataset
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.apis import (
    batch_processor,
)
from det3d.torchie.trainer import load_checkpoint
from det3d.torchie.parallel import collate_kitti
from torch.utils.data import DataLoader
import cv2
from tools.demo_utils import visual

config_path = {
    'nuscenes': 'configs/centerpoint/nusc_centerpoint_pp_02voxel_circle_nms_demo.py',
    'argoverse': 'configs/centerpoint/argoverse_centerpoint_pp_02voxel_circle_nms_demo.py',
}

checkpoint_path = {
    'nuscenes': 'work_dirs/centerpoint_pillar_512_demo/last.pth',
    'argoverse': 'work_dirs/argoverse_centerpoint_pp_02voxel_circle_nms/skynet_epoch_19.pth',
}

def convert_box(boxes, names):
    boxes = boxes.astype(np.float32)
    assert len(boxes) == len(names)
    detection = {'box3d_lidar': boxes, 'label_preds': np.zeros(len(boxes)), 'scores': np.ones(len(boxes))}
    return detection

def main(dataset_name):
    cfg = Config.fromfile(config_path[dataset_name])
    
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)

    dataset = build_dataset(cfg.data.val)

    data_loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=None,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_kitti,
        pin_memory=False,
    )

    load_checkpoint(model, checkpoint_path[dataset_name], map_location="cpu")
    model.eval()

    model = model.cuda()

    cpu_device = torch.device("cpu")

    points_list = []
    gt_annos = []
    detections = []

    for i, data_batch in enumerate(data_loader):
        if dataset_name == 'nuscenes':
            info = dataset._nusc_infos[i]
            gt_annos.append(convert_box(info["gt_boxes"], info["gt_names"]))
        else:
            info = data_batch['annos'][0]
            gt_annos.append(convert_box(info["boxes"], info["names"]))

        points = data_batch['points'][:, 1:4].cpu().numpy()
        with torch.no_grad():
            outputs = batch_processor(
                model, data_batch, train_mode=False, local_rank=0,
            )
        for output in outputs:
            for k, v in output.items():
                if k not in [
                    "metadata",
                ]:
                    output[k] = v.to(cpu_device)
            detections.append(output)

        points_list.append(points.T)
    
    print('Done model inference. Please wait a minute, the matplotlib is a little slow...')
    
    for i in range(len(points_list)):
        visual(points_list[i], gt_annos[i], detections[i], i, eval_range=50)
        print("Rendered Image {}".format(i))
    
    image_folder = 'demo'
    video_name = 'video.avi'

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort()
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 1, (width,height))
    cv2_images = [] 

    for image in images:
        cv2_images.append(cv2.imread(os.path.join(image_folder, image)))

    for img in cv2_images:
        video.write(img)

    cv2.destroyAllWindows()
    video.release()

    print("Successfully save video in the main folder")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    assert args.dataset == 'nuscenes' or args.dataset == 'argoverse'

    main(args.dataset)
