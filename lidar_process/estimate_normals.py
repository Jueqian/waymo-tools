import numpy as np
from rich.console import Console
from rich.live import Live
from rich.text import Text

cs = Console()

import argparse
import pathlib
import pickle
from pyntcloud import PyntCloud
import pandas as pd
import torch
import point_cloud_utils as pcu     # uv pip install point_cloud_utils==0.34.0
from natsort import natsorted


def load_waymo_with_labels(ply_path):

    pcloud = PyntCloud.from_file(str(ply_path))
    pdata = pcloud.points
    
    # xyz->(N,3), sensor->(N,3), labels->(N,1)
    xyz = np.stack([pdata['x'], pdata['y'], pdata['z']], axis=1)    
    sensor = np.stack([pdata['sensor_x'], pdata['sensor_y'], pdata['sensor_z']], axis=1)
    labels = pdata['semantic_label'].values.reshape(-1, 1)     # keep original uint8 type
    
    return xyz, sensor, labels



def save_xyz_normal_semantic_ply(xyz_np, sensor_np, normal_np, labels_np, save_path):

    xyz_nxyz_sem = np.hstack([xyz_np, sensor_np, labels_np, normal_np])  # (N, 10)

    types = {
        'x': 'float32', 
        'y': 'float32', 
        'z': 'float32', 
        'sensor_x': 'float32',
        'sensor_y': 'float32',
        'sensor_z': 'float32',
        'semantic_label': 'uint8',
        'normal_x': 'float32', 
        'normal_y': 'float32', 
        'normal_z': 'float32', 
    }
    df = pd.DataFrame(
        xyz_nxyz_sem, 
        columns=['x', 'y', 'z', 'sensor_x', 'sensor_y', 'sensor_z', 'semantic_label', 'normal_x', 'normal_y', 'normal_z']
    )
    df = df.astype(types)

    cloud = PyntCloud(df)
    cloud.to_file(save_path, as_text=False)  # 二进制格式更紧凑
    cs.log(f"Saved {len(xyz_nxyz_sem)} points to {save_path}")




def normal_func(xyz: torch.Tensor, normal: torch.Tensor, sensor: torch.Tensor, others: torch.Tensor):
    assert normal is None, "normal already exists"
    assert sensor is not None, "please provide sensor positions for consistent orientations"

    live = Live(Text(""), console=cs, auto_refresh=True, transient=True)
    live.start()

    live.update(Text(" get xyz_numpy", style="bold green", no_wrap=True))
    xyz_numpy = xyz.cpu().numpy()

    live.update(Text(" Estimating normals...", style="bold green", no_wrap=True))
    indices, normal = pcu.estimate_point_cloud_normals_knn(xyz_numpy, 64)

    live.update(Text(" Converting normals to torch tensors...", style="bold green", no_wrap=True))
    normal = torch.from_numpy(normal).to(xyz)
    indices = torch.from_numpy(indices).to(xyz).long()

    xyz, sensor = xyz[indices], sensor[indices]

    live.update(Text(" Flipping normals to face the sensor...", style="bold green", no_wrap=True))
    view_dir = sensor - xyz
    view_dir = view_dir / (torch.linalg.norm(view_dir, dim=-1, keepdim=True) + 1e-6)
    cos_angle = torch.sum(view_dir * normal, dim=1)
    cos_mask = cos_angle < 0.0
    normal[cos_mask] = -normal[cos_mask]

    keep_mask = torch.abs(cos_angle) > np.cos(np.deg2rad(85.0))
    xyz, normal, sensor, others = xyz[keep_mask], normal[keep_mask], sensor[keep_mask], others[keep_mask]
    live.stop()

    return xyz, normal, sensor, others


def estimate_and_save(ply_path, save_path):

    xyz_np, sensor_np, labels_np = load_waymo_with_labels(ply_path)
    input_xyz = torch.from_numpy(xyz_np).float().to('cpu')
    input_sensor = torch.from_numpy(sensor_np).float().to('cpu')
    input_others = torch.from_numpy(labels_np).float().to('cpu')


    xyz_processed, normals, sensor_processed, labels_processed = normal_func(input_xyz, None, input_sensor, input_others)  # type: ignore

    xyz_out = xyz_processed.numpy()
    normals_out = normals.numpy()
    labels_out = labels_processed.numpy()
    sensor_out = sensor_processed.numpy()

    save_xyz_normal_semantic_ply(xyz_out, sensor_out, normals_out, labels_out, save_path)


def main(root_dir: str, split: str):

    data_path = pathlib.Path(root_dir).joinpath(split)
    save_path = pathlib.Path(root_dir.replace("sensor_semantic", "sensor_semantic_normal")).joinpath(split)
    save_path.mkdir(parents=True, exist_ok=True)
    ply_files_path = natsorted(data_path.glob("*.ply"))

    for _path in ply_files_path:
        # save_path = _path.with_name(_path.stem + "_normals.ply")
        estimate_and_save(str(_path), str(save_path / _path.name))





def parse_args():
    SPLITS = ["training", "validation", "testing"]

    parser = argparse.ArgumentParser(
        prog="Estimate Normals",
        description="Estimate normals for map and save as (xyz, normals, semantic_label) ply.",
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        type=str,
        required=True,
        help="Path to the map ply files",
    )
    parser.add_argument(
        "-s",
        "--splits",
        type=str,
        choices=SPLITS,
        nargs="+",
        default=SPLITS,
        help="Specify the splits you want to process",
    )
    return parser.parse_args()


if __name__ == "__main__":

    '''
    python lidar_process/estimate_normals.py \
             -d /data/repo/waymo/individual_files/map_xyz_sensor_semantic \
             -s training
    '''

    args = parse_args()
    dataset_path = args.data_dir
    splits = args.splits


    for split in splits:
        cs.log(f"Processing {split} in mode...")
        main(dataset_path, split)



# if __name__ == "__main__":


#     ply_path = "/data/repo/waymo/individual_files/map_xyz_sensor_semantic/training/scene_50.ply"
#     xyz, sensor, labels = load_waymo_with_labels(ply_path)
#     cs.log(f"Loaded {xyz.shape[0]} points with semantic labels from {ply_path}")
#     cs.log(f"Example point: {xyz[0]}, sensor: {sensor[0]}, label: {labels[0]}")
#     import ipdb; ipdb.set_trace()