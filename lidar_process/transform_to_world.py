from torch_waymo import WaymoDataset
import numpy as np
from rich.console import Console
cs = Console()

import argparse
import pathlib
import pickle
from pyntcloud import PyntCloud
import pandas as pd

from utils import get_dynamic_bbox, filter_dynamic_pcd


def load_metadata(cache_path: pathlib.Path):
    meta_path = cache_path.joinpath("metadata.pkl")
    if meta_path.exists():
        with open(meta_path, "rb") as f:
            return pickle.load(f)
    return {"processed_scenes": {}, "frame_id": 0, "next_id": 0}


def save_metadata(cache_path: pathlib.Path, metadata: dict):
    meta_path = cache_path.joinpath("metadata.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)

def save_xyz_sensor_semantic_ply(geometries, sensorpose, save_path):
    """
    Args:
        geometries: List of np.array, 形状 (N, 4)，前三列 xyz，第四列 semantic_id
        sensorpose: List of np.array, 形状 (3,)，即平移向量 M_t
        save_path: str
    """
    all_frames = []

    for pts_sem, m_t in zip(geometries, sensorpose):

        xyz = pts_sem[:, :3]    # (N, 3)
        sem_labels = pts_sem[:, 3:4] # (N, 1)
        sensor = np.tile(m_t, (xyz.shape[0], 1))
        # (x, y, z, sensor_x, sensor_y, sensor_z, semantic_label) (N, 7)
        frame_data = np.hstack([xyz, sensor, sem_labels])
        all_frames.append(frame_data)

    all_points = np.vstack(all_frames)

    types = {
        'x': 'float32', 
        'y': 'float32', 
        'z': 'float32', 
        'sensor_x': 'float32', 
        'sensor_y': 'float32', 
        'sensor_z': 'float32', 
        'semantic_label': 'uint8'
    }
    df = pd.DataFrame(
        all_points, 
        columns=['x', 'y', 'z', 'sensor_x', 'sensor_y', 'sensor_z', 'semantic_label']
    )
    df = df.astype(types)

    cloud = PyntCloud(df)
    cloud.to_file(save_path, as_text=False)  # 二进制格式更紧凑
    cs.log(f"✅ Saved {len(all_points)} points to {save_path}")



def main_function(data_path: str, split: str):
    dataset = WaymoDataset(data_path, split)
    cs.log(f"Loaded {len(dataset)} frames from {split} split.")

    root_path = pathlib.Path(data_path).expanduser()

    output_root_name = "map_xyz_sensor_semantic"
    split_cache_path = root_path.parent.joinpath(output_root_name).joinpath(split)
    split_cache_path.mkdir(parents=True, exist_ok=True)
    cs.log(f"Cache path: {split_cache_path}")


    # import ipdb; ipdb.set_trace()
    metadata = load_metadata(split_cache_path)
    frame_id = metadata["frame_id"]
    scene_id = metadata["next_id"]

    geometries = []
    sensorpose = []
    for i in range(frame_id, len(dataset)):
        frame = dataset[i]
        basename = dataset.curr_basename
        cs.log(f"Current frame basename: {basename}")


        # import ipdb; ipdb.set_trace()

        if int(basename.split("_")[-1]) != scene_id:
            scene_path = split_cache_path.joinpath(f"scene_{scene_id}.ply").as_posix()
            save_xyz_sensor_semantic_ply(geometries, sensorpose, scene_path)
            geometries = []
            sensorpose = []

            metadata["processed_scenes"] = scene_id
            metadata["next_id"] = int(basename.split("_")[-1])
            metadata["frame_id"] = int(basename.split("_")[0])
            save_metadata(split_cache_path, metadata)
            scene_id = metadata["next_id"]
        
        pose = frame.pose   # T_l2w
        points = frame.points
        point_labels = frame.point_labels   # semantic, [[instance_id, semantic_class]]
        laser_labels = frame.laser_labels   # bbox



        lidar_top_xyz = points[0][:, :3]  # (N, 3)
        labels_top = point_labels[0] # [(N, 2)]， 第一列是 instance_id，第二列是 semantic_class_id

        semantic_class_id = labels_top[:, 1:2]

        # import ipdb; ipdb.set_trace()

        dynamic_bbox, ids = get_dynamic_bbox(laser_labels, speed_threshold=0.11)
        static_xyz, dynamic_xyz, dynamic_mask = filter_dynamic_pcd(lidar_top_xyz, dynamic_bbox)
        semantic_class_id_static = semantic_class_id[~dynamic_mask]

        T_l2w = pose
        num_pts = static_xyz.shape[0]
        homo_points = np.concatenate([static_xyz, np.ones((num_pts, 1))], axis=-1)
        pts_world = (T_l2w @ homo_points.T).T[:, :3]
        pts_world_sem = np.concatenate((pts_world[:, :3], semantic_class_id_static), axis=-1)

        sensor_xyz = pose[:3, 3]

        geometries.append(pts_world_sem)
        sensorpose.append(sensor_xyz)





def parse_args():
    SPLITS = ["training", "validation", "testing"]

    parser = argparse.ArgumentParser(
        prog="Merge points to Map",
        description="Transform lidar to world with semantic and merge to map ply.",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=True,
        help="Path to the Waymo Open Dataset",
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

    args = parse_args()
    dataset_path = args.dataset
    splits = args.splits


    for split in splits:
        cs.log(f"Processing {split} in mode...")
        main_function(dataset_path, split)
