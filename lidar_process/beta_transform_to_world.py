import os

import glob
import natsort
import numpy as np
import pickle
from pyntcloud import PyntCloud
import pandas as pd


from rich.console import Console
cs = Console()

from utils import get_dynamic_bbox, filter_dynamic_pcd


def load_pkl(path):
    with open(path, "rb") as f:
        x = pickle.load(f)
    return x

def read_momo_scene(data_dir, scene_split=None):
    frames_dir = natsort.natsorted(glob.glob(os.path.join(data_dir, "*.pkl")))
    cs.log(f"Found {len(frames_dir)} frames in {data_dir}")

    scene_dict = {}
    for frame_path in frames_dir:
        filename = os.path.basename(frame_path)
        name, _ext = os.path.splitext(filename)
        if "_" not in name:
            continue
        _frame_idx, scene_id = name.split("_", 1)
        if scene_split is not None:
            if isinstance(scene_split, (set, list, tuple)):
                if scene_id not in scene_split:
                    continue
            else:
                if scene_id != str(scene_split):
                    continue
        scene_dict.setdefault(scene_id, []).append(frame_path)

    return scene_dict


def save_waymo_compatible_ply(geometries, sensorpose, save_path):
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

    # 4. 构建 DataFrame
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
    print(f"✅ Saved {len(all_points)} points to {save_path}")


def load_waymo_example(waymo_path):

    pcloud = PyntCloud.from_file(str(waymo_path))
    pdata = pcloud.points
    xyz = np.stack([pdata['x'], pdata['y'], pdata['z']], axis=1)
    sensor = np.stack([pdata['sensor_x'], pdata['sensor_y'], pdata['sensor_z']], axis=1)
    semantic_label = np.stack([pdata['semantic_label']], axis=1)

    return xyz, sensor, semantic_label


# 建议加一个 main 保护，防止多进程报错
def main():
    data_dir = "/Users/xiaodong/repo/waymo/momo_0"

    scene_dict = read_momo_scene(data_dir)
    scene_0 = scene_dict.get("0", [])

    geometries = []
    sensorpose = [] # 世界坐标系下传感器位置（即激光雷达位置），用于后续保存到 ply 文件中
    geometries_dynamic = []
    
    for i, frame_path in enumerate(scene_0):
        frame = load_pkl(frame_path)
        # import ipdb; ipdb.set_trace()

        context = frame.context
        timestamp_micros = frame.timestamp_micros
        laser_labels = frame.laser_labels
        no_label_zones = frame.no_label_zones
        pose = frame.pose   # 这里的 pose 是激光雷达坐标系到世界坐标系的变换矩阵
        points = frame.points
        point_labels = frame.point_labels   # [[instance_id, semantic_class]]


        lidar_top_xyz = points[0][:, :3]  # (N, 3)
        labels_top = point_labels[0] # [(N, 2)]， 第一列是 instance_id，第二列是 semantic_class_id

        semantic_class_id = labels_top[:, 1]
        

        dynamic_bbox, ids = get_dynamic_bbox(laser_labels, speed_threshold=0.11)
        static_xyz, dynamic_xyz, dynamic_mask = filter_dynamic_pcd(lidar_top_xyz, dynamic_bbox)

        semantic_class_id_static = semantic_class_id[~dynamic_mask]

        T_l2w = pose
        num_pts = static_xyz.shape[0]
        homo_points = np.concatenate([static_xyz, np.ones((num_pts, 1))], axis=-1)
        pts_world = (T_l2w @ homo_points.T).T[:, :3]
        pts_world_sem = np.column_stack((pts_world[:, :3], semantic_class_id_static))


        sensor_xyz = pose[:3, 3]

        geometries.append(pts_world_sem)
        sensorpose.append(sensor_xyz)



    # all_points_sem = np.concatenate(geometries, axis=0)
    # visualize_static_dynamic(all_points_sem, all_points_dynamic)

    save_waymo_compatible_ply(geometries, sensorpose, "waymo-pcd.ply")




if __name__ == '__main__':
    main()

    