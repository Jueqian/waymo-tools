import open3d as o3d    # pip install open3d==0.16

def visualize_static_dynamic(static_pcd, dynamic_pcd):
    # 静态点染成灰色
    pcd_static = o3d.geometry.PointCloud()
    pcd_static.points = o3d.utility.Vector3dVector(static_pcd[:, :3])
    pcd_static.paint_uniform_color([0.5, 0.5, 0.5])

    # 动态点染成鲜艳的红色
    pcd_dynamic = o3d.geometry.PointCloud()
    pcd_dynamic.points = o3d.utility.Vector3dVector(dynamic_pcd[:, :3])
    pcd_dynamic.paint_uniform_color([1, 0, 0])

    o3d.visualization.draw_geometries([pcd_static, pcd_dynamic], window_name="Red are dynamic")  # type: ignore


import numpy as np
import torch


# the code modified from LidarDM (https://github.com/vzyrianov/LidarDM)
def get_dynamic_bbox(laser_labels, speed_threshold=0.11):

    bboxes = []
    ids = []
    for label in laser_labels:
        speed = np.linalg.norm(np.array([label.metadata.speed_x, 
                                            label.metadata.speed_y,
                                         label.metadata.speed_z]))
        if speed > speed_threshold:
            # cs.log(f"Label {label.id} is dynamic with speed {speed:.2f} m/s.")
            # import ipdb; ipdb.set_trace()

            b = label.box
            box_list = [b.center_x, b.center_y, b.center_z, 
                        b.length, b.width, b.height, b.heading]
            box_tensor = torch.tensor(box_list, dtype=torch.float32)
            box_tensor[3:6] += torch.tensor([1.0, 0.5, 0.5])
            bboxes.append(box_tensor)

            current_type = "Vehicle" if label.type == 1 else "Pedestrian" # 1是Vehicle, 2是Pedestrian...
            ids.append([label.id, current_type, b.heading])

    # 4. 将列表堆叠成一个大的 Tensor (N, 7)
    if len(bboxes) > 0:
        bboxes_tensor = torch.stack(bboxes)
    else:
        bboxes_tensor = torch.empty((0, 7))
        
    return bboxes_tensor, ids

def get_yaw_rotation(yaw):
    """逻辑对齐原 TF 函数：获取绕 Z 轴旋转矩阵 [..., 3, 3]"""
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    ones = torch.ones_like(yaw)
    zeros = torch.zeros_like(yaw)

    # 构造矩阵，注意 axis 对齐原代码的 -2 和 -1
    return torch.stack([
        torch.stack([cos_yaw, -1.0 * sin_yaw, zeros], dim=-1),
        torch.stack([sin_yaw, cos_yaw, zeros], dim=-1),
        torch.stack([zeros, zeros, ones], dim=-1),
    ], dim=-2)

def get_transform(rotation, translation):
    """逻辑对齐原 TF 函数：组合成 (N+1)x(N+1) 变换矩阵"""
    # translation: [..., 3] -> [..., 3, 1]
    translation_n_1 = translation.unsqueeze(-1)
    # [..., 3, 4]
    transform = torch.cat([rotation, translation_n_1], dim=-1)
    
    # 构建最后一行 [0, 0, 0, 1]
    batch_shape = translation.shape[:-1]
    last_row = torch.zeros_like(translation) # [..., 3]
    last_row = torch.cat([last_row, torch.ones((*batch_shape, 1), device=last_row.device, dtype=last_row.dtype)], dim=-1)
    
    # [..., 4, 4]
    transform = torch.cat([transform, last_row.unsqueeze(-2)], dim=-2)
    return transform

def is_within_box_3d(point, box, name=None):
    """
    点在框内判断 (Torch 版)
    Args:
        point: [N, 3] tensor (x, y, z)
        box: [M, 7] tensor (cx, cy, cz, l, w, h, yaw)
    Returns:
        point_in_box: [N, M] bool tensor
    """
    # 确保输入是 torch.Tensor
    if isinstance(point, np.ndarray):
        point = torch.from_numpy(point).float()
    if isinstance(box, np.ndarray):
        box = torch.from_numpy(box).float()

    # 逻辑 1: 提取参数
    center = box[:, 0:3]
    dim = box[:, 3:6]
    heading = box[:, 6]

    # 逻辑 2: 获取旋转矩阵 [M, 3, 3]
    rotation = get_yaw_rotation(heading)
    
    # 逻辑 3: 获取变换矩阵 [M, 4, 4]
    transform = get_transform(rotation, center)
    
    # 逻辑 4: 矩阵求逆 (World -> Box Frame)
    transform = torch.linalg.inv(transform)
    
    # 逻辑 5: 提取逆变换后的旋转和平移
    rotation_inv = transform[:, 0:3, 0:3]
    translation_inv = transform[:, 0:3, 3]

    # 逻辑 6: 坐标变换点到 Box 坐标系 [N, M, 3]
    # TF: einsum('nj,mij->nmi', point, rotation) + translation
    point_in_box_frame = torch.einsum('nj,mij->nmi', point, rotation_inv) + translation_inv

    # 逻辑 7: 范围判断 [-dim/2, dim/2]
    # TF: logical_and (point <= dim * 0.5, point >= -dim * 0.5)
    in_box_mask = (point_in_box_frame <= dim * 0.5) & (point_in_box_frame >= -dim * 0.5)
    
    # 逻辑 8: 处理 dim=0 的无效框并做 reduce_all
    valid_dim_mask = (dim != 0).all(dim=-1, keepdim=True) # [M, 1]
    point_in_box = in_box_mask & valid_dim_mask

    # 逻辑 9: 结果 reduce_prod (即所有轴都为 True) 得到 [N, M]
    point_in_box = point_in_box.all(dim=-1)

    return point_in_box





def filter_dynamic_pcd(points, bboxes_tensor):
    if bboxes_tensor.shape[0] == 0:
        return points, np.empty((0, points.shape[1])), np.zeros((points.shape[0],), dtype=bool)
    # mask [N, M]
    mask_in_boxes = is_within_box_3d(points[:, :3], bboxes_tensor).numpy()
    # 只要在任何一个框内，就是动态点
    is_dynamic = np.any(mask_in_boxes, axis=1)

    static_points = points[~is_dynamic]  # 静态点
    dynamic_points = points[is_dynamic]  # 动态点 (被剔除的部分)
    
    return static_points, dynamic_points, is_dynamic