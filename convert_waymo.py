import argparse
import pathlib
import pickle

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow.compat.v1 as tf   # type: ignore

tf.enable_eager_execution()
import tqdm
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils

from torch_waymo.dataset import SimplifiedFrame
from torch_waymo.protocol import dataset_proto
from torch_waymo.protocol.dataset_proto import Frame



def convert_range_image_to_point_cloud_labels(frame,
                                              range_images,
                                              segmentation_labels,
                                              ri_index=0):
  """Convert segmentation labels from range images to point clouds.

  Args:
    frame: open dataset frame
    range_images: A dict of {laser_name, [range_image_first_return,
       range_image_second_return]}.
    segmentation_labels: A dict of {laser_name, [range_image_first_return,
       range_image_second_return]}.
    ri_index: 0 for the first return, 1 for the second return.

  Returns:
    point_labels: {[N, 2]} list of 3d lidar points's segmentation labels. 0 for
      points that are not labeled.
  """
  calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
  point_labels = []
  for c in calibrations:
    range_image = range_images[c.name][ri_index]
    range_image_tensor = tf.reshape(
        tf.convert_to_tensor(range_image.data), range_image.shape.dims)
    range_image_mask = range_image_tensor[..., 0] > 0

    if c.name in segmentation_labels:
      sl = segmentation_labels[c.name][ri_index]
      sl_tensor = tf.reshape(tf.convert_to_tensor(sl.data), sl.shape.dims)
      sl_points_tensor = tf.gather_nd(sl_tensor, tf.where(range_image_mask))
    else:
      num_valid_point = tf.math.reduce_sum(tf.cast(range_image_mask, tf.int32))
      sl_points_tensor = tf.zeros([num_valid_point, 2], dtype=tf.int32)
      
    point_labels.append(sl_points_tensor.numpy())
  return point_labels



def load_metadata(cache_path: pathlib.Path):
    meta_path = cache_path.joinpath("metadata.pkl")
    if meta_path.exists():
        with open(meta_path, "rb") as f:
            return pickle.load(f)
    return {"processed_files": {}, "total_frames": 0}


def save_metadata(cache_path: pathlib.Path, metadata: dict):
    meta_path = cache_path.joinpath("metadata.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)


def generate_cache(root_path: pathlib.Path, split: str, simplified: bool = False):
    output_root_name = "converted_simplified" if simplified else "converted"
    split_path = root_path.joinpath(split)
    split_cache_path = root_path.joinpath(output_root_name).joinpath(split)

    split_path.mkdir(parents=True, exist_ok=True)
    split_cache_path.mkdir(parents=True, exist_ok=True)

    # 1. 加载元数据
    metadata = load_metadata(split_cache_path)
    processed_files = metadata["processed_files"]
    frame_count = metadata["total_frames"]

    # 2. 扫描当前目录下的所有 tfrecord
    current_files = sorted([
        p for p in split_path.iterdir() 
        if p.is_file() and p.suffix == '.tfrecord'
    ])
    
    # 3. 筛选出尚未处理的新文件
    new_files = [p for p in current_files if p.name not in processed_files]
    
    if not new_files:
        print(f"No new sequences found in {split_path}. All current files are already processed.")
        return

    print(f"Found {len(new_files)} new sequences. Starting incremental processing from frame {frame_count}...")

    # 4. 增量处理新文件
    for seq_path in tqdm.tqdm(new_files, desc=f"Total Progress ({split})"):
        seq_name = seq_path.name
        
        # 获取该序列的长度（只针对新文件数一次数）
        try:
            seq_len = _get_size(seq_path)
        except Exception as e:
            print(f"\n[Warning] Could not read {seq_name}, might be incomplete. Skipping. Error: {e}")
            continue

        seq_dataset = tf.data.TFRecordDataset(seq_path, compression_type="")
        
        # 内层循环处理帧
        for data in tqdm.tqdm(seq_dataset, total=seq_len, desc=f"  Processing {seq_name[:10]}", leave=False):
            frame_path = split_cache_path.joinpath(f"{frame_count}.pkl")
            
            # 如果文件已存在则跳过（增强健壮性）
            if not frame_path.exists():
                try:
                    obj = _load_frame(data, simplified=simplified)
                    with open(frame_path, "wb") as f:
                        pickle.dump(obj, f)
                except Exception as e:
                    print(f"\n[Error] Failed to parse frame in {seq_name}: {e}")
                    # 如果某一帧坏了，通常建议跳过该帧，保持计数器增长
            
            frame_count += 1
        
        # 5. 处理完一个序列，立即更新元数据并落盘
        processed_files[seq_name] = seq_len
        metadata["total_frames"] = frame_count
        save_metadata(split_cache_path, metadata)



def _get_size(s: pathlib.Path) -> int:
    return sum(1 for _ in tf.data.TFRecordDataset(s, compression_type=""))



def _load_frame(data, simplified: bool):
    """
    Load a single frame from TFRecord data.
    :param data: TFRecord data
    :param simplified: If True, return a SimplifiedFrame (no images), else return full Frame.
    :return: Frame or SimplifiedFrame
    """
    frame = open_dataset.Frame()    # type: ignore
    frame.ParseFromString(bytearray(data.numpy()))
    converted_frame = dataset_proto.from_data(Frame, frame)

    # Generate point cloud
    (
        range_images,
        camera_projections,
        segmentation_labels,
        range_image_top_pose,
    ) = frame_utils.parse_range_image_and_camera_projection(frame)
    points, _ = frame_utils.convert_range_image_to_point_cloud(
        frame, range_images, camera_projections, range_image_top_pose
    )
    point_labels = convert_range_image_to_point_cloud_labels(
        frame, range_images, segmentation_labels
    )
    converted_frame.points = points     # type: ignore
    converted_frame.point_labels = point_labels # type: ignore

    if not simplified:
        # Return full frame (images, lasers, labels). Point cloud generation skipped for speed.
        return converted_frame

    # Simplified path: compute point cloud and build SimplifiedFrame (no images stored)
    simple_frame = SimplifiedFrame(
        converted_frame.context,
        converted_frame.timestamp_micros,
        converted_frame.pose,
        converted_frame.laser_labels,
        converted_frame.no_label_zones,
        converted_frame.points,
        converted_frame.point_labels,
    )
    return simple_frame


def main():
    SPLITS = ["training", "validation", "testing"]

    parser = argparse.ArgumentParser(
        prog="Convert Waymo",
        description="Convert the Waymo Open Dataset to remove all dependencies to Tensorflow",
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
    parser.add_argument(
        "--simplified",
        action="store_true",
        help="Store simplified frames (no images) into 'converted_simplified' instead of full frames.",
    )

    args = parser.parse_args()
    dataset_path = pathlib.Path(args.dataset).expanduser()
    splits = args.splits
    simplified = args.simplified

    for split in splits:
        mode = "simplified" if simplified else "full"
        print(f"Processing {split} in {mode} mode...")
        generate_cache(dataset_path, split, simplified=simplified)


if __name__ == "__main__":
    main()

