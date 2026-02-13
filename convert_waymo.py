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

from rich.console import Console
cs = Console()



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
    return {"processed_files": {}, "total_frames": 0, "total_scenes": 0}


def save_metadata(cache_path: pathlib.Path, metadata: dict):
    meta_path = cache_path.joinpath("metadata.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)


def generate_cache(root_path: pathlib.Path, split: str, simplified: bool = False, semantic_frame_only: bool = False):
    output_root_name = "converted_simplified" if simplified else "converted"
    output_root_name += "_semantic" if semantic_frame_only else ""
    split_path = root_path.joinpath(split)
    split_cache_path = root_path.joinpath(output_root_name).joinpath(split)

    split_path.mkdir(parents=True, exist_ok=True)
    split_cache_path.mkdir(parents=True, exist_ok=True)

    # 1. Load processed_files info
    metadata = load_metadata(split_cache_path)
    processed_files = metadata["processed_files"]
    frame_count = metadata["total_frames"]
    scene_count = metadata["total_scenes"]

    # 2. get the all tfrecord belonging to the split
    current_files = sorted([
        p for p in split_path.iterdir()
        if p.is_file() and p.suffix == '.tfrecord'
    ])
    
    # 3. Filter out already processed files based on metadata.pkl
    new_files = [p for p in current_files if p.name not in processed_files]
    
    if not new_files:
        cs.log(f"No new sequences found in {split_path}. All current files are already processed.")
        return

    cs.log(f"Found {len(new_files)} new sequences. Starting incremental processing from frame {frame_count}...")

    # 4. Incrementally process new files
    for seq_path in tqdm.tqdm(new_files, desc=f"Total Progress ({split})"):
        seq_name = seq_path.name

        # import ipdb; ipdb.set_trace()
        
        # Get the length of the sequence (only count once for new files)
        try:
            seq_len = _get_size(seq_path)
        except Exception as e:
            cs.log(f"\n[Warning] Could not read {seq_name}, might be incomplete. Skipping. Error: {e}")
            continue

        seq_dataset = tf.data.TFRecordDataset(seq_path, compression_type="")
        semantic_frame_count = 0

        # Inner loop to process frames
        for frame_idx, data in tqdm.tqdm(enumerate(seq_dataset), total=seq_len, desc=f"  Processing {seq_name[:12]}", leave=False, ncols=80):
            frame_path = split_cache_path.joinpath(f"{frame_count}_{scene_count}.pkl")
            
            if not frame_path.exists():
                try:
                    obj, has_semantic = _load_frame(data, simplified=simplified, semantic_frame_only=semantic_frame_only)
                    if obj is None:
                        continue
                    with open(frame_path, "wb") as f:
                        pickle.dump(obj, f)
                    if has_semantic:
                        semantic_frame_count += 1
                except Exception as e:
                    cs.log(f"\n[Error] Failed to parse frame in {seq_name}: {e}")
                    # Skip single frame if it fails to parse.
            frame_count += 1

        scene_count += 1
        # 5. Update metadata.pkl after processing each sequence
        processed_files[seq_name] = {"scene_len": seq_len, "scene_id": scene_count-1, "semantic_frames_len": semantic_frame_count}
        metadata["total_frames"] = frame_count
        metadata["total_scenes"] = scene_count
        save_metadata(split_cache_path, metadata)




def _get_size(s: pathlib.Path) -> int:
    return sum(1 for _ in tf.data.TFRecordDataset(s, compression_type=""))



def _load_frame(data, simplified: bool, semantic_frame_only: bool):
    """
    Load a single frame from TFRecord data.
    :param data: TFRecord data
    :param simplified: If True, return a SimplifiedFrame (no images), else return full Frame.
    :return: Frame or SimplifiedFrame
    """
    frame = open_dataset.Frame()    # type: ignore
    frame.ParseFromString(bytearray(data.numpy()))
    has_semantic = frame.lasers[0].ri_return1.segmentation_label_compressed

    if semantic_frame_only and not has_semantic:
        return None, has_semantic
    
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
        return converted_frame, has_semantic

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
    return simple_frame, has_semantic


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
    parser.add_argument(
        "--semantic_only",
        action="store_true",
        help="This will only save the frames included semantic labels (waymo ssemantic_label is 2hz).",
    )


    args = parser.parse_args()
    dataset_path = pathlib.Path(args.dataset).expanduser()
    splits = args.splits
    simplified = args.simplified
    semantic_frame_only = args.semantic_only

    for split in splits:
        mode = "simplified" if simplified else "full"
        cs.log(f"Processing {split} in {mode} mode...")
        generate_cache(dataset_path, split, simplified=simplified, semantic_frame_only=semantic_frame_only)


if __name__ == "__main__":
    main()

