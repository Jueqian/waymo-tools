# waymo-tools

This repository is a modified version of [willGuimont/torch_waymo](https://github.com/willGuimont/torch_waymo).

Mainly optimization is download tfrecord from huggingface and extra 3D_semantic_label:

1. **HuggingFace Integration**: Download TFRecord files directly from HuggingFace.
2. **Data Extraction**: Extract LiDAR points, **3D semantic segmentation labels**, vehicle poses, and metadata into lightweight `.pkl` files.

3. **Incremental Processing**: Resume downloads and process data incrementally to save time and disk space.
4. **Torch Loading**: Load processed data in PyTorch without TensorFlow environment.


## Setup

Create a python environment.
```
conda create -n waymo_tools python==3.10 -y
conda activate waymo_tools
pip install uv
```

Install huggingface related and login.
```
uv pip install "huggingface_hub==1.4.0"
uv pip install "hf-cli==0.1"
sudo apt install git-lfs
```

``` huggingface login
hf auth login
```

Install waymo-open-dataset tools. (Fortunately, they finally pushed an update for the tools in 2025.)

```
uv pip install waymo-open-dataset-tf-2-12-0==1.6.5
uv pip install protobuf==3.20.3
uv pip install rich
uv pip install numpy==1.23.0
```


```
# Torch only for pkl reading, not participate in tfrecord conversion.
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Easier setup
For tfrecord conversion:
```
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e ".[waymo]"
hf auth login
```

For reading dataset easily:
```
uv pip install -e .
```



## Using

### Download tfrecords.
```
# 1.Download tfrecords
python tfrecord_download.py
# 2.valida the tfrecords
# If corrupted files are found, delete it and simply re-run step 1.
python tfrecord_validate.py
```

Notes

- Validation results are cached in 'valid_files.txt' to avoid redundant work

- Validation Failed files are logged in 'bad_record.txt' for check.

- Disk space is monitored continuously during download.

### Convert tfrecord to pkl

Converts original Waymo TFRecord files into lightweight `.pkl` files, allowing load datasets in PyTorch **without** the TensorFlow dependency.

Currently support to extra:

* **converted_frame.context**,
* **converted_frame.timestamp_micros**,
* **converted_frame.pose**,
* **converted_frame.laser_labels**,
* **converted_frame.no_label_zones**,
* **converted_frame.points**,       # list[(LiDAR XYZ)]
* **converted_frame.point_labels**, # 3D Semantic Segmentation Labels


```
python convert_waymo.py -d /data/repo/waymo/individual_files -s training --simplified

# or:
uv pip install -e ".[waymo]"
torch-waymo-convert --dataset /data/repo/waymo/individual_files  --simplified --splits training
```

Others data extraction (like camera) does not testing yet. For unlisted or experimental fields, please refer to the original repository [willGuimont/torch_waymo](https://github.com/willGuimont/torch_waymo).


### Read with torch only
```
uv pip install -e .
```

```
from torch_waymo import WaymoDataset

# Simplified frames (no images, only point clouds + labels)
train_dataset = WaymoDataset('/home/xuqingdong/repo/waymo_ceshi/individual_files/converted_simplified', 'training')
for i in range(10):
    # frame is of type SimplifiedFrame
    frame = train_dataset[i]

    context = frame.context
    timestamp_micros = frame.timestamp_micros
    laser_labels = frame.laser_labels
    no_label_zones = frame.no_label_zones
    pose = frame.pose
    points = frame.points
    point_labels = frame.point_labels


    import ipdb; ipdb.set_trace()
    
    import numpy as np
    print(np.unique(point_labels[0], axis=0))



    print(f"Frame {i}:")
    print(f"  Timestamp (micros): {timestamp_micros}")
    print(f"  Number of points: {len(points[0])}")
    print(f"  Number of point_labels: {len(point_labels[0])}")
    print(f"  Number of laser labels: {len(laser_labels)}")
    print(f"  Pose: {pose}")
    print(f"  Context: {context}")
    print(f"  No label zones: {no_label_zones}")

```




# End
Modidied by [willGuimont/torch_waymo](https://github.com/willGuimont/torch_waymo).
