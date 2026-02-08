from torch_waymo import WaymoDataset

# Simplified frames (no images, only point clouds + labels)
train_dataset = WaymoDataset('/data/repo/waymo/individual_files/converted_simplified_semantic', 'training')
for i in range(len(train_dataset)):
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
    from rich import print
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

