import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


data_dir = "/home/xuqingdong/repo/waymo_ceshi/individual_files/training"
files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.tfrecord')]

for f in files:
    try:
        for _ in tf.data.TFRecordDataset(f):
            pass
        print(f"OK: {f}")
    except Exception as e:
        print(f"FAILED: {f}")
        # os.remove(f)  # Remove corrupted file