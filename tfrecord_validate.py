import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="TFRecord Validate")
    
    # required: data_dir
    parser.add_argument(
        "-d",
        "--data_dir", 
        type=str, 
        required=True,
        help="Directory to the dataset (will be created if it doesn't exist)"
    )

    return parser.parse_args()

args = parse_args()
data_dir = os.path.join(args.data_dir)
files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.tfrecord')]

for f in files:
    try:
        for _ in tf.data.TFRecordDataset(f):
            pass
        print(f"OK: {f}")
    except Exception as e:
        print(f"FAILED: {f}")
        # os.remove(f)  # Remove corrupted file