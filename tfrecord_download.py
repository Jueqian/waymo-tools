import os
import time
import glob
import shutil
from rich.console import Console
console = Console(width=100)


from huggingface_hub import snapshot_download

try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
    import tensorflow as tf
    TENSORFLOW = True
except ImportError:
    console.log("⚠️ No TensorFlow. Validation disabled.")
    TENSORFLOW = False



# Could be set from 'segment-1' to 'segment-9'
segment_str = "segment-1" 


def huggingfaceeeeee(data_dir):
    snapshot_download(
        repo_id="AnnaZhang/waymo_open_dataset_v_1_4_3",
        repo_type="dataset",
        local_dir=data_dir,
        # Match .tfrecord files.
        allow_patterns=[f"individual_files/training/{segment_str}*.tfrecord"],
        # local_dir_use_symlinks=False,
        # resume_download=True,
        # Download two files at the same time.
        max_workers=2
    )


def get_free_space_gb(data_dir):
    """
    Check disk space is above the threshold (default: 30 GB).
    """
    # If the directory does not exist yet, check its parent directory
    check_path = data_dir
    while not os.path.exists(check_path):
        check_path = os.path.dirname(check_path)
        
    _, _, free = shutil.disk_usage(check_path)
    free_gb = free / (1024**3)
    return free_gb
    

def check_free_space(data_dir, min_gb_required=30):

    while True:
        free_gb = get_free_space_gb(data_dir)
        if free_gb > min_gb_required:
            break
        else:
            console.log(
                f"⚠️ Waiting Free Space: Only {free_gb:.2f} GB free (< {min_gb_required} GB)."
            )
            time.sleep(60)


def validate_tfrecord(data_dir):
    if not TENSORFLOW:
        console.log("⚠️ No TensorFlow. Skipping Validation.")
        return True
    
    # Load already validated list to avoid redundant work
    success_log = os.path.join(data_dir, "valid_files.txt")
    fail_log = os.path.join(data_dir, "bad_record.txt")
    verified_files = set()
    if os.path.exists(success_log):
        with open(success_log, "r", encoding="utf-8") as f:
            verified_files = {line.strip() for line in f if line.strip()}

    # Get tfrecord files and filter already verified
    search_path = os.path.join(data_dir, "**/*.tfrecord")
    all_files = glob.glob(search_path, recursive=True)
    files_to_check = [f for f in all_files if f not in verified_files]
    
    if not files_to_check:
        return True

    console.log(f"🧪 Validating {len(files_to_check)} new files...")

    RES = True
    with open(fail_log, "a", encoding="utf-8") as bad_log, \
         open(success_log, "a", encoding="utf-8") as good_log:
        
        for f in files_to_check:
            try:
                # Validation logic
                for _ in tf.data.TFRecordDataset(f):
                    pass
                console.log(f"\t[OK]: {f}")
                good_log.write(f + "\n")
                good_log.flush()
                
            except Exception as e:
                error_msg = f"\t[FAILED]: {f}"
                RES = False
                console.log(error_msg)
                
                bad_log.write(error_msg + "\n") 
                bad_log.flush() 
                
                # Remove corrupted file, HF will re-download
                if os.path.exists(f):
                    os.remove(f)
                    console.log(f"🗑️ Deleted corrupted file: {f}")
    return RES




def download_with_retries(data_dir):

    os.makedirs(data_dir, exist_ok=True)
    console.log(f"🚀 Starting download with retries...")

    attempt, max_retries = 0, 150000
    while attempt < max_retries:
        try:
            check_free_space(data_dir)
            huggingfaceeeeee(data_dir)

            if not validate_tfrecord(data_dir):
                raise RuntimeError("TFRecord validation failed")

            console.log(f"🎉 Download completed successfully.")
            break
        except KeyboardInterrupt:
            console.log("🛑 Detected Ctrl+C. Exiting...")
            os._exit(1)
        except Exception as e:
            attempt += 1
            wait_time = 5
            console.log(f"😡 Download failed (attempt {attempt}/{max_retries}): {e}. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)



if __name__ == "__main__":

    data_dir = "/home/xuqingdong/repo/waymo_ceshi"

    download_with_retries(data_dir)

