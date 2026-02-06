from huggingface_hub import snapshot_download
import tensorflow as tf
import os
import time
import shutil
from datetime import datetime



segment_str = "segment-1"


def huggingfaceeeeee(data_dir):
    snapshot_download(
        repo_id="AnnaZhang/waymo_open_dataset_v_1_4_3",
        repo_type="dataset",
        local_dir=data_dir,
        # 使用通配符匹配特定文件夹下的所有 .tfrecord 文件
        allow_patterns=[f"individual_files/training/{segment_str}*.tfrecord"],
        local_dir_use_symlinks=False,
        resume_download=True,
        # 如果文件很多，建议增加线程数提高速度
        max_workers=2
    )


def is_storage_sufficient(data_dir, min_gb_required=30):
    """
    检查剩余空间是否大于设定阈值（默认 30GB）
    """
    # 如果目录还没创建，检查其父目录
    check_path = data_dir
    while not os.path.exists(check_path):
        check_path = os.path.dirname(check_path)
        
    _, _, free = shutil.disk_usage(check_path)
    free_gb = free / (1024**3)
    
    if free_gb < min_gb_required:
        print(f"⚠️ 警告: 磁盘剩余空间仅剩 {free_gb:.2f} GB，低于阈值 {min_gb_required} GB！")
        return False
    return True


def validate_tfrecord(data_dir):
    # 1. 加载已经校验成功的名单，避免重复劳动
    success_log = os.path.join(data_dir, "valid_files.txt")
    fail_log = os.path.join(data_dir, "bad_record.txt")
    verified_files = set()
    if os.path.exists(success_log):
        with open(success_log, "r", encoding="utf-8") as f:
            verified_files = {line.strip() for line in f if line.strip()}

    # 获取目录下所有 tfrecord
    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.tfrecord')]
    
    # 过滤掉已经校验过的
    files_to_check = [f for f in all_files if f not in verified_files]
    
    if not files_to_check:
        print("☕ 所有本地文件已通过历史校验，跳过检查。")
        return

    print(f"🔍 正在校验 {len(files_to_check)} 个新文件...")

    with open(fail_log, "a", encoding="utf-8") as bad_log, \
         open(success_log, "a", encoding="utf-8") as good_log:
         
        for f in files_to_check:
            try:
                # 校验逻辑
                for _ in tf.data.TFRecordDataset(f):
                    pass
                
                # 校验通过：打印并记录
                print(f"OK: {f}")
                good_log.write(f + "\n")
                good_log.flush()
                
            except Exception as e:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                error_msg = f"[{now}] [FAILED]: {f}"
                print(error_msg)
                
                bad_log.write(error_msg + "\n") 
                bad_log.flush() 
                
                # 删除损坏文件，这样下次 HF 重启会自动补下
                if os.path.exists(f):
                    os.remove(f)
                    print(f"Deleted corrupted file: {f}")




def download_with_retries(data_dir):

    os.makedirs(data_dir, exist_ok=True)

    max_retries = 150000
    attempt = 0

    print("\n\n\n\n\n\n\n")
    validate_tfrecord(data_dir)

    while attempt < max_retries:
        try:
            # 下载路径和配置
            if not is_storage_sufficient(data_dir, 20): # 至少预留 20GB
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{now}] ❌ 空间不足，程序暂停。清理磁盘后自动重试。")
                time.sleep(60)
                continue
            huggingfaceeeeee(data_dir)
            print("🎉 Download completed successfully.")
            break  # 成功下载后退出循环
        except Exception as e:
            attempt += 1
            wait_time = 5
            print(f"😡 Download failed (attempt {attempt}/{max_retries}): {e}. Retrying in {wait_time} seconds...")
            print(f"启动文件检测...")
            validate_tfrecord(data_dir)
            time.sleep(wait_time)


if __name__ == "__main__":

    data_dir = "/data/repo/waymo/"

    download_with_retries(data_dir)

