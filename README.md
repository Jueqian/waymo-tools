# waymo-tools
Tools for processing waymo-open-dataset.

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

