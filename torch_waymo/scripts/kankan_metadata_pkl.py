import pathlib
import pickle
from rich import print

def load_metadata(cache_path: pathlib.Path):
    meta_path = cache_path.joinpath("metadata.pkl")
    if meta_path.exists():
        with open(meta_path, "rb") as f:
            return pickle.load(f)
    return {"processed_files": {}, "total_frames": 0}


if __name__ == "__main__":

    # data_dir = "/data/repo/waymo/individual_files/converted_simplified_semantic/training"
    data_dir = "/data/repo/waymo/individual_files/map_xyz_sensor_semantic/training"
    metadata = load_metadata(pathlib.Path(data_dir))
    print(metadata)

    import ipdb; ipdb.set_trace()
