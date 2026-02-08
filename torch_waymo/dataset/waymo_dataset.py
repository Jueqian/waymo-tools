import pathlib
import pickle
from typing import Callable, Optional, Union
import natsort
import warnings
from itertools import groupby
from torch.utils.data import Dataset

from ..protocol.dataset_proto import Frame
from .simplified_frame import SimplifiedFrame


class WaymoDataset(Dataset):
    def __init__(self, root_path: str, split: str, transform: Optional[Callable] = None):
        self._root_path = pathlib.Path(root_path).expanduser()
        self._split = split
        self._split_path = self._root_path.joinpath(split)
        self._split_files = natsort.natsorted(self._split_path.glob("*_*.pkl"))
        self._verify_scene_continuity(self._split_files)
        # import ipdb; ipdb.set_trace()
        self._transform = transform

        # Validate root path existence
        if not self._root_path.exists() or not self._root_path.is_dir():
            raise FileNotFoundError(f"Dataset root path does not exist or is not a directory: {self._root_path}")

        # Validate split directory existence
        if not self._split_path.exists() or not self._split_path.is_dir():
            raise FileNotFoundError(
                f"Split path does not exist: {self._split_path}. You may have the wrong root or forgot to convert this split."
            )

        self._seq_len_cache_path = self._split_path.joinpath("metadata.pkl")
        if self._seq_len_cache_path.exists():
            with open(self._seq_len_cache_path, "rb") as f:
                pkl_data = pickle.load(f)
                self._seq_lens = pkl_data['total_frames']
        else:
            raise FileNotFoundError(
                f"Could not find sequence length cache file: {self._seq_len_cache_path}. Conversion may be incomplete."
            )
        
    def _verify_scene_continuity(self, raw_files):
        # 2. 提取场景 ID 序列
        # 假设文件名是 1211_40.pkl，我们取 '40'
        scene_id_sequence = [f.stem.split('_')[-1] for f in raw_files]
        
        # 3. 使用 groupby 找出连续的区块
        # 如果序列是 [40, 40, 41, 40]，groups 会是 ['40', '41', '40']
        groups = [key for key, _ in groupby(scene_id_sequence)]
        
        # 4. 验证：如果 group 后的长度大于 set 后的长度，说明有 ID 穿插重复出现了
        if len(groups) > len(set(groups)):
            interleaved = [item for item in set(groups) if groups.count(item) > 1]
            warnings.warn(
                f"检测到场景 ID 穿插! 这些 ID 出现了不连续分布: {interleaved}. "
                f"建议检查排序逻辑或原始数据命名.", 
                UserWarning
            )
        else:
            print(f"✅ 数据验证通过: 共 {len(groups)} 个场景, 顺序完全连续")
            
        return raw_files
    

    def __len__(self) -> int:
        # return sum(self._seq_lens)
        return len(self._split_files)

    def __getitem__(self, idx: int) -> Union[SimplifiedFrame, Frame]:
        # path = self._split_path.joinpath(f"{idx}_*.pkl")
        path = self._split_files[idx]
        self.curr_basename = path.stem

        if path.exists():
            return self._get_frame(path)
        else:
            raise IndexError(f"Could not load frame at index {idx} (missing file {path}).")

    def _get_frame(self, path):
        with open(path, "rb") as f:
            x = pickle.load(f)
        if self._transform is not None:
            x = self._transform(x)
        return x
