from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms as T
import json
from typing import Tuple
import io
import lmdb


_json_path = Path('train_test_split.json')

# Load the file
with open(_json_path, 'r') as f:
    _split_data = json.load(f)

# Access the train and test entries
train_list = _split_data.get("train", [])
test_list = _split_data.get("test", [])
val_list = _split_data.get("val", [])

_reduced_path = Path('reduced_videos.json')
with open(_reduced_path, 'r') as f:
    _reduced_list = json.load(f)

moco_transform = T.Compose([
    T.RandomHorizontalFlip(p=0.5),      # flip left-right with 50% probability
    T.RandomVerticalFlip(p=0.5),        # flip top-bottom with 50% probability
    T.RandomRotation(degrees=10),       # rotate randomly between -10° and +10°
    T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # slight blur
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],  # or use dataset-specific values
                std=[0.229, 0.224, 0.225]),
])

base_path = '/scratch/cv-course-group-5/data/dataset_jpg'
dst_root   = Path(base_path + '/lmdb')

class CellDataset(Dataset):
    def __init__(self, video_list=_reduced_list, path_to_videos=dst_root, transform=T.ToTensor()):
        self.env = lmdb.open(str(path_to_videos), readonly=True, lock=False)
        with self.env.begin() as txn:
            lmdb_keys = txn.get(b"__keys__").decode().split("\n")
        self.transform = transform
        self.keys = []
        video_names = [video["name"] for video in video_list]

        for key in lmdb_keys:
            for video_name in video_names:
                if key.startswith(f"{video_name}/"):
                    self.keys.append(key)
        print(len(self.keys))


    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        """
        :param idx: Inter
        :return: Item at idx after applying transform
        :rtype: Tuple[torch.Tensor]
        """
        key = self.keys[idx]
        with self.env.begin() as txn:
            img_bytes = txn.get(key.encode("utf-8"))
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        view1 = self.transform(img)
        view2 = self.transform(img)
        return view1, view2