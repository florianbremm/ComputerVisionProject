from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms as T
import json
from typing import Tuple
import io
import torch
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

anno_file  = Path('/scratch/cv-course-group-5/data/dataset_jpg/dataset/annotations.json')

# load annotations
annos_dict = json.loads(anno_file.read_text())

annos = annos_dict.get('annotations', [])
videos = annos_dict.get('videos', [])
images = annos_dict.get('images', [])

video_id2name = {v["id"]: v["name"] for v in videos}
image_by_id = {img["id"]: img for img in images} 

labels = {}

for ann in annos:
    video_id = ann["video_id"]
    image_id = ann["image_id"]
    cell_id = str(ann["cell_id"]).zfill(3)

    frame_nmbr = image_by_id[image_id]['file_name'][-7:-4]
    video_name = video_id2name[video_id]
    
    # if there is no entry yet in the dictionary
    if video_name not in labels:
        labels[video_name] = {}
    if frame_nmbr not in labels[video_name]:
        labels[video_name][frame_nmbr] = {}
    if cell_id not in labels[video_name][frame_nmbr]:
        # if ann['time_of_death'] is None:
        #     labels[video_name][frame_nmbr][cell_id] = 150
        # else:
        #     labels[video_name][frame_nmbr][cell_id] = ann['time_of_death'] - ann['time_step']

        if ann['time_of_death'] is None:
            labels[video_name][frame_nmbr][cell_id] = 1
        else:
            labels[video_name][frame_nmbr][cell_id] = int(ann['time_step'] < ann['time_of_death'] - 6)

moco_transform = T.Compose([
    T.RandomHorizontalFlip(p=0.5),      # flip left-right with 50% probability
    T.RandomVerticalFlip(p=0.5),        # flip top-bottom with 50% probability
    T.RandomRotation(degrees=10),       # rotate randomly between -10° and +10°
    T.RandomApply([T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.08)], 0.8),
    T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # slight blur
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],  # or use dataset-specific values
                std=[0.229, 0.224, 0.225]),
])

default_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

base_path = '/scratch/cv-course-group-5/data/dataset_jpg'
dst_root   = Path(base_path + '/lmdb')

class CellDataset(Dataset):
    def __init__(self, video_list=_reduced_list, path_to_videos=dst_root, transform=default_transform, mode='training'):
        self.env = lmdb.open(str(path_to_videos), readonly=True, lock=False)
        with self.env.begin() as txn:
            lmdb_keys = txn.get(b"__keys__").decode().split("\n")
        self.transform = transform
        self.keys = []
        self.mode = mode
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

        if self.mode == 'inference':
            video_name, frame_nmbr, cell_id = self.keys[idx].split('.')[0].split('/')
            label = labels[video_name][frame_nmbr][cell_id]
            return self.transform(img), torch.tensor(label)

        view1 = self.transform(img)
        view2 = self.transform(img)
        return view1, view2