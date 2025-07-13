from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms as T
import json
from typing import Tuple
import io
import torch
import lmdb


_json_path = Path('video_lists/train_test_split.json')

# Load the file
with open(_json_path, 'r') as f:
    _split_data = json.load(f)

# Access the train and test entries
train_list = _split_data.get("train", [])
test_list = _split_data.get("test", [])
val_list = _split_data.get("val", [])

_reduced_path = Path('video_lists/reduced_videos.json')
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
    def __init__(self, video_list=_reduced_list, path_to_videos=dst_root, transform=default_transform, mode='training', label_mode='dead_alive', num_frames_labels=0):
        self.env = lmdb.open(str(path_to_videos), readonly=True, lock=False)
        with self.env.begin() as txn:
            lmdb_keys = txn.get(b"__keys__").decode().split("\n")
        self.init_labels(label_mode=label_mode, num_frames=num_frames_labels)
        self.transform = transform
        self.keys = []
        self.mode = mode
        video_names = [video["name"] for video in video_list]

        for key in lmdb_keys:
            for video_name in video_names:
                if key.startswith(f"{video_name}/"):
                    self.keys.append(key)
        print(len(self.keys))

    def init_labels(self, label_mode='dead_alive', num_frames=0):
        self.labels = {}

        for ann in annos:
            video_id = ann["video_id"]
            image_id = ann["image_id"]
            cell_id = str(ann["cell_id"]).zfill(3)

            frame_nmbr = image_by_id[image_id]['file_name'][-7:-4]
            video_name = video_id2name[video_id]
            
            # if there is no entry yet in the dictionary
            if video_name not in self.labels:
                self.labels[video_name] = {}

            if frame_nmbr not in self.labels[video_name]:
                self.labels[video_name][frame_nmbr] = {}

            # if this cell_id doesnt yet have a label for that frame
            if cell_id not in self.labels[video_name][frame_nmbr]:

                # label mode frames_till_death returns amount of frames until the cell dies 
                if label_mode == 'frames_till_death':
                    if ann['time_of_death'] is None:
                        self.labels[video_name][frame_nmbr][cell_id] = 200
                    else:
                        self.labels[video_name][frame_nmbr][cell_id] = ann['time_of_death'] - ann['time_step']

                # label mode dead_alive_dividing returns either 0: dead, 1: alive, 2: dividing
                elif label_mode == 'dead_alive_dividing':
                    if ann['time_of_death'] is None:
                        if ann['time_of_division'] is None:
                            self.labels[video_name][frame_nmbr][cell_id] = 1 # alive
                        else:
                            if ann['time_step'] < ann['time_of_division'] - num_frames:
                                self.labels[video_name][frame_nmbr][cell_id] = 2 # dividing
                            else:
                                self.labels[video_name][frame_nmbr][cell_id] = 1 # alive
                    else:
                        if ann['time_step'] < ann['time_of_death'] - num_frames:
                                self.labels[video_name][frame_nmbr][cell_id] = 1 # alive
                        else:
                            self.labels[video_name][frame_nmbr][cell_id] = 0 # dead

                # label mode dead_alive returns either 0: dead, 1: alive
                elif label_mode == 'dead_alive':
                    if ann['time_of_death'] is None:
                        self.labels[video_name][frame_nmbr][cell_id] = 1 # alive
                    else:
                        if ann['time_step'] < ann['time_of_death'] - num_frames:
                            self.labels[video_name][frame_nmbr][cell_id] = 1 # alive
                        else:
                            self.labels[video_name][frame_nmbr][cell_id] = 0 # dead
                
                # if given label mode is not implemented
                else:
                    raise ValueError(f"Unsupported label_mode: '{label_mode}'. "
                                     f"Expected one of: 'frames_till_death', 'dead_alive_dividing', 'dead_alive'.")


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
            label = self.labels[video_name][frame_nmbr][cell_id]
            return self.transform(img), torch.tensor(label)

        view1 = self.transform(img)
        view2 = self.transform(img)
        return view1, view2