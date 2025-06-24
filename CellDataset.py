from torchvision import transforms as T
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import os
import json

moco_transform = T.Compose([
    T.RandomHorizontalFlip(p=0.5),      # flip left-right with 50% probability
    T.RandomVerticalFlip(p=0.5),        # flip top-bottom with 50% probability
    T.RandomRotation(degrees=10),       # rotate randomly between -10° and +10°
    T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # slight blur
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],  # or use dataset-specific values
                std=[0.229, 0.224, 0.225]),
])

class CellDataset(Dataset):
    def __init__(self, video_list, path_to_videos, transform):
        self.image_paths = []
        self.path_to_videos = path_to_videos
        self.transform = transform

        for video_dict in video_list:
            dst_path = path_to_videos / str(video_dict['name']) / 'images'
            self.image_paths += dst_path.glob('*.jpg')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            view1 = self.transform(img)
            view2 = self.transform(img)
            return view1, view2

        return img