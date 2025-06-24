from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms as T
import json
from typing import Tuple


_json_path = Path('train_test_split.json')

# Load the file
with open(_json_path, 'r') as f:
    _split_data = json.load(f)

# Access the train and test entries
train_list = _split_data.get("train", [])
test_list = _split_data.get("test", [])
val_list = _split_data.get("val", [])

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
dst_root   = Path(base_path + '/preprocessed_dataset')

class CellDataset(Dataset):
    def __init__(self, video_list=train_list, path_to_videos=dst_root, transform=T.ToTensor()):
        self.image_paths = []
        self.path_to_videos = path_to_videos
        self.transform = transform

        for video_dict in video_list:
            dst_path = path_to_videos / str(video_dict['name']) / 'images'
            self.image_paths += dst_path.glob('*.jpg')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        :param idx: Inter
        :return: Item at idx after applying transform
        :rtype: Tuple[torch.Tensor]
        """
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")

        view1 = self.transform(img)
        view2 = self.transform(img)
        return view1, view2