import os
import pathlib

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class TestDataset(Dataset):
    def __init__(self, db_root):

        self.db_root = db_root
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # 画像データを読み込み
        self.images = sorted(pathlib.Path(self.db_root).glob("*.jpg"))
        self.images = [im.stem for im in self.images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        anchor_img = self.images[item]

        img = Image.open(os.path.join(self.db_root, f"{anchor_img}.jpg"))
        img = self.transform(img)

        return img
