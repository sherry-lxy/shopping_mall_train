import json
import os
import pathlib
import random

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CustomDataset(Dataset):
    def __init__(
        self,
        db_root="/home/li/main/dataset/shopping_mall/reference_rename",
        annotation_json="/home/li/main/dataset/shopping_mall/json/annotation_test.json",
        is_train=True,
    ):
        self.is_train = is_train
        self.db_root = db_root
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                transforms.RandomAffine([-30, 30], scale=(0.8, 1.2), shear=10),
                transforms.RandomErasing(
                    p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False
                ),
                transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
                transforms.RandomAdjustSharpness(sharpness_factor=3, p=0.5),
            ]
        )

        # annotation のファイルを読み込む
        json_open = open(annotation_json, "r")
        # 開いたファイルをJSONとして読み込む
        self.annotation = json.load(json_open)

        # 画像データを読み込み
        self.imgs = sorted(pathlib.Path(self.db_root).glob("*.jpg"))
        self.imgs = [im.stem for im in self.imgs]

        # 学習するデータを決める
        n_train = int(len(self.imgs) * 0.8)

        # 画像をシャッフルする
        random.seed(42)
        random.shuffle(self.imgs)

        if self.is_train:
            self.images = self.imgs[:n_train]
        else:
            self.images = self.imgs[n_train:]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        anchor_img = self.images[item]

        positive_list = self.annotation[anchor_img]
        positive_img = random.choice(positive_list)

        # print(f'positive_list: {positive_list}, images: {self.imgs}')

        negative_list = list(set(positive_list) ^ set(self.imgs))
        negative_img = random.choice(negative_list)

        anchor_img = Image.open(os.path.join(self.db_root, f"{anchor_img}.jpg"))
        anchor_img = self.transform(anchor_img)

        positive_img = Image.open(os.path.join(self.db_root, f"{positive_img}.jpg"))
        positive_img = self.transform(positive_img)

        negative_img = Image.open(os.path.join(self.db_root, f"{negative_img}.jpg"))
        negative_img = self.transform(negative_img)

        return {
            "anchor": anchor_img,
            "positive": positive_img,
            "negative": negative_img,
        }
