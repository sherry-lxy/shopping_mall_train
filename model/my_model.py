import numpy as np
import pytorch_lightning as pl
import timm
import torch
import torch.nn.functional as F
import torchvision
from loss.triplet_loss import TripletLoss
# from model.normalization import L2_norm
from model.pooling import GeMPooling
from torch import nn, optim

# from torchvision.models import ResNet152_Weights

# '/mnt/vmlqnap02/home/li/main/GeMPooling/data/networks/gl18-tl-resnet152-gem-w-21278d5.pth'


def my_model():
    # model = torchvision.models.resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
    # model.avgpool = GeMPooling()

    # # model.fc = nn.Linear(in_features=model.fc.in_features, out_features=1024)
    # model.fc = L2_norm()
    # # model.add_module('dropout', nn.Dropout(0.3))
    # # model.add_module('Norm1d', nn.InstanceNorm1d(64, affine=True))
    # # model.add_module('fc2', nn.Linear(1024, 512))
    # model.add_module('fc2', nn.Linear(2048, 1024))

    # # fc_in_features = model.fc.in_features
    # # print(model.fc.in_features)
    # model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=1024)

    model = timm.create_model("swin_small_patch4_window7_224", pretrained=True)
    model.head = nn.Linear(768, 1024, bias=True)

    return model


class MyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.model = my_model()
        self.criterion = TripletLoss()
        # self.l2_norm = L2_norm()

    # 推論時の処理
    def forward(self, x):
        x = self.model(x)
        # out = self.l2_norm(x)

        return x

    def training_step(self, batch, _):
        # 画像を取り出す
        anchor = self.forward(batch["anchor"])
        positive = self.forward(batch["positive"])
        negative = self.forward(batch["negative"])

        # triplet loss 計算
        train_loss = self.criterion(anchor, positive, negative)

        self.log("loss/train", train_loss, on_step=False, on_epoch=True)

        return train_loss

    def validation_step(self, batch, _):
        anchor = self.forward(batch["anchor"])
        positive = self.forward(batch["positive"])
        negative = self.forward(batch["negative"])

        # triplet loss 計算
        val_loss = self.criterion(anchor, positive, negative)

        self.log("loss/val", val_loss, on_step=False, on_epoch=True)

        return {"anchor": anchor, "positive": positive, "negative": negative}

    def validation_epoch_end(self, outputs):
        anchor = (
            torch.cat([output["anchor"] for output in outputs], dim=0).cpu().numpy()
        )
        positive = (
            torch.cat([output["positive"] for output in outputs], dim=0).cpu().numpy()
        )
        negative = (
            torch.cat([output["negative"] for output in outputs], dim=0).cpu().numpy()
        )

        anchor_positive = np.diag(
            np.dot(anchor, positive.T)
            / (np.linalg.norm(anchor, axis=1) * np.linalg.norm(positive, axis=1))
        )
        anchor_negative = np.diag(
            np.dot(anchor, negative.T)
            / (np.linalg.norm(anchor, axis=1) * np.linalg.norm(negative, axis=1))
        )

        self.log(
            "acc/val",
            (anchor_positive > anchor_negative).sum() / len(anchor),
            on_step=False,
            on_epoch=True,
        )

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=5e-7)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[30, 70], gamma=0.1
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
