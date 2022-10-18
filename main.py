import os

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from data.my_dataloader import my_dataloader
from model.my_model import MyModel

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

train_loader, val_loader = my_dataloader()

lr_monitor = LearningRateMonitor(logging_interval="epoch")
trainer = Trainer(gpus=1, max_epochs=150, callbacks=[lr_monitor])
model = MyModel()
# weight_path = "/mnt/vmlqnap02/home/li/main/GeMPooling/data/networks/gl18-tl-resnet152-gem-w-21278d5.pth"
# model.load_state_dict(torch.load(weight_path)['model'])
# trainer.fit(model, train_loader, val_loader)

torch.jit.save(model.to_torchscript(file_path="model.pth"))
