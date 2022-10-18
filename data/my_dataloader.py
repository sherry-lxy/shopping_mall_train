from data.custom_dataset import CustomDataset
from torch.utils.data import DataLoader


def my_dataloader():
    train_set = CustomDataset()
    val_set = CustomDataset(is_train=False)

    train_loader = DataLoader(train_set, batch_size=16, num_workers=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8, num_workers=16, shuffle=True)

    return train_loader, val_loader
