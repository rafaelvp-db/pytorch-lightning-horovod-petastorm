# https://colab.research.google.com/drive/1smfCw-quyKwxlj69bbsqpZhD75CnBuRh?usp=sharing#scrollTo=yaF_kykKUFpk

import pytorch_lightning as pl
from carsifier.data.dataset import Cars
from torch.utils.data import random_split, DataLoader
from torchvision import transforms

class CarsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "/tmp",
        batch_size: int = 32,
        download: bool = True
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.download = download

        # Augmentation policy for training set
        self.augmentation = transforms.Compose([
              transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
              transforms.RandomRotation(degrees=15),
              transforms.RandomHorizontalFlip(),
              transforms.CenterCrop(size=224),
              transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
        # Preprocessing steps applied to validation and test set.
        self.transform = transforms.Compose([
              transforms.Resize(size=256),
              transforms.CenterCrop(size=224),
              transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])

    def prepare_data(self):
        pass

    def setup(self, stage: str = "fit"):
        
        if stage == "fit":
            # build dataset
            dataset = Cars(
                root=self.data_dir,
                split="train",
                download=self.download,
                transform=self.augmentation
            )
            # split dataset
            self.train, self.val = random_split(dataset, [6500, 1644])

        if stage == "test":
            self.test = Cars(
                root=self.data_dir,
                split="test",
                download=self.download,
                transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return self.test_dataloader()
