from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch

class DummyDataset(Dataset):
    """Dummy dataset."""

    def __init__(self, size = 100):

        image = Image.new("RGB", (360, 240), color=0)
        self.images = np.array([image] * size)
        self.labels = np.array([0] * size)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image = (
            np.asarray(self.images[idx])
                .reshape([360, 240, 3])
                .astype(np.int8)
        )
        label = self.labels[idx]
        sample = {'image': image, 'label': label}

        return sample