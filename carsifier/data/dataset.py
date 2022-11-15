from torchvision.datasets import StanfordCars
import numpy as np


class Cars(StanfordCars):
    def __init__(
        root = "/tmp",
        split = "train",
        transform = None,
        target_transform = None,
        download = True
    ):
        super().__init__(
            root,
            split,
            transform,
            target_transform,
            download
        )

    def __getitem__(self, idx):

        pil_image, target = super().__getitem__(idx)
        image = (
            np.asarray(pil_image)
                .reshape([360, 240, 3])
                .astype(np.int8)
        )
        sample = {'image': image, 'label': target}
        return sample

