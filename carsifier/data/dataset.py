from torchvision.datasets import StanfordCars
import numpy as np


class Cars(StanfordCars):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)

    def __getitem__(self, idx):

        pil_image, target = super().__getitem__(idx)
        image = (
            np.asarray(pil_image)
                .reshape([360, 240, 3])
                .astype(np.int8)
        )
        sample = {'image': image, 'label': target}
        return sample

