import pytorch_lightning as pl
from data.data import CarsDataModule
from models.model import LitModel

def train():

    dm = CarsDataModule(batch_size=32)
    model = LitModel((3, 300, 300), 196, transfer=True)
    trainer = pl.Trainer(max_epochs=10, accelerator="cpu")
    trainer.fit(model, dm)
