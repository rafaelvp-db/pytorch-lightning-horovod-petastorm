from carsifier.petastorm.dataset import generate_dataset
from fixtures.fixtures import spark
from carsifier.data.data import CarsDataModule


def test_petastorm(spark):

    data_module = CarsDataModule()
    dataset = data_module.train_dataloader().dataset
    generate_dataset(
        output_url = '/tmp',
        dataset = dataset,
        spark = spark
    )
