from carsifier.petastorm.dataset import generate_dataset
from fixtures.spark import spark
from fixtures.dataset import DummyDataset


def test_petastorm(spark):

    dataset = DummyDataset()
    generate_dataset(
        output_url = 'file:///tmp/test.parquet',
        dataset = dataset,
        spark = spark
    )
