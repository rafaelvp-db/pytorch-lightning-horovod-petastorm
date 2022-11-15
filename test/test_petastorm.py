from carsifier.petastorm.dataset import generate_dataset
from fixtures.spark import spark
from fixtures.dataset import DummyDataset


def test_petastorm(spark):

    size = 100
    dataset = DummyDataset(size = size)
    path = 'file:///tmp/test.parquet'
    generate_dataset(
        output_url = path,
        dataset = dataset,
        spark = spark
    )
    df = spark.read.parquet(path)
    assert df.count() == size
