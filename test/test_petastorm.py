from carsifier.petastorm.dataset import generate_dataset, make_dataloader
from fixtures.spark import spark, dataframe
from fixtures.dataset import DummyDataset
import logging


def test_petastorm_from_dataset(spark):

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


def test_petastorm_to_dataloader(dataframe, spark):
    dataloader = make_dataloader(dataframe, spark = spark)
    logging.info(f"Dataset sample: {list(dataloader)[0]}")
    assert dataloader
    
