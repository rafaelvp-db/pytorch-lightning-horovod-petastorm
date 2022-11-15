from pyspark.sql import SparkSession, functions as f
import pytest
from PIL import Image


@pytest.fixture
def spark(scope = 'session'):
    session = SparkSession.builder.master("local[*]").getOrCreate()
    yield session
    session.stop()


@pytest.fixture
def dataframe(spark):
    image = Image.new("RGB", (360, 240), color=0)
    path = "/tmp/dummy.png"
    image.save(path, format = "png")
    image_df = spark.read.format("binaryFile").load(path)
    image_df = image_df \
        .select("content") \
        .withColumnRenamed("content", "image") \
        .withColumn("label", f.lit(0))
    return image_df
