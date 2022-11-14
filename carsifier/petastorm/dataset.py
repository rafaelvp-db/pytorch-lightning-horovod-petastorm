from petastorm.codecs import CompressedImageCodec, ScalarCodec
from petastorm.etl.dataset_metadata import materialize_dataset
from petastorm.unischema import Unischema, UnischemaField, dict_to_spark_row
from pyspark.sql.types import IntegerType
import numpy as np

CarsSchema = Unischema(
    'ScalarSchema',
    [
        UnischemaField('image', np.uint8, (256, 256, 3), CompressedImageCodec("png"), False),
        UnischemaField('label', np.int8, (), ScalarCodec(IntegerType()), False)
    ]
)


def row_generator(idx, dataset):
    return dataset[idx]


def generate_dataset(output_url, dataset, spark):
    rowgroup_size_mb = 8
    rows_count = len(dataset)
    parquet_files_count = 16
    
    sc = spark.sparkContext
    # Wrap dataset materialization portion. Will take care of setting up spark environment variables as
    # well as save petastorm specific metadata
    with materialize_dataset(spark, output_url, CarsSchema, rowgroup_size_mb):
        rows_rdd = sc.parallelize(range(rows_count))\
            .map(lambda x: row_generator(x, dataset))\
            .map(lambda x: dict_to_spark_row(CarsSchema, x))

        spark.createDataFrame(rows_rdd, CarsSchema.as_spark_schema()) \
            .repartition(parquet_files_count) \
            .write \
            .mode('overwrite') \
            .parquet(output_url)