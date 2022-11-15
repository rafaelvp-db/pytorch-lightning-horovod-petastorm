from petastorm.codecs import CompressedImageCodec, ScalarCodec
from petastorm.etl.dataset_metadata import materialize_dataset
from petastorm.unischema import Unischema, UnischemaField, dict_to_spark_row
from pyspark.sql.types import IntegerType
import numpy as np
import logging

CarsSchema = Unischema(
    'ScalarSchema',
    [
        UnischemaField('image', np.int8, (360, 240, 3), CompressedImageCodec("png"), False),
        UnischemaField('label', np.int8, (), ScalarCodec(IntegerType()), False)
    ]
)


def row_generator(idx, dataset):
    return dataset[idx]


def generate_dataset(output_url, dataset, spark):
    rowgroup_size_mb = 8
    rows_count = len(dataset)
    logging.info(f"Dataset sample: {dataset[0]}")
    logging.info(f"Dataset length: {rows_count}")
    schema = CarsSchema.as_spark_schema()
    logging.info(f"Schema: {schema}")
    parquet_files_count = 16
    sc = spark.sparkContext

    # Wrap dataset materialization portion. Will take care of setting up spark environment variables as
    # well as save petastorm specific metadata
    try:
        with materialize_dataset(spark, output_url, CarsSchema, rowgroup_size_mb):
            rows_list = [
                dict_to_spark_row(CarsSchema, dataset[idx])
                for idx in range(0, rows_count)
            ]

            spark.createDataFrame(rows_list, CarsSchema.as_spark_schema()) \
                .repartition(parquet_files_count) \
                .write \
                .mode('overwrite') \
                .parquet(output_url)

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        raise e