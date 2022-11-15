# Databricks notebook source
from carsifier.data.dataset import Cars
from carsifier.petastorm.dataset import generate_dataset

dataset = Cars(
  root = "/tmp",
  download = True,
  split = "train"
)

# COMMAND ----------

generate_dataset(
  output_url = "file:///dbfs/Users/rafael.pierre@databricks.com/cars_train.parquet",
  dataset = dataset,
  spark = spark
)

# COMMAND ----------

dataset[0]

# COMMAND ----------


