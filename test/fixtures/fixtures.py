from pyspark.sql import SparkSession
import pytest


@pytest.fixture
def spark(scope = 'session'):
    session = SparkSession.builder.getOrCreate()
    return session
