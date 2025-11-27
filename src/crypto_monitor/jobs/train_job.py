from math import log
import pandas as pd
import mlflow
from pyspark.sql import SparkSession
from prophet import Prophet
from py4j.protocol import Py4JJavaError
from requests import get
import logging

from crypto_monitor.utils.logging_utils import setup_logging, get_logger, log_function_execution


logger = get_logger(__name__)


@log_function_execution()
def load_training_data():
    try:
        spark = SparkSession.builder.getOrCreate()

        df = (
            spark.read.table("crypto.features")
            .orderBy("timestamp")
        )
        pdf = df.toPandas()
        pdf = pdf.rename(columns={"timestamp_utc": "ds", "price": "y"})

    except (ImportError, Py4JJavaError, OSError) as e:
        # Local fallback: Parquet / CSV
        pdf = pd.read_parquet("data/")
        pdf = pdf.rename(columns={"timestamp_utc": "ds", "price": "y"})
        pdf = pdf.sort_values("ds")

    return pdf


@log_function_execution()
def train_model(training_data):
    model = Prophet()
    model.fit(training_data)
    return model


@log_function_execution()
def log_model(model, model_name: str = "crypto_forecaster"):
    with mlflow.start_run() as run:
        mlflow.prophet.log_model(model, artifact_path="model")
        mlflow.register_model(
            f"runs:/{run.info.run_id}/model",
            model_name
        )


def run():
    training_data = load_training_data()
    model = train_model(training_data)
    log_model(model)


if __name__ == "__main__":
    setup_logging("")
    run()
