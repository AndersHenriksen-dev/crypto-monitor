import os
import mlflow
import pandas as pd
from py4j.protocol import Py4JJavaError

from crypto_monitor.ml.prepare_features import create_lag_features


def load_latest_model(model_name: str):
    model_uri = f"models:/{model_name}/latest"
    return mlflow.pyfunc.load_model(model_uri)


def load_features(delta_path: str, symbol: str):
    """
    Load recent delta data either with Spark (Databricks) or locally via Parquet.
    """
    try:
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()

        df = (
            spark.read.format("delta")
            .load(delta_path)
            .filter(f"symbol = '{symbol}'")
            .orderBy("timestamp_utc")
        )
        pdf = df.toPandas()

    except (ImportError, Py4JJavaError, OSError) as e:
        # Local fallback
        pdf = pd.read_parquet(delta_path)
        pdf = pdf[pdf["symbol"] == symbol]

    return pdf


def make_forecast(model, pdf: pd.DataFrame, horizon: int = 24):
    """
    Compute the same lag features as training, and predict.
    """

    # Compute lags using the SAME function used during training
    pdf = create_lag_features(pdf)

    # Keep only the features used during training
    feature_cols = [
        c for c in pdf.columns
        if c not in ("timestamp_utc", "symbol", "price")
    ]

    # Use the last row (most recent features) to predict forward
    latest_features = pdf[feature_cols].iloc[-1:]

    preds = []
    current_features = latest_features.copy()

    for _ in range(horizon):
        y_pred = model.predict(current_features)[0]
        preds.append(y_pred)

        # Update feature frame for next step
        # Shift lag features manually
        current_features = shift_lag_row(current_features, y_pred)

    return preds


def shift_lag_row(row: pd.DataFrame, new_value: float):
    """
    Updates a lag-feature row after each prediction step.
    """
    new_row = row.copy()

    # Example:
    # lag_1 <- new_value
    # lag_3 <- old lag_1
    # lag_6 <- old lag_3
    # etc.

    for c in row.columns:
        if c.startswith("lag_"):
            lag = int(c.split("_")[1])
            if lag == 1:
                new_row[c] = new_value
            else:
                prev = f"lag_{lag-1}"
                new_row[c] = row[prev].values[0]

    # Return_1 needs last_price
    if "return_1" in row.columns:
        old_price = row["lag_1"].values[0]
        new_row["return_1"] = (new_value - old_price) / (old_price + 1e-9)

    return new_row


def run(
    delta_path="data",
    symbol="BTCUSDT",
    horizon=24,
    model_name="crypto_forecast"
):
    model = load_latest_model(model_name)
    pdf = load_features(delta_path, symbol)
    preds = make_forecast(model, pdf, horizon)
    return preds


if __name__ == "__main__":
    result = run()
    print(result)
