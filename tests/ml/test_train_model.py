import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from crypto_monitor.ml import train_model


# Sample DataFrame for testing
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "timestamp_utc": pd.date_range("2025-01-01", periods=3, freq="H"),
        "symbol": ["BTCUSDT"] * 3,
        "price": [100, 101, 102],
        "lag_1": [99, 100, 101],
        "lag_3": [97, 98, 99],
        "return_1": [0.01, 0.01, 0.01],
    })


def test_load_latest_model():
    mock_model = MagicMock()
    with patch("mlflow.pyfunc.load_model", return_value=mock_model) as mock_load:
        model = train_model.load_latest_model("test_model")
        mock_load.assert_called_once_with("models:/test_model/latest")
        assert model == mock_model


# def test_load_features_with_spark(sample_df):
#     mock_spark = MagicMock()
#     mock_df = MagicMock()
#     mock_df.toPandas.return_value = sample_df

#     mock_spark.read.format.return_value.load.return_value.filter.return_value.orderBy.return_value = mock_df

#     with patch("crypto_monitor.ml.train_model.SparkSession.builder.getOrCreate", return_value=mock_spark):
#         result = train_model.load_features("dummy_path", "BTCUSDT")
#         pd.testing.assert_frame_equal(result, sample_df)


def test_load_features_local_fallback(sample_df):
    with patch("pandas.read_parquet", return_value=sample_df):
        result = train_model.load_features("dummy_path", "BTCUSDT")
        pd.testing.assert_frame_equal(result, sample_df[sample_df["symbol"] == "BTCUSDT"])


# def test_shift_lag_row(sample_df):
#     row = sample_df.iloc[-1:].copy()
#     new_value = 105
#     shifted = train_model.shift_lag_row(row, new_value)

#     # lag_1 should be updated to new_value
#     assert shifted["lag_1"].values[0] == new_value
#     # lag_3 should be updated to previous lag_2 (here lag_2 is not present, so it remains)
#     assert "lag_3" in shifted.columns
#     # return_1 should be correctly updated
#     expected_return = (new_value - row["lag_1"].values[0]) / (row["lag_1"].values[0] + 1e-9)
#     assert abs(shifted["return_1"].values[0] - expected_return) < 1e-9


# def test_make_forecast(sample_df):
#     mock_model = MagicMock()
#     mock_model.predict.side_effect = lambda x: [x["lag_1"].values[0] + 1]

#     preds = train_model.make_forecast(mock_model, sample_df, horizon=3)
#     assert len(preds) == 3
#     # Ensure predictions are numeric
#     assert all(isinstance(p, (int, float)) for p in preds)


def test_run_function(sample_df):
    mock_model = MagicMock()
    mock_model.predict.return_value = [100]

    with patch("crypto_monitor.ml.train_model.load_latest_model", return_value=mock_model), \
         patch("crypto_monitor.ml.train_model.load_features", return_value=sample_df), \
         patch("crypto_monitor.ml.train_model.make_forecast", return_value=[100, 101]):
        preds = train_model.run(delta_path="dummy", symbol="BTCUSDT", horizon=2, model_name="test")
        assert preds == [100, 101]
