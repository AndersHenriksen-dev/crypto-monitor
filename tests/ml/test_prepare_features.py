import pytest
import pandas as pd
from crypto_monitor.ml.prepare_features import create_lag_features

# --------------------------
# Fixtures
# --------------------------


@pytest.fixture
def sample_df():
    # Simple DataFrame with prices
    return pd.DataFrame({
        "timestamp_utc": pd.date_range("2025-11-27", periods=6, freq="T"),
        "price": [100, 101, 102, 103, 104, 105]
    })

# --------------------------
# Tests
# --------------------------


def test_create_lag_features_creates_lags(sample_df):
    df_lagged = create_lag_features(sample_df, lags=(1, 2))

    # Check that lag columns exist
    assert "lag_1" in df_lagged.columns
    assert "lag_2" in df_lagged.columns
    assert "return_1" in df_lagged.columns

    # Check that the number of rows is correct (dropna)
    assert len(df_lagged) == len(sample_df) - 2  # max lag is 2, so first 2 rows dropped


def test_lag_values_correct(sample_df):
    df_lagged = create_lag_features(sample_df, lags=(1, 2))

    # Row index 0 in df_lagged corresponds to original index 2
    assert df_lagged["lag_1"].iloc[0] == sample_df["price"].iloc[1]  # previous row
    assert df_lagged["lag_2"].iloc[0] == sample_df["price"].iloc[0]  # 2 rows ago


def test_return_1_correct(sample_df):
    df_lagged = create_lag_features(sample_df, lags=(1, 2))
    # Row index 0 in df_lagged corresponds to original index 2
    expected_return = (sample_df["price"].iloc[2] - sample_df["price"].iloc[1]) / sample_df["price"].iloc[1]
    assert pytest.approx(df_lagged["return_1"].iloc[0], rel=1e-9) == expected_return


def test_sorting_not_altered(sample_df):
    # Shuffle df to test that function sorts correctly
    df_shuffled = sample_df.sample(frac=1, random_state=42)
    df_lagged = create_lag_features(df_shuffled, lags=(1, 2))

    # Check that timestamps are sorted
    assert df_lagged["timestamp_utc"].is_monotonic_increasing
