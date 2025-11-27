import os
import pandas as pd
import pyarrow.parquet as pq
import pytest

# Path to the generated Parquet file
OUTPUT_PATH = "data"
PARQUET_FILE = os.path.join(OUTPUT_PATH, "data.parquet")


def test_parquet_file_exists():
    """Test that the Parquet file is created."""
    assert os.path.exists(PARQUET_FILE), f"{PARQUET_FILE} does not exist."


def test_parquet_file_columns():
    """Test that the Parquet file has the expected columns."""
    table = pq.read_table(PARQUET_FILE)
    df = table.to_pandas()
    expected_columns = {"timestamp_utc", "symbol", "price", "volume"}
    assert set(df.columns) == expected_columns, f"Columns mismatch: {df.columns}"


def test_parquet_file_row_count():
    """Test that the Parquet file has 200 rows."""
    table = pq.read_table(PARQUET_FILE)
    df = table.to_pandas()
    assert len(df) == 200, f"Expected 200 rows, got {len(df)}"


def test_timestamps_sorted():
    """Test that timestamps are in ascending order."""
    table = pq.read_table(PARQUET_FILE)
    df = table.to_pandas()
    assert df["timestamp_utc"].is_monotonic_increasing, "Timestamps are not sorted ascending"


def test_symbol_column_values():
    """Test that the symbol column contains only 'BTCUSDT'."""
    table = pq.read_table(PARQUET_FILE)
    df = table.to_pandas()
    assert df["symbol"].eq("BTCUSDT").all(), "Symbol column contains values other than 'BTCUSDT'"


if __name__ == "__main__":
    pytest.main([__file__])
