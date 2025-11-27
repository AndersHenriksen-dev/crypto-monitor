import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timedelta
import numpy as np
import os

output_path = "data"
os.makedirs(output_path, exist_ok=True)

# Generate 200 mock price rows at 1-minute intervals
now = datetime.utcnow()
timestamps = [now - timedelta(minutes=i) for i in range(200)]
timestamps.reverse()

df = pd.DataFrame({
    "timestamp_utc": timestamps,
    "symbol": ["BTCUSDT"] * 200,
    "price": np.linspace(45000, 46000, 200) + np.random.normal(0, 80, 200),
    "volume": np.random.uniform(10, 300, 200),
})

# Write a Parquet file (similar to Delta table's underlying storage)
pq.write_table(pa.Table.from_pandas(df), f"{output_path}/data.parquet")

print(f"Mock price data created at {output_path}/data.parquet")
