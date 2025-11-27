def create_lag_features(df, price_col='price', lags=(1, 3, 6, 12)):
    df = df.sort_values('timestamp_utc').copy()

    for lag in lags:
        df[f'lag_{lag}'] = df[price_col].shift(lag)

    df['return_1'] = df[price_col].pct_change(1)
    df = df.dropna()

    return df
