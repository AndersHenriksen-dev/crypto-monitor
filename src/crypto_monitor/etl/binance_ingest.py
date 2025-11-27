import logging
import requests
import pandas as pd
from datetime import datetime

from crypto_monitor.utils.logging_utils import log_function_execution


BINANCE_TICKER_URL = 'https://api.binance.com/api/v3/ticker/price'


@log_function_execution(level=logging.DEBUG)
def fetch_symbol_price(symbol: str = 'BTCUSDT') -> pd.DataFrame:

    params = {'symbol': symbol}
    r = requests.get(BINANCE_TICKER_URL, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()

    # Example response: {"symbol":"BTCUSDT","price":"12345.67"}
    df = pd.DataFrame([{
        'symbol': data['symbol'],
        'price': float(data['price']),
        'timestamp_utc': datetime.utcnow()
    }])
    return df


if __name__ == '__main__':
    print(fetch_symbol_price())
