"""
Databricks Job entrypoint (python file) that appends fetched prices to a Delta table.
This is meant to run as a Job task (python file task) or as part of a wheel.
"""
import os
from crypto_monitor.etl.binance_ingest import fetch_symbol_price
from crypto_monitor.utils.logging_utils import setup_logging
from crypto_monitor.utils.spark import get_spark
from crypto_monitor.utils.logging_utils import get_logger, setup_logging, log_function_execution


logger = get_logger(__name__)


@log_function_execution
def main():
    logger.info("Starting ingestion job run.")
    logger.debug("Setting up spark.")
    spark = get_spark()

    symbol = os.environ.get('SYMBOL', 'BTCUSDT')
    logger.debug("Fetching symbol price.")
    df = fetch_symbol_price(symbol)

    logger.debug("Converting from pandas dataframe to spark.")
    sdf = spark.createDataFrame(df)

    logger.debug("Writing to the delta table.")
    delta_path = os.environ.get('DELTA_PATH', 'data')

    sdf.write.format('delta').mode('append').save(delta_path)
    logger.info(f'Appended price for {symbol} to {delta_path}.')


if __name__ == '__main__':
    setup_logging("")
    main()
