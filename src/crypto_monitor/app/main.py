from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from crypto_monitor.app.model_loader import load_registered_model


app = FastAPI(title='Crypto Monitor API')

MODEL = None


class ForecastRequest(BaseModel):
    symbol: Optional[str] = 'BTCUSDT'
    horizon: Optional[int] = 24


@app.on_event('startup')
async def startup_event():
    global MODEL
    MODEL = load_registered_model()




@app.get('/health')
async def health():
    return {'status': 'ok'}




@app.get('/latest')
async def latest(symbol: str = 'BTCUSDT'):
    """In Databricks this would query the Delta table; we keep it simple here."""
    try:
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()

        sdf = spark.read.format('delta').load('/tmp/crypto/prices')
        pdf = sdf.filter(sdf.symbol == symbol).toPandas()
        row = pdf.sort_values('timestamp_utc').iloc[-1]
        return {'symbol': symbol, 'price': float(row['price']), 'timestamp': str(row['timestamp_utc'])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@app.post('/forecast')
async def forecast(req: ForecastRequest):
    global MODEL
    if MODEL is None:
        raise HTTPException(status_code=500, detail='Model not loaded')


    # A simple example: read last N rows, construct lag features and predict for horizon=1
    try:
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()
        spark.sparkContext.setLogLevel("WARN")

        sdf = spark.read.format('delta').load('/tmp/crypto/prices')
        pdf = sdf.filter(sdf.symbol == req.symbol).toPandas()
        pdf = pdf.sort_values('timestamp_utc')


        # create features like in training
        from crypto_monitor.ml.prepare_features import create_lag_features
        feat = create_lag_features(pdf)
        X = feat.drop(columns=['timestamp_utc', 'symbol', 'price']).iloc[-1:]


        pred = MODEL.predict(X)
        return {'symbol': req.symbol, 'horizon': req.horizon, 'prediction': float(pred[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
