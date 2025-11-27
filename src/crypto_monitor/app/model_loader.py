import mlflow




def load_registered_model(model_name: str = 'crypto_forecast', stage: str = 'Production'):
    """Use Databricks-managed MLflow model registry URI like 'models:/<name>/<stage>'"""
    model_uri = f'models:/{model_name}/{stage}'
    print(f'Loading model from {model_uri}')
    model = mlflow.pyfunc.load_model(model_uri)

    return model