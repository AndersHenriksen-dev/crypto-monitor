# minimal helper for Spark session in Databricks context
from pyspark.sql import SparkSession
from py4j.protocol import Py4JJavaError




def get_spark():
    """On Databricks, a SparkSession is already available as `spark`, but this helper returns one for local runs and ensures compatibility."""

    try:
        from pyspark.sql import SparkSession as _SS
        return _SS.builder.getOrCreate()

    except (ImportError, Py4JJavaError, OSError) as e:
        return SparkSession.builder.master('local[*]').appName('crypto-monitor').getOrCreate()
