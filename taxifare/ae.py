"""Outlier detection through autoencoder."""
import math
import os.path

import polars as pl
from tqdm.auto import tqdm
from tensorflow import keras

from taxifare.data import iterate_df
from taxifare.models import DataPolicy

OPTIMAL_THRESHOLD = 0.2011901901901902
TO_STD = ('fare_amount', 'pickup_longitude', 'pickup_latitude',
          'dropoff_longitude', 'dropoff_latitude')
TO_NORM = 'passenger_count', 'year', 'month', 'weekday', 'hour'
# To drop before train to prevent string conversions
# string_columns = 'pickup_borough', 'dropoff_borough'

INPUT_COLUMNS = TO_STD + TO_NORM
TARGET_COLUMNS = ('fare_amount', 'pickup_longitude', 'pickup_latitude',
                  'dropoff_longitude', 'dropoff_latitude', 'passenger_count')

DEFAULT_DATA_POLICY = DataPolicy(TO_STD, TO_NORM, INPUT_COLUMNS,
                                 TARGET_COLUMNS)
"""Empty data policy, can be used during training to fit and save it."""

MODEL_PATH = 'ae.model'
DATA_POLICY_SUBPATH = 'data_policy.json'


def get_policy(filename=os.path.join(MODEL_PATH, DATA_POLICY_SUBPATH)
               ) -> DataPolicy:
    """Get data policy for the autoencoder."""
    return DataPolicy.from_file(filename)


def compute_ree(dataframe: pl.DataFrame, model: keras.Model,
                batch_size, data_policy: DataPolicy = None) -> pl.Series:
    """Compute rec. error (mse) of dataframe samples given an AE model."""
    results = []
    for inputs, targets in tqdm(iterate_df(dataframe,
                                           batch_size=batch_size,
                                           cycle=False,
                                           data_policy=data_policy),
                                total=math.ceil(len(dataframe) / batch_size)):
        pred = model(inputs)
        loss = ((targets - pred) ** 2).numpy().mean(1)
        results.extend(loss)

    return pl.Series(results).alias('ree')
