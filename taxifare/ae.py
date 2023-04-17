"""Outlier detection through autoencoder."""
import math
import os.path

import polars as pl
from tqdm.auto import tqdm
from tensorflow import keras

from taxifare.data import standardize, normalize

OPTIMAL_THRESHOLD = 0.2011901901901902
TO_STD = ('fare_amount', 'pickup_longitude', 'pickup_latitude',
          'dropoff_longitude', 'dropoff_latitude')
TO_NORM = 'passenger_count', 'year', 'month', 'weekday', 'hour'
# To drop before train to prevent string conversions
# string_columns = 'pickup_borough', 'dropoff_borough'

INPUT_COLUMNS = TO_STD + TO_NORM
TARGET_COLUMNS = ('fare_amount', 'pickup_longitude', 'pickup_latitude',
                  'dropoff_longitude', 'dropoff_latitude', 'passenger_count')

MODEL_PATH = 'ae.model'
TRAIN_MEAN_SUBPATH = 'train_mean.csv'
TRAIN_STD_SUBPATH = 'train_std.csv'
TRAIN_MAX_SUBPATH = 'train_max.csv'
TRAIN_MIN_SUBPATH = 'train_min.csv'


def iterate_df(dataframe: pl.DataFrame, input_columns=INPUT_COLUMNS,
               target_columns=TARGET_COLUMNS, batch_size=128, cycle=True):
    """Iterate by given batch size and divide inputs from targets.

    Valuesa are standardized/normalized according to the
    :attr:`TRAIN_MEAN_SUBPATH`, :attr:`TRAIN_STD_SUBPATH`,
    :attr:`TRAIN_MAX_SUBPATH` and :attr:`TRAIN_MIN_SUBPATH`.

    Return a pair (inputs, targets)
    """
    train_mean = pl.read_csv(os.path.join(MODEL_PATH, TRAIN_MEAN_SUBPATH))
    train_std = pl.read_csv(os.path.join(MODEL_PATH, TRAIN_STD_SUBPATH))
    train_max = pl.read_csv(os.path.join(MODEL_PATH, TRAIN_MAX_SUBPATH))
    train_min = pl.read_csv(os.path.join(MODEL_PATH, TRAIN_MIN_SUBPATH))

    while True:
        for batch_start in range(0, len(dataframe), batch_size):
            batch = dataframe[batch_start : batch_start + batch_size]   # NOQA
            batch = batch.with_columns(**standardize(train_mean, train_std),
                                       **normalize(train_min, train_max))

            yield (batch.select(input_columns).to_numpy(),
                   batch.select(target_columns).to_numpy())

        if not cycle:
            break


def compute_ree(dataframe: pl.DataFrame, model: keras.Model,
                batch_size) -> pl.Series:
    """Compute rec. error (mse) of dataframe samples given an AE model."""
    results = []
    for inputs, targets in tqdm(iterate_df(dataframe,
                                           batch_size=batch_size,
                                           cycle=False),
                                total=math.ceil(len(dataframe) / batch_size)):
        pred = model(inputs)
        loss = ((targets - pred) ** 2).numpy().mean(1)
        results.extend(loss)

    return pl.Series(results).alias('ree')
