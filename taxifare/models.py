import json
from typing import Iterable, Optional, Set, Union, Tuple
from itertools import chain

import polars as pl
from tensorflow import keras

from taxifare import data

ColumnNames = Iterable[str]
DataOrLazyFrame = Union[pl.DataFrame, pl.LazyFrame]


class DataPolicy:
    """Describe feature normalization, input feature and output vars.

    Specifically designed to resemble ``scikit-learn`` interface, but
    to be applied to polars dataframes.

    Can be split in two objects: a normalizer that fits on all features,
    an "applier" that transforms the transformation on the specified
    subset of columns. For the purpose of this project keeping a whole
    class should be fine.
    """

    def __init__(self, to_std: ColumnNames, to_norm: ColumnNames,
                 to_input: ColumnNames, to_output: ColumnNames):
        self.to_std = tuple(to_std)
        self.to_norm = tuple(to_norm)
        self.to_input = tuple(to_input)
        self.to_output = tuple(to_output)

        self.mean_dataframe: Optional[pl.DataFrame] = None
        self.std_dataframe: Optional[pl.DataFrame] = None
        self.min_dataframe: Optional[pl.DataFrame] = None
        self.max_dataframe: Optional[pl.DataFrame] = None

    @classmethod
    def from_file(cls, filename: str) -> 'DataPolicy':
        """Load a policy from file.

        Previously saved with :func:`to_file`.
        """
        with open(filename) as file:
            data_dict = json.load(file)

        policy = cls(data_dict['to_std'], data_dict['to_norm'],
                     data_dict['to_input'], data_dict['to_output'])
        policy.mean_dataframe = pl.from_dict(data_dict['mean'])
        policy.std_dataframe = pl.from_dict(data_dict['std'])
        policy.min_dataframe = pl.from_dict(data_dict['min'])
        policy.max_dataframe = pl.from_dict(data_dict['max'])

        return policy

    def to_file(self, filename: str):
        """Save policy to file."""
        data_dict = {
            'mean': self.mean_dataframe.to_dict(False),
            'std': self.std_dataframe.to_dict(False),
            'min': self.min_dataframe.to_dict(False),
            'max': self.max_dataframe.to_dict(False),
            'to_std': self.to_std,
            'to_norm': self.to_norm,
            'to_input': self.to_input,
            'to_output': self.to_output
        }

        with open(filename, 'w') as file:
            json.dump(data_dict, file)

    def missing_columns(self, df: DataOrLazyFrame) -> Set[str]:
        """Return requested columns that were not found in ``df``.

        If the columns do not match, it will not be possible to use
        the given dataframe with this policy.
        """
        return (
            set(chain(self.to_input, self.to_output, self.to_std,
                      self.to_norm))
            .difference(df.columns)
        )

    def fit(self, df: DataOrLazyFrame):
        """Fit internal statistics on the given dataframe."""
        missing_columns = self.missing_columns(df)
        assert not missing_columns, ('Given dataframe does not provide the '
                                     f'necessary columns {missing_columns}')

        df = df.lazy()
        self.mean_dataframe = df.select(pl.col(self.to_std).mean()).collect()
        self.std_dataframe = df.select(pl.col(self.to_std).std()).collect()
        self.max_dataframe = df.select(pl.col(self.to_norm).max()).collect()
        self.min_dataframe = df.select(pl.col(self.to_norm).min()).collect()

    def transform(self, df: DataOrLazyFrame) -> pl.DataFrame:
        """Apply stored transformations to the given dataframe."""
        missing_columns = self.missing_columns(df)
        assert not missing_columns, ('Given dataframe does not provide the '
                                     f'necessary columns {missing_columns}')

        lazyframe = df.lazy()

        # Skip transformations if uninitialized
        if self.mean_dataframe is not None and self.std_dataframe is not None:
            lazyframe = lazyframe.with_columns(
                **data.standardize(self.mean_dataframe, self.std_dataframe))

        # Skip transformations if uninitialized
        if self.min_dataframe is not None and self.max_dataframe is not None:
            lazyframe = lazyframe.with_columns(
                **data.normalize(self.min_dataframe, self.max_dataframe))

        return lazyframe.collect()

    def split_transform(self, df: DataOrLazyFrame) -> Tuple[pl.DataFrame,
                                                            pl.DataFrame]:
        """Apply transformations and return splitted input, output."""
        transformed = self.transform(df)
        return (transformed.select(self.to_input),
                transformed.select(self.to_output))


def iterate_df(dataframe: pl.DataFrame, batch_size=128, cycle=True,
               data_policy: DataPolicy = None):
    """Iterate by given batch size and divide inputs from targets.

    Values are standardized/normalized according to the given data
    policy. The policy must be fit before feeding it to this function.

    Yield a pair (inputs, targets) that can be used to feed keras fit
    function.
    """
    while True:
        for batch_start in range(0, len(dataframe), batch_size):
            batch = dataframe[batch_start : batch_start + batch_size]   # NOQA
            if data_policy is not None:
                batch = data_policy.transform(batch)

            # By default input columns are all columns except the first,
            # target column is just the first one
            input_columns = dataframe.columns[1:]
            target_columns = dataframe.columns[0],
            # If a policy is provided, use that instead
            if data_policy is not None:
                input_columns = data_policy.to_input
                target_columns = data_policy.to_output

            yield (batch.select(input_columns).to_numpy(),
                   batch.select(target_columns).to_numpy())

        if not cycle:
            break


MLP_TO_STD = 'fare_amount',
MLP_TO_NORM = ('pickup_longitude', 'pickup_latitude',
               'dropoff_longitude', 'dropoff_latitude', 'passenger_count',
               'year', 'month', 'weekday', 'hour')

MLP_TO_INPUT = ('pickup_longitude', 'pickup_latitude',
                'dropoff_longitude', 'dropoff_latitude',
                'passenger_count', 'year', 'month', 'weekday', 'hour',
                'after2012', 'weekend')
MLP_TO_OUTPUT = 'fare_amount',

MLP_DATA_POLICY = DataPolicy(MLP_TO_STD, MLP_TO_NORM, MLP_TO_INPUT,
                             MLP_TO_OUTPUT)
"""Empty data policy, can be used during training to fit and save it."""

MLP_URBAN_TO_NORM = MLP_TO_NORM + ('travel_time',)
MLP_URBAN_TO_INPUT = MLP_TO_INPUT + ('travel_time',)

MLP_URBAN_DATA_POLICY = DataPolicy(MLP_TO_STD, MLP_URBAN_TO_NORM,
                                   MLP_URBAN_TO_INPUT, MLP_TO_OUTPUT)
"""Empty data policy, can be used during training to fit and save it."""


def get_cone_mlp(input_nodes, output_nodes, first_layer_nodes,
                 num_hidden_layers, activation='relu',
                 name='MLP') -> keras.Model:
    """Build a MLP with descending number of nodes.

    Each layer halves the number of nodes.
    """
    layers = [keras.layers.InputLayer((input_nodes,))]

    nodes = first_layer_nodes
    for _ in range(num_hidden_layers):
        layers.append(keras.layers.Dense(nodes, activation=activation))
        nodes //= 2

    layers.append(keras.layers.Dense(output_nodes, activation='linear'))

    return keras.Sequential(layers, name)
