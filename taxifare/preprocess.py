"""Preprocess the given dataset and dump it to a parquet file.

Entrypoint, run with ``python -m texifare.preprocess [...]``
"""
import argparse
import enum
import pickle
import warnings
from functools import partial
from typing import Optional

import polars as pl
import tensorflow as tf
import osmnx as ox

import taxifare.data as data
import taxifare.boroughs as boroughs
import taxifare.ae as ae

DESCRIPTION = __doc__


class PreprocessingFlags(enum.IntFlag):
    """Types of preprocessing for the data.

    Can be summed with binary OR like int flags: ``PREP1 | PREP2``.
    """
    BASE = 0
    TIME_FEATURES = enum.auto()
    PASSENGER_COUNT_OUTLIERS = enum.auto()
    OCEAN_OUTLIERS = enum.auto()
    BOROUGH_OUTLIERS = enum.auto()
    OCEAN_FEATURES = enum.auto()
    BOROUGH_FEATURES = enum.auto()
    PASSENGER_COUNT_FEATURES = enum.auto()
    REE_OUTLIERS = enum.auto()
    GRAPH_TRAVEL_TIME = enum.auto()


OCEAN_FLAGS = (PreprocessingFlags.OCEAN_FEATURES
               | PreprocessingFlags.OCEAN_OUTLIERS)

BOROUGH_FLAGS = (PreprocessingFlags.BOROUGH_FEATURES
                 | PreprocessingFlags.BOROUGH_OUTLIERS)

PASSENGER_COUNT_FLAGS = (PreprocessingFlags.PASSENGER_COUNT_FEATURES
                         | PreprocessingFlags.PASSENGER_COUNT_OUTLIERS)

FOR_AUTOENCODER_FLAGS = (PreprocessingFlags.TIME_FEATURES
                         | PreprocessingFlags.PASSENGER_COUNT_FEATURES
                         | PreprocessingFlags.BOROUGH_FEATURES
                         | PreprocessingFlags.OCEAN_FEATURES)

FOR_TRAIN_FLAGS = (PreprocessingFlags.TIME_FEATURES
                   | PreprocessingFlags.OCEAN_OUTLIERS
                   | PreprocessingFlags.BOROUGH_OUTLIERS
                   | PreprocessingFlags.PASSENGER_COUNT_OUTLIERS)
#                    | PreprocessingFlags.REE_OUTLIERS)


class Namespace:
    """Custom namespace for CLI parameters."""
    input_path: str
    output_path: str
    preprocessing_flags: PreprocessingFlags = PreprocessingFlags.BASE
    samples: Optional[int] = None
    ree_threshold: float = ae.OPTIMAL_THRESHOLD
    ree_model: Optional[str] = None
    city_graph: Optional[str] = None
    city_distance_matrix: Optional[str] = None


class AddFlagEnumAction(argparse.Action):
    """Custom argparse action: add enum flag (binary or).

    Overrides defaults.
    """

    def __call__(self, parser, namespace, value, option_string=None):
        new_value = getattr(
            namespace, self.dest) | PreprocessingFlags[value]
        setattr(namespace, self.dest, new_value)


def preprocess(namespace: Namespace) -> pl.DataFrame:
    """Process the given arguments and return dataframe."""
    # Initial preprocessing
    df = data.load_data(namespace.input_path)

    # Select specified number of samples
    if namespace.samples is None:
        df = df.collect()
    else:
        df = df.fetch(namespace.samples)

    # Travel time according to NYC graph
    # This requires a lot of working memory, which will be freed after
    # the process is over. Hence, this is done before expanding other
    # features to minimize the total memory footprint.
    if PreprocessingFlags.GRAPH_TRAVEL_TIME in namespace.preprocessing_flags:
        if (namespace.city_graph is None
                or namespace.city_distance_matrix is None):
            raise ValueError('Please provide both a city graph (--city-graph) '
                             'and a distance matrix (--city-distance-matrix)')

        with open(namespace.city_distance_matrix, 'rb') as f:
            warnings.warn('Distance matrix is a pickled file: use only '
                          'from trusted sources')
            distance_matrix = pickle.load(f)

        G = ox.load_graphml(namespace.city_graph)
        kdtree = data.kdtree_from_graph(G)
        del G               # Save ~1GB of memory for the next computation

        df = df.with_columns(
            travel_time=pl.struct('pickup_longitude', 'pickup_latitude',
                                  'dropoff_longitude', 'dropoff_latitude')
            .map(partial(data.fast_travel_distance,
                         distance_matrix=distance_matrix,
                         kdtree=kdtree)))

        del kdtree
        del distance_matrix

    # Time features
    if PreprocessingFlags.TIME_FEATURES in namespace.preprocessing_flags:
        df = data.expand_time_features(df)

    # Passenger count
    if PASSENGER_COUNT_FLAGS & namespace.preprocessing_flags:
        passenger_outliers = df.select(
            passenger_outlier=pl.col('passenger_count') > 6).get_columns()[0]

        if (PreprocessingFlags.PASSENGER_COUNT_FEATURES
                in namespace.preprocessing_flags):
            df = df.with_columns(passenger_outliers)

        if (PreprocessingFlags.PASSENGER_COUNT_OUTLIERS
                in namespace.preprocessing_flags):
            df = df.filter(~passenger_outliers)

    # Compute ocean features if necessary, then apply them as requested
    if OCEAN_FLAGS & namespace.preprocessing_flags:
        x = df['pickup_longitude'].append(df['dropoff_longitude'])
        y = df['pickup_latitude'].append(df['dropoff_latitude'])
        points_area = data.get_square_area(x, y)

        ocean_pickup = df.select(
            pl.struct(['pickup_longitude', 'pickup_latitude',
                       'dropoff_longitude', 'dropoff_latitude'])
            .map(data.polars_point_on_ocean(points_area, pickup=True))
            ).get_columns()[0].alias('ocean_pickup')
        ocean_dropoff = df.select(
            pl.struct(['pickup_longitude', 'pickup_latitude',
                       'dropoff_longitude', 'dropoff_latitude'])
            .map(data.polars_point_on_ocean(points_area, dropoff=True))
            ).get_columns()[0].alias('ocean_dropoff')

        if PreprocessingFlags.OCEAN_FEATURES in namespace.preprocessing_flags:
            df = df.with_columns(ocean_pickup, ocean_dropoff)

        if PreprocessingFlags.OCEAN_OUTLIERS in namespace.preprocessing_flags:
            df = df.filter(~ocean_pickup & ~ocean_dropoff)

    # Compute borough features if necessary, then apply them as requested
    if BOROUGH_FLAGS & namespace.preprocessing_flags:
        x = df['pickup_longitude'].append(df['dropoff_longitude'])
        y = df['pickup_latitude'].append(df['dropoff_latitude'])
        points_area = data.get_square_area(x, y)

        boros = boroughs.load()
        boros_image, boros_colors = boroughs.get_image_boroughs(boros,
                                                                points_area)

        pickup_borough = (
            df.select(
                pickup_borough=pl.struct(['pickup_longitude',
                                          'pickup_latitude'])
                .map(boroughs.point_boroughs(boros_image, boros_colors,
                                             points_area, "pickup_"))
            ).get_columns()[0]
        )

        dropoff_borough = (
            df.select(
                dropoff_borough=pl.struct(['dropoff_longitude',
                                           'dropoff_latitude'])
                .map(boroughs.point_boroughs(boros_image, boros_colors,
                                             points_area, "dropoff_"))
            ).get_columns()[0]
        )

        if (PreprocessingFlags.BOROUGH_FEATURES
                in namespace.preprocessing_flags):
            df = df.with_columns(pickup_borough, dropoff_borough)

        if (PreprocessingFlags.BOROUGH_OUTLIERS
                in namespace.preprocessing_flags):
            df = df.filter((pickup_borough != 'None')
                           & (dropoff_borough != 'None'))

        # df = df.get_dummies(columns=('pickup_borough', 'dropoff_borough'))

    # Outlier detection through autoencoder
    if PreprocessingFlags.REE_OUTLIERS in namespace.preprocessing_flags:
        if namespace.ree_model is None:
            raise ValueError('In order to detect reconstruction error '
                             'outlier please specify a path to a trained model'
                             ' (--ree-model via CLI).')

        ae_model = tf.keras.models.load_model(namespace.ree_model)
        ree = ae.compute_ree(df, ae_model, 2048)
        df = df.filter(ree < namespace.ree_threshold)

    return df


def dump_preprocess(namespace: Namespace):
    """Process the given arguments and dump dataframe to file."""
    df = preprocess(namespace)
    df.write_parquet(namespace.output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('input_path', type=str, metavar='INPUT_FILE')
    parser.add_argument('output_path', type=str, metavar='OUTPUT_FILE')
    parser.add_argument('-p', '--preprocess', type=str,
                        metavar='OPERATION', dest='preprocessing_flags',
                        action=AddFlagEnumAction,
                        choices=PreprocessingFlags.__members__.keys())
    parser.add_argument('-n', '--num-samples', type=int, dest='samples')
    parser.add_argument('--for-autoencoder', dest='preprocessing_flags',
                        action='store_const', const=FOR_AUTOENCODER_FLAGS)
    parser.add_argument('--ree-model', dest='ree_model', type=str)
    parser.add_argument('--ree-threshold', dest='ree_threshold', type=float)
    parser.add_argument('--for-train', dest='preprocessing_flags',
                        const=FOR_TRAIN_FLAGS, action='store_const')
    parser.add_argument('--city-graph', dest='city_graph', type=str)
    parser.add_argument('--city-distance-matrix', dest='city_distance_matrix',
                        type=str)

    args = parser.parse_args(namespace=Namespace())
    dump_preprocess(args)
