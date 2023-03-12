"""Preprocess the given dataset and dump it to a parquet file.

Entrypoint, run with ``python -m texifare.preprocess [...]``
"""
import argparse
import enum
from typing import Optional

import polars as pl

import taxifare.data as data
import taxifare.boroughs as boroughs

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


OCEAN_FLAGS = (PreprocessingFlags.OCEAN_FEATURES
               | PreprocessingFlags.OCEAN_OUTLIERS)

BOROUGH_FLAGS = (PreprocessingFlags.BOROUGH_FEATURES
                 | PreprocessingFlags.BOROUGH_OUTLIERS)


class Namespace:
    """Custom namespace for CLI parameters."""
    input_path: str
    output_path: str
    preprocessing_flags: PreprocessingFlags = PreprocessingFlags.BASE
    samples: Optional[int] = None


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

    # Time features
    if PreprocessingFlags.TIME_FEATURES in namespace.preprocessing_flags:
        df = data.expand_time_features(df)

    # Passenger count
    if (PreprocessingFlags.PASSENGER_COUNT_OUTLIERS
            in namespace.preprocessing_flags):
        df = df.filter(pl.col('passenger_count') <= 6)

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
            breakpoint()
            df = df.with_columns(pickup_borough, dropoff_borough)

        if (PreprocessingFlags.BOROUGH_OUTLIERS
                in namespace.preprocessing_flags):
            df = df.filter((pickup_borough != 'None')
                           & (dropoff_borough != 'None'))

        # df = df.get_dummies(columns=('pickup_borough', 'dropoff_borough'))

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

    args = parser.parse_args(namespace=Namespace())
    dump_preprocess(args)
