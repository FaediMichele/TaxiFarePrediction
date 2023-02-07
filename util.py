import math

import polars as pl
import staticmaps
import matplotlib.pyplot as plt

DATASET_PATH = 'datasets/train.csv'
NEW_YORK_AREA = [(40.506797, 41.130785), (-74.268086, -73.031593)]




def load_data(dataset_path=DATASET_PATH) -> pl.LazyFrame:
    return pl.scan_csv(dataset_path).filter(
        (pl.col("pickup_longitude").is_between(*NEW_YORK_AREA[1], closed='both')) &
        (pl.col("pickup_latitude").is_between(*NEW_YORK_AREA[0], closed='both')) &
        (pl.col("dropoff_longitude").is_between(*NEW_YORK_AREA[1], closed='both')) &
        (pl.col("dropoff_latitude").is_between(*NEW_YORK_AREA[0], closed='both')) &
        (pl.col("fare_amount") > 0) &
        (pl.col("passenger_count") > 0)).with_columns([
            pl.col("pickup_datetime").str.strptime(pl.Datetime, fmt="%Y-%m-%d %H:%M:%S UTC", strict=True)]
        ).drop('key')

def get_image_from_coordinate(points_area, sizes):
    context = staticmaps.Context()
    context.set_tile_provider(staticmaps.tile_provider_OSM)
    left, right, down, up = points_area
    polygon = [
        (down, left),
        (down, right),
        (up, right),
        (up, left)
    ]

    context.add_object(
        staticmaps.Area(
            [staticmaps.create_latlng(lat, lng) for lat, lng in polygon],
            fill_color=staticmaps.TRANSPARENT,
            width=0,
            color=staticmaps.TRANSPARENT,
        )
    )
    return context.render_pillow(*sizes)


def plot_distributions(dataframe: pl.DataFrame, num_cols=4):
    """Plot the distribution for each feature in a multiplot."""
    num_rows = math.ceil(len(dataframe.columns) / num_cols)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(16, 16))
    for i, column in enumerate(dataframe.columns):
        ax = axs[i // num_cols, i % num_cols]
        ax.hist(dataframe[column])
        ax.title.set_text(column)
