import datetime
import math
from functools import lru_cache
from typing import Callable, Tuple, Sequence
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import PIL.Image
from sklearn.linear_model import LinearRegression
from scipy.spatial import KDTree
import taxifare.basemap as basemap
from tqdm import tqdm
import osmnx as ox
import networkx as nx


DATASET_PATH = 'datasets/train.csv'
NEW_YORK_AREA = [(40.506797, 41.130785), (-74.268086, -73.031593)]
TREND_DATETIME_GAP = pl.datetime(2012, 9, 1)

IMAGE_API_URL = 'https://b.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}.png'


def load_data(dataset_path=DATASET_PATH) -> pl.LazyFrame:
    return pl.scan_csv(dataset_path).filter(
        (pl.col("pickup_longitude").is_between(*NEW_YORK_AREA[1], closed='both')) &
        (pl.col("pickup_latitude").is_between(*NEW_YORK_AREA[0], closed='both')) &
        (pl.col("dropoff_longitude").is_between(*NEW_YORK_AREA[1], closed='both')) &
        (pl.col("dropoff_latitude").is_between(*NEW_YORK_AREA[0], closed='both')) &
        (pl.col("passenger_count") > 0)).with_columns([
            pl.col("pickup_datetime").str.strptime(pl.Datetime, fmt="%Y-%m-%d %H:%M:%S UTC", strict=True)]
        ).drop('key')


def normalize_points(x: pl.Series, y: pl.Series, points_area: tuple[float, float, float, float],
    image_size: tuple[float,float]) -> tuple[pl.Series, pl.Series]:
    """Return copies of x and y, normalized based on image size.

    Can be used to plot points on image.
    """
    x_min, x_max, y_min, y_max = points_area

    x_norm = image_size[0] * (x - x_min) / (x_max - x_min)
    y_norm = image_size[1] * (y - y_min) / (y_max - y_min)

    return x_norm, y_norm


def plot_distributions(dataframe: pl.DataFrame, num_cols=4):
    """Plot the distribution for each feature in a multiplot."""
    num_rows = math.ceil(len(dataframe.columns) / num_cols)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(16, 16))
    for i, column in enumerate(dataframe.columns):
        ax = axs[i // num_cols, i % num_cols]
        ax.hist(dataframe[column])
        ax.title.set_text(column)

def regula_falsi(f: Callable[[float], float], a:float, b:float, tol:float) -> tuple[float,float]:
    """Compute the zero of a function with the given tolerance and two initial points

    return the zero position and it's approximated error.
    """
    nMax = math.ceil(math.log(abs(b-a)/tol)/math.log(2))
    fa = f(a)
    fb = f(b)
    x = a
    fx = fa
    n = 0

    while abs(a-b) >= tol+1e-16*max(abs(a),abs(b)) and abs(fx) >= tol and n < nMax:
        n += 1
        x = a-fa*(b-a) / (fb-fa)
        fx = f(x)
        if (fx >= 0 and fa >= 0) or (fa <0 and fa < 0):
            a = x
            fa = fx
        else:
            b = x
            fb = fx

    return x, fx

def find_latitude_correction(p: tuple[float, float], additional_space: float, b:float, tol=1e-4)-> tuple[float, float]:
    """Calculate the new latitude above or below(sign of b) additional_space(in km)

    return the new latitude and the approximated error.
    """
    f = lambda x: distance(p, (p[0], x)) - additional_space
    a = p[1]
    b = a + b
    return regula_falsi(f, a, b, tol)



def distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Calculate distance in km between two point on earth"""
    lon1, lat1 = math.radians(p1[0]), math.radians(p1[1])
    lon2, lat2 = math.radians(p2[0]), math.radians(p2[1])

    a = math.sin((lat1-lat2)/2)**2 + math.cos(lat1)*math.cos(lat2)*(math.sin((lon1-lon2)/2)**2)
    return math.atan2(math.sqrt(a), math.sqrt(1-a))*2*6371


def get_square_area(x: pl.Series, y: pl.Series
                    ) -> Tuple[float, float, float, float]:
    """Given a sequence of points the min square area containing them."""
    points_area = x.min(), x.max(), y.min(), y.max()

    # Make the area a square
    width = distance((points_area[0], points_area[2]),
                     (points_area[1], points_area[2]))
    height = distance((points_area[0], points_area[2]),
                      (points_area[0], points_area[3]))

    additional_space = (width - height) / 2

    new_lat_min, _ = find_latitude_correction((points_area[0], points_area[2]),
                                              additional_space, b=-1)
    new_lat_max, _ = find_latitude_correction((points_area[0], points_area[3]),
                                              additional_space, b=1)

    return points_area[0], points_area[1], new_lat_min, new_lat_max


@lru_cache
def new_york_map(points_area) -> PIL.Image.Image:
    """Retrieve a cached image of the New York map."""
    left, right, bottom, top = points_area
    return basemap.image(top, right, bottom, left, zoom=10, url=IMAGE_API_URL)


def point_on_ocean(x: float, y: float, image: PIL.Image.Image,
                    ocean_color=(212,218,220), color_sensitivity=5) -> bool:
    """Return whether a point is appears to be on the ocean."""
    try:
        pixel = image.getpixel((round(x), image.size[1] - round(y)))
    except IndexError:
        return False

    return math.dist(ocean_color, pixel[:3]) <= color_sensitivity


def polars_point_on_ocean(points_area, pickup=False, dropoff=False):
    """Apply function for polars dataframes."""

    if not pickup and not dropoff:
        raise Exception("no field passed")

    def return_function(coords: pl.Series):
        image = new_york_map(points_area)

        if pickup:
            pickup_x, pickup_y = normalize_points(coords.struct.field('pickup_longitude'),
                                                coords.struct.field('pickup_latitude'),
                                                points_area, image.size)

        if dropoff:
            dropoff_x, dropoff_y = normalize_points(coords.struct.field('dropoff_longitude'),
                                                    coords.struct.field('dropoff_latitude'),
                                                    points_area, image.size)

        if pickup and dropoff:
            return pl.Series([
                point_on_ocean(x, y, image) or point_on_ocean(d_x, d_y, image)
                for x, y, d_x, d_y in tqdm(zip(pickup_x, pickup_y, dropoff_x, dropoff_y), total=len(pickup_x))
            ])

        if pickup:
            return pl.Series([
                point_on_ocean(x, y, image)
                for x, y in tqdm(zip(pickup_x, pickup_y), total=len(pickup_x))
            ])
        else:
            return pl.Series([
                point_on_ocean(d_x, d_y, image)
                for d_x, d_y in tqdm(zip(dropoff_x, dropoff_y), total=len(dropoff_x))
            ])
    return return_function


def months_from(from_: datetime.datetime, to: datetime.datetime) -> int:
    """Return the total number of months between the two dates."""
    return (to.year - from_.year) * 12 + to.month


def detrend(data: pl.DataFrame,
            estimator1: LinearRegression, estimator2: LinearRegression,
            threshold: pl.Datetime,
            date_column='pickup_datetime',
            trended_column='fare_amount') -> pl.DataFrame:
    """Detrend dataframe.

    Two estimators are used based on a treshold date. The output
    shape is: ``(rows, 2)``, where ``rows`` is the number of input rows
    and ``2`` is given by the date column and the detrended column.

    The estimators are used on predictions month-wise (i.e. the first
    month in the series is the index 0, and so on).
    """
    min_date = data[date_column].min()
    max_date = data[date_column].max()
    threshold = pl.select(threshold)[0, 0]      # Cast to datetime.datetime

    features = np.array(range(months_from(min_date, max_date))).reshape(-1, 1)
    split = months_from(min_date, threshold) - 1

    predictions = np.concatenate((estimator1.predict(features[:split]),
                                  estimator2.predict(features[split:])))

    return data.select(
        [date_column,
         pl.col(trended_column)
         - pl.col(date_column).apply(lambda d: predictions[months_from(min_date, d) - 1])])


def expand_time_features(data: pl.DataFrame,
                         gap_threshold: pl.Datetime = TREND_DATETIME_GAP,
                         date_column='pickup_datetime') -> pl.DataFrame:
    """Return a new dataframe with properlu engineered time features.

    In particular, remove general datetime column in favor of: year,
    month, weekday, hour, 2012 gap boolean, weekend boolean.
    Year, month, weekday and hour are simple ordinal encodings.
    """
    if date_column not in data.columns:
        print(f'No column named {date_column}, skipping')
        return data

    return (data.with_columns(year=pl.col(date_column).dt.year(),
                              month=pl.col(date_column).dt.month(),
                              weekday=pl.col(date_column).dt.weekday(),
                              hour=pl.col(date_column).dt.hour(),
                              after2012=pl.col(date_column) >= gap_threshold,
                              weekend=pl.col(date_column).dt.weekday() > 5)
            .drop(date_column))


def split(df: pl.DataFrame, valid, test, seed=1,
          shuffle=True) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """"""
    if shuffle:
        df = df.select(pl.all().shuffle(seed))

    valid_size = int(len(df) * valid)
    test_size = int(len(df) * test)

    train_df = df.slice(valid_size + test_size)
    valid_df = df.slice(0, valid_size)
    test_df = df.slice(valid_size, test_size)

    return train_df, valid_df, test_df


def standardize(mean: pl.DataFrame, std: pl.DataFrame):
    """Standardize (zero mean, unit variance) polars columns.

    Usage::

        df = df.with_columns(**standardize(mean, std))

    ``mean`` and ``std`` with the following requisites:

    - Only one row
    - The columns must be a subset of the ones of ``df`` (i.e. the
        dataframe on which the expression in being applied)

    The values of the ``mean`` dataframe will be subtracted to the
    corresponding columns in the dataframe, while the values of the
    ``std`` will be used as divisor.
    """
    assert mean.columns == std.columns
    return {column: (pl.col(column) - mean[column]) / std[column]
            for column in mean.columns}


def normalize(min_: pl.DataFrame, max_: pl.DataFrame):
    """Normalize polars columns between ``min_`` and ``max_``.

    Same as :func:`standardize` but for minmax scaling.
    """
    assert min_.columns == max_.columns
    return {column: (pl.col(column) - min_[column])
                    / (max_[column] - min_[column])             # NOQA
            for column in min_.columns}

def calculate_travel_distance(data: tuple[float, float, float, float],
                              G: nx.MultiDiGraph) -> list[float]:
    lon1, lat1, lon2, lat2 = data
    start_node = ox.nearest_nodes(G, lon1, lat1)
    end_node = ox.nearest_nodes(G, lon2, lat2)
    path = nx.shortest_path(G, start_node, end_node, weight='travel_time')
    return nx.path_weight(G, path, weight='travel_time')


def calculate_travel_distance_astar(data: tuple[float, float, float, float],
                                    G: nx.MultiDiGraph) -> list[float]:
    lon1, lat1, lon2, lat2 = data
    def dist(a, b):
        (x1, y1) = G.nodes[a]['x'], G.nodes[a]['y']
        (x2, y2) = G.nodes[b]['x'], G.nodes[b]['y']
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    start_node = ox.nearest_nodes(G, lon1, lat1)
    end_node = ox.nearest_nodes(G, lon2, lat2)
    path = nx.astar_path(G, start_node, end_node, heuristic=dist, weight='travel_time')
    return nx.path_weight(G, path, weight='travel_time')

def calculate_travel_distance_with_matrix(data: tuple[float, float, float, float],
                                          G: nx.MultiDiGraph,
                                          distance_matrix: np.ndarray,
                                          kdtree: KDTree=None) -> list[float]:
    lon1, lat1, lon2, lat2 = data
    if kdtree is None:
        start_node = ox.nearest_nodes(G, lon1, lat1)
        end_node = ox.nearest_nodes(G, lon2, lat2)
    else:
        start_node, end_node = kdtree.query([[lon1, lat1], [lon2, lat2]])
    return distance_matrix[start_node, end_node]