import math
from functools import lru_cache
from typing import Callable

import polars as pl
import staticmaps
import matplotlib.pyplot as plt
import PIL.Image

DATASET_PATH = 'datasets/train.csv'
NEW_YORK_AREA = [(40.506797, 41.130785), (-74.268086, -73.031593)]

NEW_YORK_MAP_IMAGE_SIZE = (2048, 2048)


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

def get_image_from_coordinate(points_area: tuple[float,float,float,float], sizes:tuple[float,float]) -> PIL.Image.Image:
    """Download the image from a Open Street Map of an area (left, right,down,up) coordinate with a specific size"""
    context = staticmaps.Context()
    context.set_tile_provider(staticmaps.tile_provider_OSM)
    left, right, down, up = points_area
    polygon = [
        (down, left),
        (down, right),
        (up, right),
        (up, left),
        (down, left)
    ]

    context.add_object(
        staticmaps.Area(
            [staticmaps.create_latlng(lat, lng) for lat, lng in polygon],
            fill_color=staticmaps.TRANSPARENT,
            width=2,
            color=staticmaps.BLACK,
        )
    )
    return context.render_pillow(*sizes)

def crop_image_with_borders(image:PIL.Image.Image) -> PIL.Image.Image:
    """Crop the image with a rectangle inside returning only the image inside the rectangle"""
    width, height = image.size
    top = 0
    left = 0
    right = width
    bottom = height

    for k in range(0,width):
        pixel = image.getpixel((k, height//2))
        if pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0:
            left = k
            break
    for k in reversed(range(0,width)):
        pixel = image.getpixel((k, height//2))
        if pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0:
            right = k
            break
    for k in range(0,height):
        pixel = image.getpixel((width//2, k))
        if pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0:
            top = k
            break
    for k in reversed(range(0,height)):
        pixel = image.getpixel((width//2, k))
        if pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0:
            bottom = k
            break

    return image.crop((left, top, right, bottom))


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


@lru_cache
def new_york_map(filename='map.png') -> PIL.Image.Image:
    """Retrieve a cached image of the New York map."""
    return PIL.Image.open(filename)


def point_on_ocean(x: float, y: float, image: PIL.Image.Image,
                   ocean_color=(170, 211, 223), color_sensitivity=5) -> bool:
    """Return whether a point is appears to be on the ocean."""
    try:
        pixel = image.getpixel((round(x), image.size[1] - round(y)))
    except IndexError:
        # print(f'{x}, {y} is out of range, image size '
        #      f'{image.width}, {image.height}')
        return False

    return math.dist(ocean_color, pixel[:3]) <= color_sensitivity


def polars_point_on_ocean(points_area, only_pickup=True, both=True):
    """Apply function for polars dataframes."""

    def return_function(coords: pl.Series):
        image = new_york_map()

        if only_pickup or both:
            pickup_x, pickup_y = normalize_points(coords.struct.field('pickup_longitude'),
                                                coords.struct.field('pickup_latitude'),
                                                points_area, image.size)

        if not only_pickup or both:
            dropoff_x, dropoff_y = normalize_points(coords.struct.field('dropoff_longitude'),
                                                    coords.struct.field('dropoff_latitude'),
                                                    points_area, image.size)

        
        if both:
            return pl.Series([
                point_on_ocean(x, y, image) or point_on_ocean(d_x, d_y, image)
                for x, y, d_x, d_y in zip(pickup_x, pickup_y, dropoff_x, dropoff_y)
            ])
        
        if only_pickup:
            return pl.Series([
                point_on_ocean(x, y, image)
                for x, y in zip(pickup_x, pickup_y)
            ])
        else:
            return pl.Series([
                point_on_ocean(d_x, d_y, image)
                for d_x, d_y in zip(dropoff_x, dropoff_y)
            ])
    return return_function
