import math

import polars as pl
import staticmaps
import matplotlib.pyplot as plt
import math

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

def crop_image_with_borders(image):
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


def plot_distributions(dataframe: pl.DataFrame, num_cols=4):
    """Plot the distribution for each feature in a multiplot."""
    num_rows = math.ceil(len(dataframe.columns) / num_cols)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(16, 16))
    for i, column in enumerate(dataframe.columns):
        ax = axs[i // num_cols, i % num_cols]
        ax.hist(dataframe[column])
        ax.title.set_text(column)

def regula_falsi(f, a, b, tol):
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

def find_latitude_correction(p, additional_space, b, tol=1e-4):
    f = lambda x: distance(p, (p[0], x)) - additional_space
    a = p[1]
    b = a + b
    return regula_falsi(f, a, b, tol)



def distance(p1, p2):
    lon1, lat1 = math.radians(p1[0]), math.radians(p1[1])
    lon2, lat2 = math.radians(p2[0]), math.radians(p2[1])
    
    a = math.sin((lat1-lat2)/2)**2 + math.cos(lat1)*math.cos(lat2)*(math.sin((lon1-lon2)/2)**2)
    return math.atan2(math.sqrt(a), math.sqrt(1-a))*2*6371
