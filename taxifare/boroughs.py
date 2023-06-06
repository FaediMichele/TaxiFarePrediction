from typing import Tuple

import shapely as sh
import json
import polars as pl
import numpy as np
from tqdm import tqdm
from rasterio import features, transform

import taxifare.data as data

DEFAULT_DATA = 'datasets/sde-columbia-nycp_2007_nynh-geojson.json'


def load(file: str = DEFAULT_DATA) -> dict:
    with open(file) as f:
        raw_json = json.load(f)

    boros = {}
    for feature in raw_json['features']:
        hood_name = feature['properties']['nhoodname'].replace(" ", "_")
        boro_name = feature['properties']['boroname'].replace(" ", "_")
        hood = {
            'name': hood_name,
            'geometry': sh.unary_union(sh.from_geojson(
                        json.dumps(feature['geometry'])))
        }
        if boro_name in boros:
            boros[boro_name]['hoods'].append(hood)
        else:
            boros[boro_name] = {
                'name': boro_name,
                'hoods': [hood]
                }

    for b in boros.values():
        b['total_geometry'] = sh.union_all([h['geometry']
                                            for h in b['hoods']])
    return boros


def get_color(image, x, y, shape):
    try:
        return image[shape[1] - round(y), round(x)]
    except Exception:
        return 0


def point_boroughs(image, colors, points_area, prefix: str = ""):
    def return_function(coords: pl.Series):
        longitude = coords.struct.field(prefix + 'longitude')
        latitude = coords.struct.field(prefix + 'latitude')

        x_min, x_max, y_min, y_max = points_area

        x_norm = image.shape[0] * (longitude - x_min) / (x_max - x_min)
        y_norm = image.shape[1] * (latitude - y_min) / (y_max - y_min)

        return pl.Series([
            colors.get(get_color(image, x, y, image.shape), 'None')
            for x, y in tqdm(zip(x_norm, y_norm), total=len(longitude))
        ])

    return return_function


def get_neighborhood_names(boros: dict) -> int:
    names = []
    for b in boros.values():
        names.extend([h['name'] for h in b['hoods']])
    return names


def get_image_neighborhood(boros: dict, points_area,
                           out_shape=(2000, 2000), dtype=np.float32):
    west, east, south, north = points_area
    geometries = []

    for b in boros.values():
        geometries.extend([h['geometry'] for h in b['hoods']])

    names = get_neighborhood_names(boros)
    colors = {(k+1): name for k, name in enumerate(names)}
    colors[0] = 'None'

    out = np.ndarray(out_shape, dtype=dtype)

    features.rasterize(zip(geometries, colors), out=out, fill=0,
                       transform=transform.from_bounds(
                            west, south, east, north, *out.shape))

    return out, colors


def get_image_boroughs(boros: dict, points_area, out_shape=(2000, 2000),
                       dtype=np.float32):
    west, east, south, north = points_area

    geometries = [b['total_geometry'] for b in boros.values()]
    names = [b['name'] for b in boros.values()]
    colors = {(k+1): name for k, name in enumerate(names)}
    colors[0] = 'None'

    out = np.ndarray(out_shape, dtype=dtype)

    features.rasterize(zip(geometries, colors), out=out, fill=0,
                       transform=transform.from_bounds(
                            west, south, east, north, *out.shape))

    return out, colors


def compute_boroughs(df: pl.DataFrame) -> Tuple[pl.Series, pl.Series]:
    """Return pickup and dropoff boroughs for a given dataframe."""
    # The algorithm is calibrated for the default NYC area defined in
    # data. Use those point to normalize.
    in_area = df.select(in_area=data.in_newyork_area_expr()).get_columns()[0]
    x = df['pickup_longitude'].filter(in_area).append(
        df['dropoff_longitude'].filter(in_area))
    y = df['pickup_latitude'].filter(in_area).append(
        df['dropoff_latitude'].filter(in_area))
    points_area = data.get_square_area(x, y)

    boros = load()
    boros_image, boros_colors = get_image_boroughs(boros, points_area)

    pickup_borough = (
        df.select(
            pickup_borough=pl.struct(['pickup_longitude', 'pickup_latitude'])
            .map(point_boroughs(boros_image, boros_colors,
                                points_area, "pickup_"))
        ).get_columns()[0]
    )

    dropoff_borough = (
        df.select(
            dropoff_borough=pl.struct(['dropoff_longitude',
                                       'dropoff_latitude'])
            .map(point_boroughs(boros_image, boros_colors,
                                points_area, "dropoff_"))
        ).get_columns()[0]
    )

    return pickup_borough, dropoff_borough
