import shapely as sh
import json
import polars as pl
import numpy as np
from tqdm import tqdm
from rasterio import features, transform

def load(file: str = 'datasets/sde-columbia-nycp_2007_nynh-geojson.json') -> dict:
    with open(file) as f:
        raw_json = json.load(f)

    boros = {}
    for feature in raw_json['features']:

        hood = {'name': feature['properties']['nhoodname'], 'geometry': sh.unary_union(sh.from_geojson(json.dumps(feature['geometry'])))}
        if feature['properties']['boroname'] in boros:
            boros[feature['properties']['boroname']]['hoods'].append(hood)
        else:
            boros[feature['properties']['boroname']] = {
                'name': feature['properties']['boroname'],
                'hoods' : [hood]
                }

    for b in boros.values():
        b['total_geometry'] = sh.union_all([h['geometry'] for h in b['hoods']])
    return boros


def get_color(image, x, y, shape):
    try:
        return image[shape[1] - round(y), round(x)]
    except Exception as e:
        print(str(e))
        return 0

def point_boroughs(image, colors, points_area, prefix: str = ""):
    def return_function(coords: pl.Series):
        longitude = coords.struct.field(prefix + 'longitude')
        latitude = coords.struct.field(prefix + 'latitude')

        x_min, x_max, y_min, y_max = points_area

        x_norm = image.shape[0] * (longitude - x_min) / (x_max - x_min)
        y_norm = image.shape[1] * (latitude - y_min) / (y_max - y_min)
        return pl.Series([
            colors[get_color(image, x, y, image.shape)] for x, y in tqdm(zip(x_norm, y_norm), total=len(longitude))
        ])

    return return_function

def get_neighborhood_names(boros: dict) -> int:
    l = []
    for b in boros.values():
        l.extend([h['name'] for h in b['hoods']])
    return l


def get_image_neighborhood(boros: dict, points_area, out_shape=(2000,2000), dtype=np.float32):
    west, east, south, north = points_area
    geometries = []
    
    for b in boros.values():
        geometries.extend([h['geometry'] for h in b['hoods']])
        
    names = get_neighborhood_names(boros)
    colors = {(k+1): name for k, name in enumerate(names)}
    colors[0] = 'None'

    out = np.ndarray(out_shape, dtype=dtype)
    
    features.rasterize(zip(geometries, colors), out=out, fill=0, transform=transform.from_bounds(west, south, east, north, *out.shape))

    return out, colors


def get_image_boroughs(boros: dict, points_area, out_shape=(2000,2000), dtype=np.float32):
    west, east, south, north = points_area

    geometries = [b['total_geometry'] for b in boros.values()]
    names = [b['name'] for b in boros.values()]
    colors = {(k+1): name for k, name in enumerate(names)}
    colors[0] = 'None'

    out = np.ndarray(out_shape, dtype=dtype)

    features.rasterize(zip(geometries, colors), out=out, fill=0, transform=transform.from_bounds(west, south, east, north, *out.shape))

    return out, colors