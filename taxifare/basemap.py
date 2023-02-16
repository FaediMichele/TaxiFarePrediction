from itertools import product
from io import BytesIO
from PIL import Image
import requests
import math

URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
RADIUS = 6371   # earth's average radius in km
TILE_SIZE = 256 # each tile's size in pixels


def point_to_pixels(lon, lat, zoom):
    """convert gps coordinates to web mercator"""
    r = math.pow(2, zoom) * TILE_SIZE
    lat = math.radians(lat)

    x = int((lon + 180.0) / 360.0 * r)
    y = int((1.0 - math.log(math.tan(lat) + (1.0 / math.cos(lat))) /
             math.pi) / 2.0 * r)

    return x, y


def tile(session=None, **url_format_args):
    """download tile as PIL.Image from Tile Server API"""
    url = url_format_args.get('url', URL)
    if not session:
        session = requests

    with session.get(url.format(**url_format_args), headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.84 Safari/537.36'}) as resp:
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content))


def image(top, right, bottom, left, zoom=14, **tile_args):
    """return an osm map from the given bounding box points"""

    # convert gps coordinates to web mercator
    x0, y0 = point_to_pixels(left, top, zoom)
    x1, y1 = point_to_pixels(right, bottom, zoom)

    # conver pixel coordinates to tiles
    x0_tile, y0_tile = int(x0 / TILE_SIZE), int(y0 / TILE_SIZE)
    x1_tile, y1_tile = math.ceil(x1 / TILE_SIZE), math.ceil(y1 / TILE_SIZE)

    # crate a single large image to add the tiles to
    img = Image.new('RGB', (
        int(x1_tile - x0_tile) * TILE_SIZE,
        int(y1_tile - y0_tile) * TILE_SIZE))

    # download all tiles
    with requests.Session() as session:
        for x_tile, y_tile in product(
                range(x0_tile, x1_tile), range(y0_tile, y1_tile)):

            img.paste(
                im=tile(session, x=x_tile, y=y_tile, z=zoom, **tile_args),
                box=(int(x_tile - x0_tile) * TILE_SIZE,
                     int(y_tile - y0_tile) * TILE_SIZE))

    # crop image to the given bounding box
    x, y = x0_tile * TILE_SIZE, y0_tile * TILE_SIZE
    return img.crop((
        int(x0 - x),  # left
        int(y0 - y),  # top
        int(x1 - x),  # right
        int(y1 - y))) # bottom