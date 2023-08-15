import os
import sys
import time
from glob import glob
from loguru import logger
from tqdm import tqdm
from yaml import load, FullLoader

import geopandas as gpd
import rasterio
from rasterio.features import rasterize

sys.path.insert(1, 'scripts')
import functions.fct_misc as misc

logger = misc.format_logger(logger)

tic = time.time()
logger.info('Starting...')

logger.info(f"Using config.yaml as config file.")

with open('config/config.yaml') as fp:
        cfg = load(fp, Loader = FullLoader)[os.path.basename(__file__)]

WORKING_DIR = cfg['working_dir']
ROOFS = cfg['roofs']
IMAGE_FOLDER = cfg['image_folder']

os.chdir(WORKING_DIR)

output_dir = 'processed/tiles/mask'
misc.ensure_dir_exists(output_dir)

logger.info('Loading data...')
roofs = gpd.read_file(ROOFS)
tiles = glob(os.path.join(IMAGE_FOLDER, '*.tif'))

logger.info('Vector data processing...')
roofs = roofs.buffer(1)
merged_roofs_geoms = roofs.unary_union
merged_roofs = gpd.GeoDataFrame({'id': [i for i in range(len(merged_roofs_geoms.geoms))],
                               'geometry': [geom for geom in merged_roofs_geoms.geoms]})

for tile in tqdm(tiles, desc='Producing the masks...'):

    with rasterio.open(tile, "r") as src:
        tile_img = src.read()
        tile_meta = src.meta

    im_size = (tile_meta['height'], tile_meta['width'])

    polygons = [misc.poly_from_utm(geom, src.meta['transform']) for geom in merged_roofs_geoms.geoms]
    mask = rasterize(shapes=polygons, out_shape=im_size)

    mask_meta = src.meta.copy()
    mask_meta.update({'count': 1, 'dtype': 'uint8'})

    filepath = os.path.join(output_dir,
                            tile.split('/')[-1].split('.')[0] + '_mask.tif')
    with rasterio.open(filepath, 'w', **mask_meta) as dst:
        dst.write(mask, 1)

logger.success(f'The masks were written in the folder {output_dir}.')