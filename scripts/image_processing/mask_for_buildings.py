import os
import sys
import time
from glob import glob
from loguru import logger
from tqdm import tqdm
from yaml import load, FullLoader

import geopandas as gpd
import rasterio as rio

sys.path.insert(1, 'scripts')
import functions.fct_misc as misc
import functions.fct_raster as raster

logger = misc.format_logger(logger)

tic = time.time()
logger.info('Starting...')

logger.info(f"Using config.yaml as config file.")

with open('config/config_lidar_products.yaml') as fp:
        cfg = load(fp, Loader = FullLoader)[os.path.basename(__file__)]

WORKING_DIR = cfg['working_dir']
ROOFS = cfg['roofs']
IMAGE_FOLDER = cfg['image_folder']

os.chdir(WORKING_DIR)

output_dir = misc.ensure_dir_exists('processed/tiles/mask')

logger.info('Loading data...')
roofs = gpd.read_file(ROOFS)
tiles = glob(os.path.join(IMAGE_FOLDER, '*.tif'))

logger.info('Vector data processing...')
roofs = roofs.buffer(1)
merged_roofs_geoms = roofs.unary_union

for tile in tqdm(tiles, desc='Producing the masks...'):

    with rio.open(tile, "r") as src:
        tile_meta = src.meta

    mask, mask_meta = raster.polygons_to_raster_mask(merged_roofs_geoms, tile_meta)
    mask_meta.update({'crs': rio.CRS.from_epsg(2056)})

    filepath = os.path.join(output_dir, os.path.basename(tile).split('.')[0] + '_mask.tif')
    with rio.open(filepath, 'w', **mask_meta) as dst:
        dst.write(mask, 1)

logger.success(f'The masks were written in the folder {output_dir}.')