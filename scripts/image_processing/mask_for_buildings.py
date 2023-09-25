import argparse
import os
import sys
import time
from glob import glob
from loguru import logger
from tqdm import tqdm
from yaml import load, FullLoader

import geopandas as gpd
import rasterio as rio
from rasterio.mask import mask
from shapely.geometry import mapping

sys.path.insert(1, 'scripts')
import functions.fct_misc as misc
import functions.fct_raster as raster

logger = misc.format_logger(logger)

tic = time.time()
logger.info('Starting...')

# Argument and parameter specification
parser = argparse.ArgumentParser(description="The script makes a mask for the building images.")
parser.add_argument('config_file', type=str, help='Framework configuration file')
args = parser.parse_args()

logger.info(f"Using {args.config_file} as config file.")
with open(args.config_file) as fp:
    cfg = load(fp, Loader=FullLoader)[os.path.basename(__file__)]

TRANSPARENCY = cfg['transparency']

WORKING_DIR = cfg['working_dir']
ROOFS = cfg['roofs']
IMAGE_FOLDER = cfg['image_folder']

os.chdir(WORKING_DIR)

if TRANSPARENCY:
    output_dir = misc.ensure_dir_exists(os.path.join(IMAGE_FOLDER, 'masked_images'))
else:
    output_dir = misc.ensure_dir_exists('processed/tiles/mask')

logger.info('Loading data...')
roofs = gpd.read_file(ROOFS)
tiles = glob(os.path.join(IMAGE_FOLDER, '*.tif'))

logger.info('Vector data processing...')
roofs = roofs.buffer(1)
merged_roofs_geoms = roofs.unary_union

for tile in tqdm(tiles, desc='Producing the masks...'):

    if TRANSPARENCY:
        geoms_list = [mapping(merged_roofs_geoms)]

        with rio.open(tile) as src:
            mask_image, mask_transform = mask(src, geoms_list)
            mask_meta=src.meta

        mask_meta.update({'transform': mask_transform, 'crs': rio.CRS.from_epsg(2056)})
        filepath=os.path.join(output_dir, os.path.basename(tile).split('.')[0] + '_masked.tif')
        
        with rio.open(filepath, 'w', **mask_meta) as dst:
            dst.write(mask_image)
    
    else:
        with rio.open(tile, "r") as src:
            tile_meta = src.meta

        mask_polygons, mask_meta = raster.polygons_to_raster_mask(merged_roofs_geoms, tile_meta)
        mask_meta.update({'crs': rio.CRS.from_epsg(2056)})

        filepath = os.path.join(output_dir, os.path.basename(tile).split('.')[0] + '_mask.tif')
        with rio.open(filepath, 'w', **mask_meta) as dst:
            dst.write(mask_polygons, 1)

logger.success(f'The masks were written in the folder {output_dir}.')