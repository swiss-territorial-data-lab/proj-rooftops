import os, sys
import time
from glob import glob
from loguru import logger
from tqdm import tqdm
from yaml import load, FullLoader

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.features import rasterize
from shapely.geometry import Polygon, mapping

sys.path.insert(1, 'scripts')
import functions.fct_misc as fct_misc

logger=fct_misc.format_logger(logger)

# Define functions ------------------------------

def poly_from_utm(polygon, transform):
    poly_pts = []
    
    for i in np.array(polygon.exterior.coords):
        
        # Convert polygons to the image CRS
        poly_pts.append(~transform * tuple(i))
        
    # Generate a polygon object
    new_poly = Polygon(poly_pts)
    return new_poly


tic = time.time()
logger.info('Starting...')

# Define contants ------------------------------

logger.info(f"Using config.yaml as config file.")
with open('scripts/image_processing/config.yaml') as fp:
        cfg = load(fp, Loader=FullLoader)['mask_for_buildings.py']

logger.info('Defining constants...')

TRANSPARENCY=cfg['transparency']

WORKING_DIR=cfg['working_dir']
SHP_ROOFS=cfg['roofs_dir']
LAYER=cfg['layer']
IMAGE_FOLDER=cfg['image_dir']

os.chdir(WORKING_DIR)

logger.info('Importing data...')
roofs=gpd.read_file(SHP_ROOFS)
tiles=glob(os.path.join(IMAGE_FOLDER, '*.tif'))
if '\\' in tiles[0]:
     tiles=[tile.replace('\\', '/') for tile in tiles]

logger.info('Processing vector data...')
roofs=roofs.buffer(0)
merged_roofs_geoms=roofs.unary_union

for tile in tqdm(tiles, desc='Producing the masks', total=len(tiles)):

    if TRANSPARENCY:
        geoms_list = [mapping(merged_roofs_geoms)]

        with rasterio.open(tile) as src:
            mask_image, mask_transform = mask(src, geoms_list)
            mask_meta=src.meta

        mask_meta.update({'transform': mask_transform})
        filepath=os.path.join(fct_misc.ensure_dir_exists(os.path.join(IMAGE_FOLDER, 'masked_images')),
                            tile.split('/')[-1].split('.')[0] + '_masked.tif')
        
        with rasterio.open(filepath, 'w', **mask_meta) as dst:
            dst.write(mask_image)

    else:
        with rasterio.open(tile, "r") as src:
            tile_img = src.read()
            tile_meta = src.meta

        im_size = (tile_meta['height'], tile_meta['width'])

        polygons=[poly_from_utm(geom, src.meta['transform']) for geom in merged_roofs_geoms.geoms]
        mask_image = rasterize(shapes=polygons, out_shape=im_size)

        mask_meta = src.meta.copy()
        mask_meta.update({'count': 1, 'dtype': 'uint8', 'nodata':99})

        filepath=os.path.join(fct_misc.ensure_dir_exists(os.path.join(IMAGE_FOLDER, 'mask')),
                                tile.split('/')[-1].split('.')[0] + '_mask.tif')    

        with rasterio.open(filepath, 'w', **mask_meta) as dst:
            dst.write(mask_image, 1)

logger.success(f'The masks were written in the folder {os.path.join(IMAGE_FOLDER, "mask")}.')