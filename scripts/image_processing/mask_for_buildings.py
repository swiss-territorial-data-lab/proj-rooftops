import os, sys
import time
from glob import glob
from loguru import logger
from tqdm import tqdm
from yaml import load, FullLoader

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import Polygon

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

WORKING_DIR=cfg['working_dir']
ROOFS=cfg['roofs']
LAYER=cfg['layer']
IMAGE_FOLDER=cfg['image_folder']

os.chdir(WORKING_DIR)

logger.info('Importing data...')
roofs=gpd.read_file(ROOFS)
tiles=glob(os.path.join(IMAGE_FOLDER, '*.tif'))

logger.info('Treating vector data...')
roofs=roofs.buffer(1)
merged_roofs_geoms=roofs.unary_union
merged_roofs=gpd.GeoDataFrame({'id': [i for i in range(len(merged_roofs_geoms.geoms))],
                               'geometry': [geom for geom in merged_roofs_geoms.geoms]})

for tile in tqdm(tiles, desc='Producing the masks...'):

    with rasterio.open(tile, "r") as src:
        tile_img = src.read()
        tile_meta = src.meta

    im_size = (tile_meta['height'], tile_meta['width'])

    polygons=[poly_from_utm(geom, src.meta['transform']) for geom in merged_roofs_geoms.geoms]
    mask = rasterize(shapes=polygons, out_shape=im_size)

    mask_meta = src.meta.copy()
    mask_meta.update({'count': 1, 'dtype': 'uint8'})

    filepath=os.path.join(fct_misc.ensure_dir_exists('processed/tiles/mask'),
                            tile.split('/')[-1].split('.')[0] + '_mask.tif')
    with rasterio.open(filepath, 'w', **mask_meta) as dst:
        dst.write(mask,1)

logger.success(f'The masks were written in the folder processed/tiles/mask.')