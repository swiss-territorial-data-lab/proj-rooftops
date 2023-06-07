#!/bin/python
# -*- coding: utf-8 -*-

#  Proj rooftops
#
#      Clemence Herny 
#      Gwenaelle Salamin
#      Alessandro Cerioni 


import os, sys
import time
import argparse
import yaml
from loguru import logger
from glob import glob
from tqdm import tqdm

import pandas as pd
import geopandas as gpd
import whitebox
wbt = whitebox.WhiteboxTools()
import rasterio as rio
import rasterio.mask
from rasterio.merge import merge

sys.path.insert(1, 'scripts')
import functions.fct_misc as fct_misc
import functions.common as c

# the following allows us to import modules from within this file's parent folder
sys.path.insert(0, '.')

logger.remove()
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}", level="INFO")


if __name__ == "__main__":

    # Start chronometer
    tic = time.time()
    logger.info('Starting...')

    # Argument and parameter specification
    parser = argparse.ArgumentParser(description="The script prepares dataset to process the rooftops project (STDL.proj-rooftops)")
    parser.add_argument('config_file', type=str, help='Framework configuration file')
    args = parser.parse_args()

    logger.info(f"Using {args.config_file} as config file.")
 
    with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)[os.path.basename(__file__)]

    # Load input parameters
    WORKING_DIR = cfg['working_dir']
    IMAGE_DIR = cfg['image_dir']
    TILES_DIR = cfg['tiles_dir']
    SHP_ROOFS = cfg['roofs_dir']
    BUFFER = cfg['buffer']
    OUTPUT_DIR = cfg['output_dir']

    os.chdir(WORKING_DIR)

    # Create an output directory in case it doesn't exist
    fct_misc.ensure_dir_exists(OUTPUT_DIR)

    written_files = []

    # Get the rooftops shapes
    SHP_ROOFS, ROOFS_NAME = os.path.split(SHP_ROOFS)
    feature_path = os.path.join(SHP_ROOFS, ROOFS_NAME[:-4]  + "_EGID.shp")

    if os.path.exists(feature_path):
        logger.info(f"File {ROOFS_NAME[:-4]}_EGID.shp already exists")
        rooftops = gpd.read_file(feature_path)
    else:
        logger.info(f"File {ROOFS_NAME[:-4]}_EGID.shp does not exist")
        logger.info(f"Create it")
        gdf_roofs = gpd.read_file(WORKING_DIR  + '/' + SHP_ROOFS  + '/' + ROOFS_NAME)
        logger.info(f"Dissolved shapes by EGID number")
        rooftops = gdf_roofs.dissolve('EGID', as_index=False)
        rooftops.drop(['OBJECTID', 'ALTI_MAX', 'DATE_LEVE', 'SHAPE_AREA', 'SHAPE_LEN'], axis=1)
        rooftops.to_file(feature_path)
        written_files.append(feature_path)  
        logger.info(f"...done. A file was written: {feature_path}")

    # AOI 
    logger.info("Select rooftop's shapes in the AOI")
    tiles = gpd.read_file(TILES_DIR)
    AOI = tiles.dissolve()
    rooftops_list = gpd.clip(rooftops, AOI)

    # Find the image's tiles intersecting the rooftops' shapes  
    logger.info("Intersection of rooftops' shape and images' tile")
    join = gpd.sjoin(rooftops_list, tiles, how="left")
    
    image_list = []  
    egid_list = []
    coords_list = [] 

    # Open or produce the rooftops boundary boxes shapes 
    feature_path = os.path.join(OUTPUT_DIR + 'bbox.gpkg')

    if os.path.exists(feature_path):
        logger.info(f"File bbox.gpkg already exists")
        bbox_list = gpd.read_file(feature_path)
    else:
        logger.info(f"File bbox.gpkg does not exist")
        logger.info(f"Create it")

        for row in join.itertuples():
            egid = row.EGID
            egid_list.append(egid)
            bounds = row.geometry.bounds
            coords = c.bbox(bounds)
            coords_list.append(coords)

        bbox_list = gpd.GeoDataFrame(pd.DataFrame(egid_list, columns=['EGID']), crs = 'epsg:2056', geometry = coords_list).drop_duplicates(subset='EGID')
        bbox_list.to_file(feature_path)
        written_files.append(feature_path)  
        logger.info(f"...done. A file was written: {feature_path}")

    # Get the image tile(s) number intersecting the rooftop shape 
    logger.info("Find the image tile(s) number(s) intersecting the rooftop shape")
    unique_egid = join["EGID"].unique() 

    for i in tqdm(unique_egid[:10], desc='EGID in AOI', total=len(unique_egid)):
        tiles_list = join[join['EGID'] == i]
        image_list = ((tiles_list['TileName'].to_numpy()).tolist())

        tiles_list = [] 
        for image in image_list:
            tiles_list.append(os.path.join(IMAGE_DIR, image + '.tif'))            

            raster_to_mosaic = []   
            for tile in tqdm(tiles_list, desc='Applying SAM to tiles', total=len(tiles_list)):
                raster = rio.open(tile)
                
                # Mosaic images if rooftops shape is spread over several tiles 
                if len(tiles_list) > 1:  
                    logger.info("Mosaic production")
                    raster_to_mosaic.append(raster)
                    mosaic, output = merge(raster_to_mosaic)

                    output_meta = raster.meta.copy()
                    output_meta.update(
                    {"driver": "GTiff",
                        "height": mosaic.shape[1],
                        "width": mosaic.shape[2],
                        "transform": output,
                    })

                    mosaic_path = os.path.join(OUTPUT_DIR, 'mosaic.tiff')
                    with rio.open(mosaic_path, "w", **output_meta) as m:
                        m.write(mosaic)
                    raster = rio.open(mosaic_path)

                image = raster

        logger.info("Rooftops' boundary box shapes")
        bbox_shape = bbox_list[bbox_list['EGID'] == i]['geometry'].buffer(BUFFER, join_style=2)  

        # Clip image by the rooftops bounding box shape
        logger.info("Clip image with bounding box")
        out_image, out_transform = rio.mask.mask(image, bbox_shape, crop = True)
        out_meta = image.meta
        out_meta.update({"driver": "GTiff",
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform})

        feature_path = os.path.join(OUTPUT_DIR, f"tile_EGID_{int(i)}.tif")
        with rasterio.open(feature_path, "w", **out_meta) as dest:
            dest.write(out_image)
        written_files.append(feature_path)  
        logger.info(f"...done. A file was written: {feature_path}")

        # Delete the produced image mosaic 
        try:
            if os.path.exists(mosaic_path):
                os.remove(mosaic_path)
        except NameError:
            print()

    print()
    logger.info("The following files were written. Let's check them out!")
    for written_file in written_files:
        logger.info(written_file)
    print()

    # Stop chronometer  
    toc = time.time()
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()