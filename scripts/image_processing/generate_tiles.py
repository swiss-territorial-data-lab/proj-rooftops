#!/bin/python
# -*- coding: utf-8 -*-

#  proj-rooftops
#
#      Clemence Herny 
#      Gwenaelle Salamin
#      Alessandro Cerioni 


import os
import sys
import time
import argparse
import yaml
from loguru import logger
from tqdm import tqdm

import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.merge import merge

# the following allows us to import modules from within this file's parent folder
sys.path.insert(1, 'scripts')
import functions.fct_misc as misc

logger = misc.format_logger(logger)


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
    TILES = cfg['tiles']
    ROOFS = cfg['roofs']
    OUTPUT_DIR = cfg['output_dir']
    BUFFER = cfg['buffer']
    MASK = cfg['mask']

    os.chdir(WORKING_DIR)

    # Create an output directory in case it doesn't exist
    misc.ensure_dir_exists(OUTPUT_DIR)

    written_files = []

    # Get the rooftops shapes
    ROOFS_DIR, ROOFS_NAME = os.path.split(ROOFS)
    attribute = 'EGID'
    original_file_path = os.path.join(ROOFS_DIR, ROOFS_NAME)
    desired_file_path = os.path.join(ROOFS_DIR, ROOFS_NAME[:-4]  + "_" + attribute + ".shp")
    
    roofs = misc.dissolve_by_attribute(desired_file_path, original_file_path, name=ROOFS_NAME[:-4], attribute=attribute)

    # AOI 
    logger.info("Select rooftop's shapes in the AOI")
    tiles = gpd.read_file(TILES)

    # Get the EGID list from file
    roofs_gdf = roofs.copy()
    logger.info(f"Number of building to process: {len(roofs_gdf)}")

    # Find the image's tiles intersecting the rooftops' shapes  
    logger.info("Intersection of rooftop shapes and image tiles")
    join_tiles_roofs = gpd.sjoin(roofs_gdf, tiles, how="left")
    
    image_list = []  
    egid_list = []
    coords_list = [] 

    # Open or produce the rooftops boundary boxes shapes 
    feature_path = os.path.join(OUTPUT_DIR, 'bbox.gpkg')

    if os.path.exists(feature_path):
        logger.info(f"File bbox.gpkg already exists")
        bbox_list = gpd.read_file(feature_path)
    else:
        logger.info(f"File bbox.gpkg does not exist")
        logger.info(f"Create it")

        for row in join_tiles_roofs.itertuples():
            egid = row.EGID
            egid_list.append(egid)
            bounds = row.geometry.bounds
            coords = misc.bbox(bounds)
            coords_list.append(coords)

        bbox_list = gpd.GeoDataFrame(pd.DataFrame(egid_list, columns=['EGID']), crs='epsg:2056', geometry=coords_list).drop_duplicates(subset='EGID')

        bbox_list.to_file(feature_path)
        written_files.append(feature_path)  
        logger.info(f"...done. A file was written: {feature_path}")


    # Get the image tile(s) number intersecting the rooftop shape 
    logger.info("Find the number of image tile(s) intersecting the rooftop shape")
    unique_egid = join_tiles_roofs["EGID"].unique() 

    if MASK:
        logger.info("Applying building mask")

    for egid in tqdm(unique_egid, desc="Production of tiles fitting the roof's extent", total=len(unique_egid)):
        image_list = join_tiles_roofs.loc[join_tiles_roofs['EGID'] == egid, 'TileName'].to_numpy().tolist()

        tiles_list = [os.path.join(IMAGE_DIR, image + '.tif') for image in image_list]   

        raster_to_mosaic = []   
        for tilepath in tiles_list:
            raster = rasterio.open(tilepath)

            # Mosaic images if rooftops shape is spread over several tiles 
            if len(tiles_list) > 1:  
                raster_to_mosaic.append(raster)

        if len(tiles_list) > 1:
            mosaic, output = merge(raster_to_mosaic)

            output_meta = raster.meta.copy()
            output_meta.update(
            {"driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": output
            })
            
            mosaic_path = os.path.join(OUTPUT_DIR, 'mosaic.tif')
            with rasterio.open(mosaic_path, "w", **output_meta) as dst_mosaic:
                dst_mosaic.write(mosaic)

            raster = rasterio.open(mosaic_path)
            tile = mosaic_path

        image = raster

        if MASK:
            egid_shape = roofs.loc[roofs['EGID'] == egid, 'geometry'].buffer(BUFFER, join_style=2)                

            mask_image, mask_transform = mask(raster, egid_shape)
            mask_meta=raster.meta

            mask_meta.update({'transform': mask_transform})
            feature_mask_path=os.path.join(misc.ensure_dir_exists(os.path.join(OUTPUT_DIR, 'masked_images')),
                                    f"tile_EGID_{int(egid)}_masked.tif")
                
            with rasterio.open(feature_mask_path, 'w', **mask_meta) as dst:
                dst.write(mask_image)
            raster = rasterio.open(feature_mask_path)

            image = raster

        bbox_shape = bbox_list.loc[bbox_list['EGID'] == egid, 'geometry'].buffer(BUFFER, join_style=2)  

        # Clip image by the rooftops bounding box shape
        out_image, out_transform = mask(image, bbox_shape, crop = True)
        out_meta = image.meta
        out_meta.update({"driver": "GTiff",
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform,
                        "crs": rasterio.CRS.from_epsg(2056)})

        if MASK:
           feature_path = feature_mask_path
        else: 
            feature_path = os.path.join(OUTPUT_DIR, f"tile_EGID_{int(egid)}.tif")
        
        # Close readers so we can re-write, erase file
        del raster, image
        with rasterio.open(feature_path, "w", **out_meta) as dst:
            dst.write(out_image)
        written_files.append(feature_path)  

        # Delete the produced image mosaic 
        try:
            if os.path.exists(mosaic_path):
                os.remove(mosaic_path)
        except NameError:
            pass

    print()
    logger.info("The following files were written. Let's check them out!")
    for written_file in written_files:
        logger.info(written_file)
    print()

    # Stop chronometer  
    toc = time.time()
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()