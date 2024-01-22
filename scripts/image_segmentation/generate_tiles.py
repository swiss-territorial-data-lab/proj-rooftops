#!/bin/python
# -*- coding: utf-8 -*-

#  proj-rooftops


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
    logger.info('Generate tiles')
    logger.info('Starting...')

    # Argument and parameter specification
    parser = argparse.ArgumentParser(description="The script prepares the image dataset to process the rooftops project (STDL.proj-rooftops)")
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
    EGIDS = cfg['egids']
    FILTERS=cfg['filters']
    BUILDING_TYPE = FILTERS['building_type']
    ROOF_INCLINATION = FILTERS['roof_inclination']
    OUTPUT_DIR = cfg['output_dir']
    BUFFER = cfg['buffer']
    MASK = cfg['mask']

    os.chdir(WORKING_DIR)

    # Create an output directory in case it doesn't exist
    misc.ensure_dir_exists(OUTPUT_DIR)

    written_files = []

    logger.info("Get input data")

    # Get the EGIDS of interest
    egids = pd.read_csv(EGIDS)
    if BUILDING_TYPE in ['administrative', 'industrial', 'residential']:
        logger.info(f'Only the building with the type "{BUILDING_TYPE}" are considered.')
        egids = egids[egids.roof_type==BUILDING_TYPE].copy()
    elif BUILDING_TYPE != 'all':
        logger.critical('Unknown building type passed.')
        sys.exit(1)
    if ROOF_INCLINATION in ['flat', 'pitched', 'mixed']:
        logger.info(f'Only the roofs with the type "{ROOF_INCLINATION}" are considered.')
        egids = egids[egids.roof_inclination == ROOF_INCLINATION].copy()
    elif ROOF_INCLINATION != 'all':
        logger.critical('Unknown roof type passed.')
        sys.exit(1) 

    feature_path = EGIDS[:-4] + '_' + BUILDING_TYPE + '_' + ROOF_INCLINATION + '.csv'
    egids.to_csv(feature_path, index=False)
    written_files.append(feature_path)  

    # Get the rooftop shapes
    logger.info("- Roof shapes")
    ROOFS_DIR, ROOFS_NAME = os.path.split(ROOFS)
    desired_file_path = ROOFS[:-4]  + "_EGID.shp"
    roofs = misc.dissolve_by_attribute(desired_file_path, ROOFS, name=ROOFS_NAME[:-4], attribute='EGID')
    roofs['EGID'] = roofs['EGID'].astype(int)
    roofs_gdf = roofs[roofs.EGID.isin(egids.EGID.to_numpy())].copy()
    logger.info(f"  Number of buildings to process: {len(roofs_gdf)}")

    # AOI 
    logger.info("- Tiles name")
    tiles = gpd.read_file(TILES)

    # Find the image tiles intersecting the rooftop shapes  
    logger.info("Intersection of rooftop shapes and image tiles")
    join_tiles_roofs = gpd.sjoin(roofs_gdf, tiles, how="left")
    
    image_list = []  
    egid_list = []
    coords_list = [] 

    # Open or produce the bounding boxes of the rooftops
    logger.info("Produce the bounding boxes of the buildings")

    for row in join_tiles_roofs.itertuples():
        egid = row.EGID
        egid_list.append(egid)
        bounds = row.geometry.bounds
        coords = misc.bbox(bounds)
        coords_list.append(coords)

    bbox_list = gpd.GeoDataFrame(pd.DataFrame(egid_list, columns=['EGID']), crs='epsg:2056', geometry=coords_list).drop_duplicates(subset='EGID')
    feature_path = os.path.join(OUTPUT_DIR, 'bbox.gpkg')
    bbox_list.to_file(feature_path)
    written_files.append(feature_path)  

    # Get the number of image tile(s) intersecting the rooftop shape 
    logger.info("Finding the number of image tile(s) intersecting the rooftop shape")
    unique_egid = join_tiles_roofs["EGID"].unique() 

    if MASK:
        logger.info("Applying building mask")

    for egid in tqdm(unique_egid, desc="Production of tiles fitting the extent of the roof", total=len(unique_egid)):
        image_list = join_tiles_roofs.loc[join_tiles_roofs['EGID'] == egid, 'TileName'].to_numpy().tolist()

        tiles_list = [os.path.join(IMAGE_DIR, image + '.tif') for image in image_list]   

        if len(tiles_list) > 1:
            # Mosaic images if the rooftop shape is spread over several tiles 
            raster_to_mosaic = [rasterio.open(tilepath) for tilepath in tiles_list]

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

        else:
            raster = rasterio.open(tiles_list[0])

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

        # Clip image by the bounding boxes of the rooftops
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
        
        # Close reader so we can re-write it
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