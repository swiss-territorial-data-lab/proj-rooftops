#!/bin/python
# -*- coding: utf-8 -*-

#  proj-rooftops


import os
import sys
import time
import argparse
import yaml
from glob import glob
from loguru import logger
from tqdm import tqdm

import re
import geopandas as gpd
import pandas as pd


# the following allows us to import modules from within this file's parent folder
sys.path.insert(1, 'scripts')
import functions.fct_misc as misc

logger = misc.format_logger(logger)




def main(WORKING_DIR, EGIDS, ROOFS, OUTPUT_DIR, SHP_EXT, CRS):

    os.chdir(WORKING_DIR)

    # Create an output directory in case it doesn't exist
    misc.ensure_dir_exists(OUTPUT_DIR)
    detection_dir = os.path.join(OUTPUT_DIR, 'segmented_images')
    output_dir = misc.ensure_dir_exists(os.path.join(OUTPUT_DIR, 'vectors'))

    written_files = []

    # Get the EGIDS of interest
    logger.info("- List of selected EGID")
    egids = pd.read_csv(EGIDS)

    # Get the rooftops shapes
    logger.info("- Roofs shapes")
    ROOFS_DIR, ROOFS_NAME = os.path.split(ROOFS)
    attribute = 'EGID'
    original_file_path = os.path.join(ROOFS_DIR, ROOFS_NAME)
    desired_file_path = os.path.join(ROOFS_DIR, ROOFS_NAME[:-4]  + "_" + attribute + ".shp")
    roofs = misc.dissolve_by_attribute(desired_file_path, original_file_path, name=ROOFS_NAME[:-4], attribute=attribute)
    roofs['EGID'] = roofs['EGID'].astype(int)
    roofs_gdf = roofs[roofs.EGID.isin(egids.EGID.to_numpy())]

    logger.info(f"  Number of building to process: {len(roofs_gdf)}")


    # Read all the shapefiles produced, filter them and merge them in a single layer  
    logger.info(f"Read shapefiles' name")
    tiles = glob(os.path.join(detection_dir, '*.' + SHP_EXT))
    vector_layer = gpd.GeoDataFrame() 

    for tile in tqdm(tiles, desc='Read detection shapefiles', total=len(tiles)):

        # Prepare objects shp 
        objects = gpd.read_file(tile)
        objects_shp = objects.copy()
        objects_shp.crs = CRS
        objects_shp['area_shp'] = objects_shp.area 
        objects_shp['geometry_shp'] = objects_shp.geometry
        objects_shp['geometry_noholes_shp'] = objects_shp.apply(misc.fillit, axis=1)
        objects_shp['area_noholes_shp'] = objects_shp.geometry_noholes_shp.area 

        # Prepare roofs shp
        egid = float(re.sub('[^0-9]','', os.path.basename(tile)))
        egid_shp = roofs_gdf[roofs_gdf['EGID'] == egid]
        egid_shp['area_roof'] = egid_shp.area
        egid_shp['geometry_roof'] = egid_shp.geometry
        roof_buffer = 1
        egid_shp.geometry = egid_shp.geometry.buffer(roof_buffer)

        misc.test_crs(objects_shp, egid_shp)

        # Filter vectorised objects. Threshold values have been set
        objects_selection = objects_shp.sjoin(egid_shp, how='inner', predicate="within")
        objects_selection['intersection_frac'] = objects_selection['geometry_roof'].intersection(objects_selection['geometry_shp']).area / objects_selection['area_shp']
        objects_filtered = objects_selection[(objects_selection['area_shp'] >= 0.1) &
                                            (objects_selection['area_noholes_shp'] <= 0.8 * objects_selection['area_roof']) &
                                            (objects_selection['intersection_frac'] >= 0.5)]

        objects_filtered['area'] = objects_filtered.area 

        objects_filtered = objects_filtered.drop(['geometry_shp', 'geometry_noholes_shp', 'geometry_roof'], axis=1)
        objects_clip = gpd.clip(objects_filtered, egid_shp.geometry.buffer(-roof_buffer))
        
        # Merge/Combine multiple shapefiles into one gdf
        vector_layer = gpd.pd.concat([vector_layer, objects_clip], ignore_index=True)

    # # Save the vectors for each EGID in layers in a gpkg !!! Will be deleted at the end !!!
    # feature_path = os.path.join(output_dir, "roof_segmentation_egid.gpkg")
    # for egid in vector_layer.EGID.unique():
    #     vector_layer[vector_layer.EGID == egid].to_file(feature_path, driver="GPKG", layer=str(int(egid)))
    # written_files.append(feature_path)  
    # logger.info(f"...done. A file was written: {feature_path}")

    # Save the vectors layer in a gpkg 
    vector_layer['fid'] = vector_layer.index
    feature_path = os.path.join(output_dir, "roof_segmentation.gpkg")
    vector_layer.to_file(feature_path)
    written_files.append(feature_path)  
    logger.info(f"...done. A file was written: {feature_path}")

    print()
    logger.info("The following files were written. Let's check them out!")
    for written_file in written_files:
        logger.info(written_file)
    print()


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
    # detection_dir = cfg['detection_dir']
    EGIDS = cfg['egids']
    ROOFS = cfg['roofs']
    OUTPUT_DIR = cfg['output_dir']
    SHP_EXT = cfg['vector_extension']
    CRS = cfg['crs']

    main(WORKING_DIR, EGIDS, ROOFS, OUTPUT_DIR, SHP_EXT, CRS)

    # Stop chronometer  
    toc = time.time()
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()