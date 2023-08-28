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
from glob import glob
from loguru import logger
from tqdm import tqdm

import re
import geopandas as gpd

# the following allows us to import modules from within this file's parent folder
sys.path.insert(1, 'scripts')
import functions.fct_misc as misc

logger = misc.format_logger(logger)

def main(WORKING_DIR, ROOFS, OUTPUT_DIR, SHP_EXT, SRS):

    os.chdir(WORKING_DIR)

    # Create an output directory in case it doesn't exist
    misc.ensure_dir_exists(OUTPUT_DIR)
    detection_dir = os.path.join(OUTPUT_DIR, 'segmented_images')

    written_files = []

    # Get the rooftops shapes
    rooftops = gpd.read_file(ROOFS)

    # Read all the shapefile produced, filter them with rooftop extension and merge them in a single layer  
    logger.info(f"Read shapefiles' name")
    tiles = glob(os.path.join(detection_dir, '*.' + SHP_EXT))
    vector_layer = gpd.GeoDataFrame() 

    for tile in tqdm(tiles, desc='Read detection shapefiles', total=len(tiles)):
    
        objects = gpd.read_file(tile)

        # Set CRS
        objects.crs = SRS
        object_shp = objects.dissolve('value', as_index=False) 
        # object_shp = objects.dissolve('mask_value', as_index=False)
        egid = float(re.sub('[^0-9]','', os.path.basename(tile)))
        print('egid', egid, tile)

        egid_shp = rooftops[rooftops['EGID'] == egid]
        egid_shp.buffer(0)
        # egid_shp.geometry = egid_shp.geometry.buffer(1)

        misc.test_crs(object_shp, egid_shp)

        selection = object_shp.sjoin(egid_shp, how='inner', predicate="within")
        selection['area'] = selection.area 
        final_gdf = selection.drop(['index_right'], axis=1)
        
        # Merge/Combine multiple shapefiles into one gdf
        vector_layer = gpd.pd.concat([vector_layer, final_gdf], ignore_index=True)

    # Save the vectors for each EGID in layers in a gpkg 
    feature_path = os.path.join(OUTPUT_DIR, "SAM_egid_vectors.gpkg")
    for egid in vector_layer.EGID.unique():
        vector_layer[vector_layer.EGID == egid].to_file(feature_path, driver="GPKG", layer=str(int(egid)))
    written_files.append(feature_path)  
    logger.info(f"...done. A file was written: {feature_path}")

    # Save all the vectors in a gpkg 
    feature_path = os.path.join(OUTPUT_DIR, "SAM_vectors.gpkg")
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
    ROOFS = cfg['roofs']
    OUTPUT_DIR = cfg['output_dir']
    SHP_EXT = cfg['vector_extension']
    SRS = cfg['srs']

    main(WORKING_DIR, ROOFS, OUTPUT_DIR, SHP_EXT, SRS)

    # Stop chronometer  
    toc = time.time()
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()