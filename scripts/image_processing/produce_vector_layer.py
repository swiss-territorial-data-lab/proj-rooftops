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

import re
import geopandas as gpd

sys.path.insert(1, 'scripts')
import functions.fct_misc as fct_misc

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
    DETECTION_DIR = cfg['detection_dir']
    SHP_ROOFS = cfg['shp_roofs']
    OUTPUT_DIR = cfg['output_dir']
    SHP_EXT = cfg['vector_extension']
    SRS = cfg['srs']

    os.chdir(WORKING_DIR)

    # Create an output directory in case it doesn't exist
    fct_misc.ensure_dir_exists(OUTPUT_DIR)

    written_files = []

    # Get the rooftops shapes
    ROOFS_DIR, ROOFS_NAME = os.path.split(SHP_ROOFS)

    new_filename=os.path.splitext(ROOFS_NAME)[0]  + "_EGID.shp"
    feature_path = os.path.join(ROOFS_DIR, new_filename)

    if os.path.exists(feature_path):
        logger.info(f"File {new_filename} already exists")
        rooftops = gpd.read_file(feature_path)
    else:
        logger.info(f"File {new_filename} does not exist")
        logger.info(f"Create it")
        gdf_roofs = gpd.read_file(os.path.join(WORKING_DIR, ROOFS_DIR,  ROOFS_NAME))
        logger.info(f"Dissolved shapes by EGID number")
        rooftops = gdf_roofs.dissolve('EGID', as_index=False)
        rooftops.drop(['OBJECTID', 'ALTI_MAX', 'ALTI_MIN', 'DATE_LEVE', 'SHAPE_AREA', 'SHAPE_LEN'], axis=1, inplace=True)
        rooftops.to_file(feature_path)
        written_files.append(feature_path)  
        logger.info(f"...done. A file was written: {feature_path}")

    # Read all the shapefile produced, filter them with rooftop extension and merge them in a single layer  
    logger.info(f"Read shapefiles' name")
    tiles=glob(os.path.join(DETECTION_DIR, '*.' + SHP_EXT))
    print(os.path.join(DETECTION_DIR, '*.' + SHP_EXT))

    vector_layer = gpd.GeoDataFrame() 

    for tile in tqdm(tiles, desc='Read detection shapefiles', total=len(tiles)):

        objects = gpd.read_file(tile)

        # Set CRS
        objects.crs = SRS
        shape_objects = objects.dissolve('value', as_index=False)
        egid = float(re.sub('[^0-9]','', os.path.basename(tile)))
        shape_egid  = rooftops[rooftops['EGID'] == egid]
        shape_egid.buffer(0)
        # shape_egid.geometry = shape_egid.geometry.buffer(1)

        fct_misc.test_crs(shape_objects, shape_egid)

        selection = shape_objects.sjoin(shape_egid, how='inner', predicate="within")
        selection['area'] = selection.area 
        final_gdf = selection.drop(['index_right'], axis=1)
        feature_path = os.path.join(OUTPUT_DIR, f"tile_EGID_{int(egid)}_segment_selection.gpkg")
        final_gdf.to_file(feature_path)
        written_files.append(feature_path)  
        
        # Merge/Combine multiple shapefiles into one
        vector_layer = gpd.pd.concat([vector_layer, final_gdf], ignore_index=True)
    
    feature_path = os.path.join(OUTPUT_DIR, "SAM_vector_layer.gpkg")
    vector_layer.to_file(feature_path)
    written_files.append(feature_path)  
    logger.info(f"...done. A file was written: {feature_path}")

    print()
    logger.info("The following files were written. Let's check them out!")
    for written_file in written_files:
        logger.info(written_file)
    print()

    # Stop chronometer  
    toc = time.time()
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()