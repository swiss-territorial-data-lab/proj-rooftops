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

def main(WORKING_DIR, EGIDS, ROOFS, OUTPUT_DIR, SHP_EXT, SRS):

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
    roofs_gdf = roofs[roofs.EGID.isin(egids.EGID.to_numpy())]
    roofs_gdf['EGID'] = roofs_gdf['EGID'].astype(int)

    logger.info(f"  Number of building to process: {len(roofs_gdf)}")


    # Read all the shapefile produced, filter them with rooftop extension and merge them in a single layer  
    logger.info(f"Read shapefiles' name")
    tiles = glob(os.path.join(detection_dir, '*.' + SHP_EXT))
    vector_layer = gpd.GeoDataFrame() 

    for tile in tqdm(tiles, desc='Read detection shapefiles', total=len(tiles)):
    
        objects = gpd.read_file(tile)

        # Set CRS
        objects.crs = SRS

        # object_shp = objects.explode(index_parts=True)
        object_shp = objects
        object_shp['area_shp'] = object_shp.area 
        object_shp['geometry_shp'] = object_shp.geometry
        # object_shp = objects.dissolve('mask_value', as_index=False)
        egid = float(re.sub('[^0-9]','', os.path.basename(tile)))

        egid_shp = roofs[roofs['EGID'] == egid]
        # egid_shp = egid_shp.explode(index_parts=True)
        egid_shp['area_roof'] = egid_shp.area
        egid_shp['geometry_roof'] = egid_shp.geometry
        egid_shp.geometry = egid_shp.geometry.buffer(1.0)

        misc.test_crs(object_shp, egid_shp)

        object_selection = object_shp.sjoin(egid_shp, how='inner', predicate="within")

        object_selection['intersection_frac'] = object_selection['geometry_roof'].intersection(object_selection['geometry_shp']).area / object_selection['area_shp']
        object_filtered = object_selection[(object_selection['area_shp'] >= 0.1) &
                                            (object_selection['area_shp'] <= 0.8 * object_selection['area_roof']) &
                                            (object_selection['intersection_frac'] >= 0.5)]

        
        object_filtered['area'] = object_filtered.area 
        final_gdf = object_filtered.drop(['geometry_shp', 'geometry_roof'], axis=1)
        final_gdf = gpd.clip(final_gdf, egid_shp)
        
        # Merge/Combine multiple shapefiles into one gdf
        vector_layer = gpd.pd.concat([vector_layer, final_gdf], ignore_index=True)

    # Save the vectors for each EGID in layers in a gpkg !!! Will be deleted at the end !!!
    feature_path = os.path.join(output_dir, "roof_segmentation_egid.gpkg")
    for egid in vector_layer.EGID.unique():
        vector_layer[vector_layer.EGID == egid].to_file(feature_path, driver="GPKG", layer=str(int(egid)))
    written_files.append(feature_path)  
    logger.info(f"...done. A file was written: {feature_path}")

    # Save all the vectors in a gpkg 
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
    SRS = cfg['srs']

    main(WORKING_DIR, EGIDS, ROOFS, OUTPUT_DIR, SHP_EXT, SRS)

    # Stop chronometer  
    toc = time.time()
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()