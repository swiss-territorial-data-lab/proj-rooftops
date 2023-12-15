#!/bin/python
# -*- coding: utf-8 -*-

#  proj-rooftops

import argparse
import os
import sys
import time
from loguru import logger
from tqdm import tqdm
from yaml import load, FullLoader

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(1, 'scripts')
import functions.fct_figures as figures
import functions.fct_misc as misc
import functions.fct_metrics as metrics

logger = misc.format_logger(logger)

# Functions --------------------------
def read_gpd(DETECTIONS):

    if isinstance(DETECTIONS, str):
        detections_gdf = gpd.read_file(DETECTIONS) #, layer='occupation_for_all_EGIDS')
    elif isinstance(DETECTIONS, gpd.GeoDataFrame):
        detections_gdf = DETECTIONS.copy()
    else:
        logger.critical(f'Unrecognized variable type for the detections: {type(DETECTIONS)}.')
        sys.exit(1)

    if 'occupation' in detections_gdf.columns:
        print(len(detections_gdf))
        detections_gdf = detections_gdf[detections_gdf['occupation'].astype(int) == 1].copy()
        print(len(detections_gdf))

    if 'det_id' in detections_gdf.columns:
        detections_gdf['detection_id'] = detections_gdf.det_id.astype(int)
    else:
        detections_gdf['detection_id'] = detections_gdf.index

    return detections_gdf


def main(WORKING_DIR, OUTPUT_DIR, DETECTIONS_PCD, LAYER_PCD, DETECTIONS_IMG, EGIDS, ROOFS):
    """Assess the results by calculating the precision, recall and f1-score.

    Args:
        WORKING_DIR (path): working directory
        OUTPUT_DIR (path): output directory
        DETECTIONS_PCD (path): file of the pcd segmentation detections
        LAYER_PCD (string): test or training layer
        DETECTIONS_IMG (path): file of the img segmentation detections
        EGIDS (list): EGIDs of interest
        ROOFS (string): file of the roofs. Defaults to None.

    Returns:
    """

    os.chdir(WORKING_DIR)

    # Create an output directory in case it doesn't exist
    output_dir = misc.ensure_dir_exists(os.path.join(OUTPUT_DIR))

    written_files = {}

    logger.info("Get input data")

#    # Get the EGIDS of interest
#     egids = pd.read_csv(EGIDS)
#     array_egids = egids.EGID.to_numpy()
#     logger.info(f"- {egids.shape[0]} selected EGIDs")

#     roofs = gpd.read_file(ROOFS)
#     roofs['EGID'] = roofs['EGID'].astype(int)
#     roofs_gdf = roofs[roofs.EGID.isin(array_egids)].copy()
#     logger.info(f"Read the file for roofs: {len(roofs_gdf)} shapes")
#     feature_path = os.path.join(output_dir, "roofs_egid.gpkg")
#     roofs_gdf.to_file(feature_path, driver="GPKG")  

    # Read detections shapefile 
    DETECTIONS_PCD = os.path.join(DETECTIONS_PCD)
    _ = misc.ensure_file_exists(DETECTIONS_PCD)

    DETECTIONS_IMG = os.path.join(DETECTIONS_IMG)
    _ = misc.ensure_file_exists(DETECTIONS_IMG)
    pcd_gdf = gpd.read_file(DETECTIONS_PCD, layer=LAYER_PCD)
    img_gdf = read_gpd(DETECTIONS_IMG)
    logger.info(f"- {len(pcd_gdf)} detection's shapes")
    logger.info(f"- {len(img_gdf)} detection's shapes")


    pcd_gdf = pcd_gdf.drop(['ID_DET', 'geohash', 'group_id', 'TP_charge', 'FP_charge', 'nearest_distance_centroid', 'nearest_distance_border'], axis=1)
 
    # retain img_geometry  
    pcd_gdf.geometry = pcd_gdf.geometry.buffer(-0.01, join_style='mitre')
    left_join = gpd.sjoin(img_gdf, pcd_gdf, how='left', predicate='intersects', lsuffix='left', rsuffix='right')
    left_join = left_join[left_join['det_id'].notnull()] 
    left_join.drop_duplicates(subset=['detection_id'], inplace=True)
    left_join = left_join.rename(columns={"EGID_left": "EGID", "area_left": "area"})
    left_join = left_join.drop('EGID_right', axis=1)
    feature_path = os.path.join(output_dir, "roof_segmentation.gpkg")
    left_join.to_file(feature_path, driver="GPKG") 

    # layer_clip = gpd.clip(img_gdf, pcd_gdf)
    # print(layer_clip.keys())  
    # layer_clip.geometry = layer_clip.geometry.buffer(0.001, join_style='mitre')
    # layer_clip = layer_clip.unary_union
    # layer_clip = layer_clip.explode(ignore_index=True)
    # # res_union = gpd.overlay(df1, df2, how='union')
    # feature_path = os.path.join(output_dir, "test_clip.gpkg")
    # layer_clip.to_file(feature_path, driver="GPKG") 


    return written_files

# ------------------------------------------

if __name__ == "__main__":

    # Start chronometer
    tic = time.time()
    logger.info("Results assessment")
    logger.info('Starting...')

    # Argument and parameter specification
    parser = argparse.ArgumentParser(description="The script allows to evaluate the workflow results (STDL.proj-rooftops)")
    parser.add_argument('config_file', type=str, help='Framework configuration file')
    args = parser.parse_args()

    logger.info(f"Using {args.config_file} as config file.")

    with open(args.config_file) as fp:
        cfg = load(fp, Loader=FullLoader)[os.path.basename(__file__)]

    # Load input parameters
    WORKING_DIR = cfg['working_dir']
    OUTPUT_DIR = cfg['output_dir']

    DETECTIONS_PCD = cfg['pcd_seg']
    LAYER_PCD = cfg['layer_pcd']
    DETECTIONS_IMG = cfg['img_seg']
    ROOFS = cfg['roofs']
    EGIDS = cfg['egids']


    written_files = main(WORKING_DIR, OUTPUT_DIR, DETECTIONS_PCD, LAYER_PCD, DETECTIONS_IMG, EGIDS, ROOFS)

    logger.success("The following files were written. Let's check them out!")
    for path in written_files.keys():
        logger.success(f'  file: {path}{"" if written_files[path] == "" else f", layer: {written_files[path]}"}')


    # Stop chronometer  
    toc = time.time()
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()