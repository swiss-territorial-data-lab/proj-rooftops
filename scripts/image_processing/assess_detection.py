#!/bin/python
# -*- coding: utf-8 -*-

#  Proj rooftops
#
#      Clemence Herny 
#      Gwenaelle Salamin
#      Alessandro Cerioni 
#      Copyright (c) 2020 Republic and Canton of Geneva
#

import os, sys
import time
import argparse
import yaml
import pandas as pd
import geopandas as gpd
from loguru import logger
# the following allows us to import modules from within this file's parent folder
sys.path.insert(1, 'scripts')
# import functions.fct_misc as fct_misc
import functions.common as fct

logger.remove()
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}", level="INFO")



if __name__ == "__main__":

    # Start chronometer
    tic = time.time()
    logger.info('Starting...')

    # Argument and parameter specification
    parser = argparse.ArgumentParser(description="Results assessment of object detection by SAM (STDL.proj-rooftops)")
    parser.add_argument('config_file', type=str, help='Framework configuration file')
    args = parser.parse_args()

    logger.info(f"Using {args.config_file} as config file.")
 
    with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)[os.path.basename(__file__)]


    # Load input parameters
    WORKING_DIR = cfg['working_dir']
    GT = cfg['gt']
    DETECTION = cfg['detection']
    OUTPUT_DIR = cfg['output_dir']
    EGID = cfg['egid']

    os.chdir(WORKING_DIR)

    # Create an output directory in case it doesn't exist
    fct.ensure_dir_exists(OUTPUT_DIR)

    written_files = []

    # Open shapefiles
    gdf_gt = gpd.read_file(GT)
    gdf_gt = gdf_gt[gdf_gt['occupation'] == 1]
    gdf_gt['ID_GT'] = gdf_gt.index
    gdf_gt = gdf_gt.rename(columns={"area": "area_GT"})
    logger.info(f"Read GT file: {len(gdf_gt)} shapes")

    gdf_detec = gpd.read_file(DETECTION)
    gdf_detec = gdf_detec# [gdf_detec['occupation'] == 1]
    gdf_detec['ID_DET'] = gdf_detec.index
    gdf_detec = gdf_detec.rename(columns={"area": "area_DET"})
    logger.info(f"Read detection file: {len(gdf_detec)} shapes")


    logger.info(f"Metrics computation:")
    logger.info(f" - Compute TP, FP and FN")


    tp_gdf, fp_gdf, fn_gdf = fct.get_fractional_sets(
                            gdf_detec, gdf_gt)

    # Tag predictions   
    tp_gdf['tag'] = 'TP'
    fp_gdf['tag'] = 'FP'
    fn_gdf['tag'] = 'FN'

    # Compute metrics
    precision, recall, f1 = fct.get_metrics(tp_gdf, fp_gdf, fn_gdf)

    TP = len(tp_gdf)
    FP = len(fp_gdf)
    FN = len(fn_gdf)

    logger.info(f"   TP = {TP}, FP = {FP}, FN = {FN}")
    logger.info(f"   precision = {precision:.2f}, recall = {recall:.2f}, f1 = {f1:.2f}")
    logger.info(f" - Compute mean Jaccard index")
    iou_average = tp_gdf['IOU'].mean()
    logger.info(f"   IOU average = {iou_average:.2f}")


    # Set the final dataframe with tagged prediction
    logger.info(f"Set the final dataframe")

    tagged_preds_gdf = []
    tagged_preds_gdf_dict = pd.concat([tp_gdf, fp_gdf, fn_gdf])
    tagged_preds_gdf_dict = tagged_preds_gdf_dict.drop(['index_right', 'path', 'layer', 'occupation', 'geom_GT', 'geom_DET'], axis = 1)
    tagged_preds_gdf_dict = tagged_preds_gdf_dict.rename(columns={'value': 'mask_value'})
    # tagged_preds_gdf_dict = tagged_preds_gdf_dict.reindex(columns=['fid','EGID', 'mask_value', 'ID_DET', 'area_DET', 'ID_GT', 'area_GT', 'IOU', 'tag', 'geometry'])

    feature_path = os.path.join(OUTPUT_DIR, f'tagged_predictions.gpkg')
    # tagged_preds_gdf[['geometry', 'score', 'tag', 'dataset']].to_file(file_to_write, driver='GPKG', index=False)
    tagged_preds_gdf_dict.to_file(feature_path, driver='GPKG', index=False)
    written_files.append(feature_path)
    
    print()
    logger.info("The following files were written. Let's check them out!")
    for written_file in written_files:
        logger.info(written_file)

    # Stop chronometer  
    toc = time.time()
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()