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
import pandas as pd
import geopandas as gpd

# the following allows us to import modules from within this file's parent folder
sys.path.insert(1, 'scripts')
import functions.fct_misc as misc

logger = misc.format_logger(logger)

def main(WORKING_DIR, OUTPUT_DIR, GT, DETECTION, EGID):

    os.chdir(WORKING_DIR)

    # Create an output directory in case it doesn't exist
    misc.ensure_dir_exists(OUTPUT_DIR)

    written_files = []

    # Open shapefiles
    gt_gdf = gpd.read_file(GT)
    gt_gdf = gt_gdf[gt_gdf['occupation'] == 1]
    gt_gdf['ID_GT'] = gt_gdf.index
    gt_gdf = gt_gdf.rename(columns={"area": "area_GT"})
    nbr_labels = len(gt_gdf)
    logger.info(f"Read GT file: {nbr_labels} shapes")

    detec_gdf = gpd.read_file(DETECTION)
    detec_gdf = detec_gdf# [detec_gdf['occupation'] == 1]
    detec_gdf['ID_DET'] = detec_gdf.index
    detec_gdf = detec_gdf.rename(columns={"area": "area_DET"})
    logger.info(f"Read detection file: {len(detec_gdf)} shapes")

    logger.info(f"Metrics computation:")
    logger.info(f" - Compute TP, FP and FN")

    tp_gdf, fp_gdf, fn_gdf = misc.get_fractional_sets(detec_gdf, gt_gdf)

    # Compute metrics
    precision, recall, f1 = misc.get_metrics(tp_gdf, fp_gdf, fn_gdf)

    TP = len(tp_gdf)
    FP = len(fp_gdf)
    FN = len(fn_gdf)

    logger.info(f"   TP = {TP}, FP = {FP}, FN = {FN}")
    logger.info(f"   precision = {precision:.2f}, recall = {recall:.2f}, f1 = {f1:.2f}")
    logger.info(f" - Compute mean Jaccard index")
    iou_average = tp_gdf['IOU'].mean()
    logger.info(f"   IOU average = {iou_average:.2f}")


    # Check if no label has been lost during the results assessment 
    nbr_tagged_labels = TP + FN 
    filename = os.path.join(OUTPUT_DIR, 'problematic_objects.gpkg')
    if os.path.exists(filename):
        os.remove(filename)
    if nbr_labels != nbr_tagged_labels:
        logger.error(f'There are {nbr_labels} labels in input and {nbr_tagged_labels} labels in output.')
        logger.info(f'The list of the problematic labels is exported to {filename}.')

        if nbr_labels > nbr_tagged_labels:
            tagged_labels = tp_gdf['ID_GT'].unique().tolist() + fn_gdf['ID_GT'].unique().tolist()

            untagged_gt_gdf = gt_gdf[~gt_gdf['ID_GT'].isin(tagged_labels)]
            untagged_gt_gdf.drop(columns=['geom_GT', 'OBSTACLE'], inplace=True)

            layer_name = 'missing_label_tags'
            untagged_gt_gdf.to_file(filename, layer=layer_name, index=False)
            written_files.append(filename)

        elif nbr_labels < nbr_tagged_labels:
            all_tagged_labels_gdf = pd.concat([tp_gdf, fn_gdf])

            duplicated_id_gt = all_tagged_labels_gdf.loc[all_tagged_labels_gdf.duplicated(subset=['ID_GT']), 'ID_GT'].unique().tolist()
            duplicated_labels = all_tagged_labels_gdf[all_tagged_labels_gdf['ID_GT'].isin(duplicated_id_gt)]
            duplicated_labels.drop(columns=['geom_GT', 'OBSTACLE', 'geom_DET', 'index_right', 'fid', 'FID', 'fme_basena'], inplace=True)

            layer_name = 'duplicated_label_tags'
            duplicated_labels.to_file(filename, layer=layer_name, index=False)
            written_files.append(filename)

    # Set the final dataframe with tagged prediction
    logger.info(f"Set the final dataframe")

    tagged_preds_gdf = []
    tagged_preds_gdf_dict = pd.concat([tp_gdf, fp_gdf, fn_gdf])
    tagged_preds_gdf_dict = tagged_preds_gdf_dict.drop(['index_right', 'occupation', 'geom_GT', 'geom_DET'], axis = 1)
    tagged_preds_gdf_dict = tagged_preds_gdf_dict.rename(columns={'value': 'mask_value'})

    feature_path = os.path.join(OUTPUT_DIR, f'tagged_predictions.gpkg')
    tagged_preds_gdf_dict.to_file(feature_path, driver='GPKG', index=False)
    written_files.append(feature_path)
    
    print()
    logger.info("The following files were written. Let's check them out!")
    for written_file in written_files:
        logger.info(written_file)

    return f1

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
    GT_SHP = cfg['gt_shp']
    DETECTION_SHP = cfg['detection_shp']
    OUTPUT_DIR = cfg['output_dir']
    EGID = cfg['egid']

    main(WORKING_DIR, OUTPUT_DIR, GT_SHP, DETECTION_SHP, EGID)

    # Stop chronometer  
    toc = time.time()
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()