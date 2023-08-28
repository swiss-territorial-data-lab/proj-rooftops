#!/bin/python
# -*- coding: utf-8 -*-

#  proj-rooftops
#
#      Clemence Herny 
#      Gwenaelle Salamin
#      Alessandro Cerioni 


import os, sys
import time
import argparse
from loguru import logger
from yaml import load, FullLoader

import numpy as np
import pandas as pd
import geopandas as gpd

sys.path.insert(1, 'scripts')
import functions.fct_misc as fct_misc
import functions.fct_metrics as metrics

logger = fct_misc.format_logger(logger)

# Define functions --------------------------

def main(WORKING_DIR, OUTPUT_DIR, DETECTIONS, GT, EGIDS, METHOD):
    """Assess the results by calculating the precision, recall and f1-score.

    Args:
        WORKING_DIR (path): working directory
        OUTPUT_DIR (path): output directory
        DETECTIONS (path): file of the detections
        GT (path): file of the ground truth
        EGIDS (list): EGIDs of interest
        METHOD (string): method to use for the assessment of the results, either one-to-one or one-to-many.

    Returns:
        float, int: f1-score and number of multiple predictions corresponding to one label.
    """

    os.chdir(WORKING_DIR)

    # Create an output directory in case it doesn't exist
    _ = fct_misc.ensure_dir_exists(OUTPUT_DIR)

    written_files={}

    # Get the EGIDS of interest
    egids=pd.read_csv(EGIDS)
    # Open shapefiles
    gdf_gt = gpd.read_file(GT)
    if 'OBSTACLE' in gdf_gt.columns:
        gdf_gt.rename(columns={'OBSTACLE': 'occupation'}, inplace=True)
    gdf_gt = gdf_gt[(gdf_gt.occupation.astype(int) == 1) & (gdf_gt.EGID.isin(egids.EGID.to_numpy()))]
    gdf_gt['ID_GT'] = gdf_gt.index
    gdf_gt = gdf_gt.rename(columns={"area": "area_GT", 'EGID': 'EGID_GT'})
    nbr_labels=gdf_gt.shape[0]
    logger.info(f"Read GT file: {nbr_labels} shapes")

    if isinstance(DETECTIONS, str):
        gdf_detec = gpd.read_file(DETECTIONS, layer='occupation_for_all_EGIDS')
    elif isinstance(DETECTIONS, gpd.GeoDataFrame):
        gdf_detec = DETECTIONS
    else:
        logger.critical(f'Unrecognized variable type for the detections: {type(DETECTIONS)}')
    gdf_detec = gdf_detec[gdf_detec['occupation'].astype(int) == 1]
    gdf_detec['ID_DET'] = gdf_detec.pred_id
    gdf_detec = gdf_detec.rename(columns={"area": "area_DET"})
    logger.info(f"Read detection file: {len(gdf_detec)} shapes")

    logger.info(f"Metrics computation")
    if METHOD=='one-to-one':
        logger.info('Using the one-to-one method.')
    elif METHOD=='one-to-many':
        logger.info('Using one-to-many method.')
    else:
        logger.warning('Unknown method, defaulting to one-to-one.')

    logger.info(f"Metrics computation:")
    logger.info(f" - Compute TP, FP and FN")

    tp_gdf, fp_gdf, fn_gdf = metrics.get_fractional_sets(gdf_detec, gdf_gt, method=METHOD)

    # Compute metrics
    precision, recall, f1 = metrics.get_metrics(tp_gdf, fp_gdf, fn_gdf)

    TP = tp_gdf.shape[0]
    FP = fp_gdf.shape[0]
    FN = fn_gdf.shape[0]

    if METHOD=='one-to-many':
        tp_with_duplicates=tp_gdf.copy()
        dissolved_tp_gdf=tp_with_duplicates.dissolve(by=['ID_DET'], as_index=False)

        geom1 = dissolved_tp_gdf.geometry.values.tolist()
        geom2 = dissolved_tp_gdf['geom_GT'].values.tolist()
        iou = []
        for (i, ii) in zip(geom1, geom2):
            iou.append(metrics.intersection_over_union(i, ii))
        dissolved_tp_gdf['IOU'] = iou

        tp_gdf=dissolved_tp_gdf.copy()

        logger.info(f'{tp_with_duplicates.shape[0]-tp_gdf.shape[0]} labels are under a shared predictions with at least one other label.')

    logger.info(f"   TP = {TP}, FP = {FP}, FN = {FN}")
    logger.info(f"   precision = {precision:.2f}, recall = {recall:.2f}, f1 = {f1:.2f}")
    logger.info(f" - Compute mean Jaccard index")
    if TP!=0:
        iou_average = tp_gdf['IOU'].mean()
        logger.info(f"   IOU average = {iou_average:.2f}")
        

    nbr_tagged_labels = TP + FN
    diff_in_labels = nbr_labels - nbr_tagged_labels
    filename=os.path.join(OUTPUT_DIR, 'problematic_objects.gpkg')
    if os.path.exists(filename):
        os.remove(filename)
    if diff_in_labels != 0:
        logger.error(f'There are {nbr_labels} labels in input and {nbr_tagged_labels} labels in output.')
        logger.info(f'The list of the problematic labels in exported to {filename}.')

        if diff_in_labels > 0:
            tagged_labels=tp_gdf['ID_GT'].unique().tolist() + fn_gdf['ID_GT'].unique().tolist()

            untagged_gt_gdf=gdf_gt[~gdf_gt['ID_GT'].isin(tagged_labels)]
            untagged_gt_gdf.drop(columns=['geom_GT', 'OBSTACLE'], inplace=True)

            layer_name='missing_label_tags'
            untagged_gt_gdf.to_file(filename, layer=layer_name, index=False)

        elif diff_in_labels < 0:
            all_tagged_labels_gdf=pd.concat([tp_gdf, fn_gdf])

            duplicated_id_gt=all_tagged_labels_gdf.loc[all_tagged_labels_gdf.duplicated(subset=['ID_GT']), 'ID_GT'].unique().tolist()
            duplicated_labels=all_tagged_labels_gdf[all_tagged_labels_gdf['ID_GT'].isin(duplicated_id_gt)]
            duplicated_labels.drop(columns=['geom_GT', 'geom_DET', 'index_right', 'EGID', 'occupation_left', 'occupation_right'], inplace=True)

            layer_name='duplicated_label_tags'
            duplicated_labels.to_file(filename, layer=layer_name, index=False)
        
        written_files[filename]=layer_name


    # Set the final dataframe with tagged prediction
    tagged_preds_gdf = pd.concat([tp_gdf, fp_gdf, fn_gdf])

    tagged_preds_gdf.drop(['index_right', 'occupation_left', 'occupation_right', 'geom_GT', 'geom_DET', 'ID_DET', 'area_DET'], axis = 1, inplace=True)
    tagged_preds_gdf=tagged_preds_gdf.round({'IOU': 2})
    tagged_preds_gdf.reset_index(drop=True, inplace=True)

    layer_name='tagged_predictions'
    feature_path = os.path.join(OUTPUT_DIR, 'tagged_predictions.gpkg')
    tagged_preds_gdf.to_file(feature_path, layer=layer_name, index=False)
    written_files[feature_path]=layer_name


    logger.success("The following files were written. Let's check them out!")
    for path in written_files.keys():
        logger.success(f'  file: {path}, layer: {written_files[path]}')

    return f1, diff_in_labels       # change for 1/(1 + diff_in_labels) if metrics can only be maximized.

# ------------------------------------------

if __name__ == "__main__":
    # Start chronometer
    tic = time.time()
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

    DETECTIONS=cfg['detections']
    GT = cfg['gt']
    EGIDS = cfg['egids']
    METHOD = cfg['method']

    f1, diff_in_labels=main(WORKING_DIR, OUTPUT_DIR, DETECTIONS, GT, EGIDS, METHOD)

    # Stop chronometer  
    toc = time.time()
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()