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

def main(WORKING_DIR, OUTPUT_DIR, LABELS, DETECTIONS, EGIDS, METHOD):
    """Assess the results by calculating the precision, recall and f1-score.

    Args:
        WORKING_DIR (path): working directory
        OUTPUT_DIR (path): output directory
        DETECTIONS (path): file of the detections
        LABELS (path): file of the ground truth
        EGIDS (list): EGIDs of interest
        METHOD (string): method to use for the assessment of the results, either one-to-one or one-to-many.

    Returns:
        float, int: f1-score and number of multiple predictions corresponding to one label.
    """

    os.chdir(WORKING_DIR)

    # Create an output directory in case it doesn't exist
    output_dir = os.path.join(OUTPUT_DIR, 'vectors')
    fct_misc.ensure_dir_exists(output_dir)

    written_files = {}

    # Get the EGIDS of interest
    egids = pd.read_csv(EGIDS)

    # Open shapefiles
    labels_gdf = gpd.read_file(LABELS)
    # if 'OBSTACLE' in labels_gdf.columns:
    #     labels_gdf.rename(columns={'OBSTACLE': 'occupation'}, inplace=True)

    labels_gdf['fid'] = labels_gdf['fid'].astype(int)
    labels_gdf['class'] = labels_gdf['class'].astype(int)
    labels_gdf['EGID'] = labels_gdf['EGID'].astype(int)

    # Class 12 corresponds to free surfaces, other classes are ojects
    labels_gdf = labels_gdf[(labels_gdf.type != 12) & (labels_gdf.EGID.isin(egids.EGID.to_numpy()))]
    labels_gdf['label_id'] = labels_gdf.index

    nbr_labels = labels_gdf.shape[0]
    logger.info(f"Read LABELS file: {nbr_labels} shapes")


    if isinstance(DETECTIONS, str):
        print('hello')
        # detections_gdf = gpd.read_file(DETECTIONS, layer='occupation_for_all_EGIDS')
        # detections_gdf = gpd.read_file(DETECTIONS, layer='EGID_occupation')
        detections_gdf = gpd.read_file(os.path.join(output_dir, DETECTIONS))
    elif isinstance(DETECTIONS, gpd.GeoDataFrame):
        print('coucou')
        detections_gdf = DETECTIONS
    else:
        logger.critical(f'Unrecognized variable type for the detections: {type(DETECTIONS)}')

    detections_gdf['EGID'] = detections_gdf['EGID'].astype(int)
    if 'value' in detections_gdf.columns:
         detections_gdf.rename(columns={'value': 'detection_id'}, inplace=True)
    detections_gdf['detection_id'] = detections_gdf['detection_id'].astype(int)
    detections_gdf = detections_gdf.rename(columns={"area": "detection_area"})

    logger.info(f"Read detection file: {len(detections_gdf)} shapes")

    # Detections count
    logger.info(f"Method used for detections counting")
    if METHOD == 'one-to-one':
        logger.info('Using the one-to-one method.')
    elif METHOD == 'one-to-many':
        logger.info('Using one-to-many method.')
    else:
        logger.warning('Unknown method, default one-to-one.')

    logger.info(f"Metrics computation:")
    logger.info(f" - Count TP, FP and FN")

    tp_gdf, fp_gdf, fn_gdf = metrics.get_fractional_sets(detections_gdf, labels_gdf, method=METHOD)

    # Compute metrics
    precision, recall, f1 = metrics.get_metrics(tp_gdf, fp_gdf, fn_gdf)

    TP = tp_gdf.shape[0]
    FP = fp_gdf.shape[0]
    FN = fn_gdf.shape[0]

    if METHOD == 'one-to-many':
        tp_with_duplicates = tp_gdf.copy()
        dissolved_tp_gdf = tp_with_duplicates.dissolve(by=['detection_id'], as_index=False)

        geom1 = dissolved_tp_gdf.geometry.values.tolist()
        geom2 = dissolved_tp_gdf['label_geometry'].values.tolist()
        iou = []
        for (i, ii) in zip(geom1, geom2):
            iou.append(metrics.intersection_over_union(i, ii))
        dissolved_tp_gdf['IOU'] = iou

        tp_gdf = dissolved_tp_gdf.copy()

        logger.info(f'{tp_with_duplicates.shape[0] - tp_gdf.shape[0]} labels are under a shared predictions with at least one other label.')

    logger.info(f"   TP = {TP}, FP = {FP}, FN = {FN}")
    logger.info(f"   precision = {precision:.2f}, recall = {recall:.2f}, f1 = {f1:.2f}")
    logger.info(f" - Compute mean Jaccard index")
        
    # Compute Jaccard index at the scale of a roof (by EGID)
    labels_egid_gdf, detections_egid_gdf = metrics.get_jaccard_index_roof(labels_gdf, detections_gdf)
    iou_average = detections_egid_gdf['IOU_EGID'].mean()
    logger.info(f"   averaged IOU for all EGIDs = {iou_average:.2f}")

    # Check if detection or labels have been lost in the process
    nbr_tagged_labels = TP + FN
    labels_diff= nbr_labels - nbr_tagged_labels
    filename = os.path.join(output_dir, 'problematic_objects.gpkg')
    if os.path.exists(filename):
        os.remove(filename)
    if labels_diff != 0:
        logger.error(f'There are {nbr_labels} labels in input and {nbr_tagged_labels} labels in output.')
        logger.info(f'The list of the problematic labels in exported to {filename}.')

        if labels_diff > 0:
            tagged_labels = tp_gdf['label_id'].unique().tolist() + fn_gdf['label_id'].unique().tolist()

            untagged_labels_gdf = labels_gdf[~labels_gdf['label_id'].isin(tagged_labels)]
            untagged_labels_gdf.drop(columns=['label_geometry'], inplace=True)

            layer_name = 'missing_label_tags'
            untagged_labels_gdf.to_file(filename, layer=layer_name, index=False)

        elif labels_diff < 0:
            all_tagged_labels_gdf=pd.concat([tp_gdf, fn_gdf])

            duplicated_label_id = all_tagged_labels_gdf.loc[all_tagged_labels_gdf.duplicated(subset=['label_id']), 'label_id'].unique().tolist()
            duplicated_labels = all_tagged_labels_gdf[all_tagged_labels_gdf['label_id'].isin(duplicated_label_id)]
            duplicated_labels.drop(columns=['label_geometry', 'detection_geometry', 'index_right', 'EGID', 'occupation_left', 'occupation_right'], inplace=True)

            layer_name = 'duplicated_label_tags'
            duplicated_labels.to_file(filename, layer=layer_name, index=False)
        
        written_files[filename] = layer_name


    # Set the final dataframe with tagged prediction
    tagged_preds_gdf = pd.concat([tp_gdf, fp_gdf, fn_gdf])
    tagged_preds_gdf.drop(['index_right', 'label_geometry', 'detection_geometry', 'detection_id', 'detection_area'], axis=1, inplace=True)
    tagged_preds_gdf = tagged_preds_gdf.round({'IOU': 2})
    tagged_preds_gdf.reset_index(drop=True, inplace=True)
    tagged_preds_gdf['fid'] = tagged_preds_gdf.index

    layer_name = 'tagged_predictions'
    feature_path = os.path.join(output_dir, 'tagged_predictions.gpkg')
    tagged_preds_gdf.to_file(feature_path, layer=layer_name, index=False)

    written_files[feature_path] = layer_name

    
    logger.success("The following files were written. Let's check them out!")
    for path in written_files.keys():
        logger.success(f'  file: {path}, layer: {written_files[path]}')

    return f1, labels_diff       # change for 1/(1 + diff_in_labels) if metrics can only be maximized.

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
    LABELS = cfg['ground_truth']
    EGIDS = cfg['egids']
    METHOD = cfg['method']

    f1, labels_diff = main(WORKING_DIR, OUTPUT_DIR, LABELS, DETECTIONS, EGIDS, METHOD)

    # Stop chronometer  
    toc = time.time()
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()