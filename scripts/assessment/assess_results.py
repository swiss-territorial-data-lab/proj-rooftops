#!/bin/python
# -*- coding: utf-8 -*-

#  proj-rooftops


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
import functions.fct_misc as misc
import functions.fct_metrics as metrics

logger = misc.format_logger(logger)

# Define functions --------------------------

def main(WORKING_DIR, OUTPUT_DIR, LABELS, DETECTIONS, ROOFS, EGIDS, METHOD, THRESHOLD):
    """Assess the results by calculating the precision, recall and f1-score.

    Args:
        WORKING_DIR (path): working directory
        OUTPUT_DIR (path): output directory
        DETECTIONS (path): file of the detections
        LABELS (path): file of the ground truth
        ROOFS (path): file of the roof border and main elements
        EGIDS (list): EGIDs of interest
        METHOD (string): method to use for the assessment of the results, either one-to-one or one-to-many.

    Returns:
        float, int: f1-score and number of multiple predictions corresponding to one label.
    """

    os.chdir(WORKING_DIR)

    # Create an output directory in case it doesn't exist
    output_dir = os.path.join(OUTPUT_DIR)
    misc.ensure_dir_exists(output_dir)
    threshold_str = str(THRESHOLD).replace('.', 'dot')
    written_files = {}

    logger.info("Get input data")

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

    # Get labels shapefile
    logger.info("- GT")
    labels_gdf = gpd.read_file(LABELS)
    # if 'OBSTACLE' in labels_gdf.columns:
    #     labels_gdf.rename(columns={'OBSTACLE': 'occupation'}, inplace=True)

    # labels_gdf['fid'] = labels_gdf['fid'].astype(int)
    labels_gdf['type'] = labels_gdf['type'].astype(int)
    labels_gdf['EGID'] = labels_gdf['EGID'].astype(int)

    # Type 12 corresponds to free surfaces, other classes are ojects
    logger.info("  Filter objects and EGID")
    labels_gdf = labels_gdf[(labels_gdf['type'] != 12) & (labels_gdf.EGID.isin(egids.EGID.to_numpy()))]
    labels_gdf['label_id'] = labels_gdf.index
    
    labels_gdf = labels_gdf.explode()
    # labels_gdf = labels_gdf[labels_gdf['geometry'].geom_type.values == 'Polygon']

    # Creat geohash to GT shapes
    logger.info("  Geohashing GT")
    # labels_gdf = labels_gdf.explode()
    GT_PREFIX= 'gt_'
    labels_gdf = misc.add_geohash(labels_gdf, prefix=GT_PREFIX)
    labels_gdf = misc.drop_duplicates(labels_gdf, subset='geohash')

    nbr_labels = labels_gdf.shape[0]
    logger.info(f"  Read labels file: {nbr_labels} shapes")

    # Add geometry attributes
    logger.info("  Add geometry attributes")

    ## Area
    labels_gdf['label area'] = round(labels_gdf.area, 4)

    ## Nearest distance between polygons
    labels_gdf_tmp = labels_gdf.join(roofs_gdf[['EGID', 'geometry']].set_index('EGID'), on='EGID', how='left', lsuffix='_label', rsuffix='_roof', validate='m:1')

    ### Nearest distance between the centroid's polygon of an object to the roof's border 
    geom1 = labels_gdf_tmp['geometry_roof'].to_numpy().tolist()
    geom2 = labels_gdf_tmp['geometry_label'].centroid.to_numpy().tolist()
    nearest_distance = misc.distance_shape(geom1, geom2)
    labels_gdf['nearest_distance_centroid'] = nearest_distance
    labels_gdf['nearest_distance_centroid'] = round(labels_gdf.nearest_distance_centroid, 4)

    ### Nearest distance between the centroid's border of an object to the roof's border 
    geom1 = labels_gdf_tmp['geometry_roof'].to_numpy().tolist()
    geom2 = labels_gdf_tmp['geometry_label'].to_numpy().tolist()
    nearest_distance = misc.distance_shape(geom1, geom2)
    labels_gdf['nearest_distance_border'] = nearest_distance
    labels_gdf['nearest_distance_border'] = round(labels_gdf.nearest_distance_border, 4)
    
    labels_gdf = labels_gdf.drop(columns=['fid'], axis=1)
    feature_path = os.path.join(output_dir, 'labels.gpkg')
    labels_gdf.to_file(feature_path, index=False)

    # Get detections shapefile
    logger.info("- Detections")
    detections_gdf = gpd.read_file(DETECTIONS)

    # detections_gdf = detections_gdf[detections_gdf['geometry'].geom_type.values == 'Polygon']
    detections_gdf = detections_gdf.explode()

    logger.info("  Geohashing detections")
    DETS_PREFIX = "dt_"
    detections_gdf = misc.add_geohash(detections_gdf, prefix=DETS_PREFIX)
    detections_gdf = misc.drop_duplicates(detections_gdf, subset='geohash')

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
    elif METHOD == 'many-to-many':
        logger.info('Using many-to-many method.')
    else:
        logger.warning('Unknown method, default one-to-one.')

    logger.info(f"Metrics computation:")
    logger.info(f"- Count TP, FP and FN")

    metrics_df = pd.DataFrame()

    if METHOD == 'many-to-many':
        tagged_gt_gdf, tagged_dets_gdf = metrics.tag(gt=labels_gdf, dets=detections_gdf, gt_buffer=-0.05, gt_prefix=GT_PREFIX, dets_prefix=DETS_PREFIX, threshold=THRESHOLD)

        logger.info("- Global metrics")

        TP, FP, FN = metrics.get_count(tagged_gt_gdf, tagged_dets_gdf)
        metrics_results = metrics.get_metrics(TP, FP, FN)
        tmp_df = pd.DataFrame.from_records([{'EGID': 'ALL', **metrics_results}])
        metrics_df = pd.concat([metrics_df, tmp_df])

        logger.info("- Per egid metrics")
        for egid in sorted(labels_gdf.EGID.unique()):
            TP, FP, FN = metrics.get_count(
                tagged_gt = tagged_gt_gdf[tagged_gt_gdf.EGID == egid],
                tagged_dets = tagged_dets_gdf[tagged_dets_gdf.EGID == egid],
            )
            metrics_results = metrics.get_metrics(TP, FP, FN)
            tmp_df = pd.DataFrame.from_records([{'EGID': egid, **metrics_results}])
            metrics_df = pd.concat([metrics_df, tmp_df])

        # Get output files 
        feature_path = os.path.join(output_dir, 'detection_tags.gpkg')
        layer_name = 'tagged_labels_' + METHOD + '_thd_' + threshold_str
        tagged_gt_gdf.astype({'TP_charge': 'str', 'FN_charge': 'str'}).to_file(feature_path, layer=layer_name, driver='GPKG')
        written_files[feature_path] = layer_name
        layer_name = 'tagged_detections_' + METHOD + '_thd_' + threshold_str
        tagged_dets_gdf.astype({'TP_charge': 'str', 'FP_charge': 'str'}).to_file(feature_path, layer=layer_name, driver='GPKG')


    else:
        # Count 
        tp_gdf, fp_gdf, fn_gdf = metrics.get_fractional_sets(detections_gdf, labels_gdf, method=METHOD)
        TP = len(tp_gdf)
        FP = len(fp_gdf)
        FN = len(fn_gdf)
        
        # Compute metrics
        logger.info("- Global metrics")
        metrics_results = metrics.get_metrics(TP, FP, FN)
        tmp_df = pd.DataFrame.from_records([{'EGID': 'ALL', **metrics_results}])
        metrics_df = pd.concat([metrics_df, tmp_df])

        logger.info("- Per egid metrics")
        for egid in sorted(labels_gdf.EGID.unique()):
            tp_gdf, fp_gdf, fn_gdf = metrics.get_fractional_sets(detections_gdf, labels_gdf, method=METHOD)
            TP = len(tp_gdf)
            FP = len(fp_gdf)
            FN = len(fn_gdf)
            metrics_results = metrics.get_metrics(TP, FP, FN)
            tmp_df = pd.DataFrame.from_records([{'EGID': egid, **metrics_results}])
            metrics_df = pd.concat([metrics_df, tmp_df])

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

            logger.info(f'{tp_with_duplicates.shape[0] - tp_gdf.shape[0]} labels are under a shared detections with at least one other label.')

        # Set the final dataframe with tagged prediction
        tagged_dets_gdf = pd.concat([tp_gdf, fp_gdf, fn_gdf])
        tagged_dets_gdf.drop(['index_right', 'label_geometry', 'detection_geometry', 'detection_id'], axis=1, inplace=True)
        tagged_dets_gdf = tagged_dets_gdf.round({'IOU': 4})
        tagged_dets_gdf = tagged_dets_gdf.round({'detection_area': 4})
        tagged_dets_gdf.reset_index(drop=True, inplace=True)
        tagged_dets_gdf['fid'] = tagged_dets_gdf.index

        layer_name = 'tagged_detections_' + METHOD + '_thd_' + threshold_str
        feature_path = os.path.join(output_dir, 'detection_tags.gpkg')
        tagged_dets_gdf.to_file(feature_path, layer=layer_name, index=False)

        written_files[feature_path] = layer_name
        

    logger.info(f"- Compute mean Jaccard index")
    # Compute Jaccard index at the scale of a roof (by EGID)
    labels_egid_gdf, detections_egid_gdf = metrics.get_jaccard_index(labels_gdf, detections_gdf, attribute='EGID')
    iou_average = detections_egid_gdf['IOU_EGID'].mean()
    metrics_df['IoU'] = 0
    metrics_df['IoU'] = np.where(metrics_df['EGID'] == 'ALL', iou_average,metrics_df['IoU'])    

    for egid in sorted(labels_gdf.EGID.unique()):
        labels_egid_gdf, detections_egid_gdf = metrics.get_jaccard_index(
            labels_gdf[labels_gdf.EGID == egid], 
            detections_gdf[detections_gdf.EGID == egid], attribute='EGID'
        )
        iou_average = detections_egid_gdf['IOU_EGID'].mean()
        metrics_df['IoU'] = np.where(metrics_df['EGID'] == egid, iou_average,metrics_df['IoU'])

    # Sump-up results and save files
    TP = metrics_df['TP'][metrics_df.EGID == 'ALL'][0]
    FP = metrics_df['FP'][metrics_df.EGID == 'ALL'][0]
    FN = metrics_df['FN'][metrics_df.EGID == 'ALL'][0]
    precision = metrics_df['precision'][metrics_df.EGID == 'ALL'][0]
    recall = metrics_df['recall'][metrics_df.EGID == 'ALL'][0]
    f1 = metrics_df['f1'][metrics_df.EGID == 'ALL'][0]
    iou = metrics_df['IoU'][metrics_df.EGID == 'ALL'][0]

    written_files[feature_path] = layer_name
    feature_path = os.path.join(output_dir, 'metrics.csv')
    metrics_df.to_csv(feature_path, sep=',', index=False, float_format='%.4f')

    print('')
    logger.info(f"TP = {TP}, FP = {FP}, FN = {FN}")
    logger.info(f"TP+FN = {TP+FN}, TP+FP = {TP+FP}")
    logger.info(f"precision = {precision:.2f}, recall = {recall:.2f}, f1 = {f1:.2f}")
    logger.info(f"IoU for all EGIDs = {iou:.2f}")
    print('')

    # Check if detection or labels have been lost in the prbbbcocess
    nbr_tagged_labels = TP + FN
    labels_diff = nbr_labels - nbr_tagged_labels
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

    logger.success("The following files were written. Let's check them out!")
    for path in written_files.keys():
        logger.success(f'  file: {path}, layer: {written_files[path]}')

    return f1, labels_diff       # change for 1/(1 + diff_in_labels) if metrics can only be maximized.

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

    DETECTIONS=cfg['detections']
    LABELS = cfg['ground_truth']
    ROOFS = cfg['roofs']
    EGIDS = cfg['egids']
    METHOD = cfg['method']
    THRESHOLD = cfg['threshold']

    f1, labels_diff = main(WORKING_DIR, OUTPUT_DIR, LABELS, DETECTIONS, ROOFS, EGIDS, METHOD, THRESHOLD)

    # Stop chronometer  
    toc = time.time()
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()