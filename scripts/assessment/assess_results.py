#!/bin/python
# -*- coding: utf-8 -*-

#  proj-rooftops

import argparse
import os
import time
import sys
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

def main(WORKING_DIR, OUTPUT_DIR, LABELS, DETECTIONS, EGIDS, ROOFS, method='one-to-one', threshold=0.1, object_parameters=[], ranges=[], 
         additional_metrics=False, visualisation=False):
    """Assess the results by calculating the precision, recall and f1-score.

    Args:
        WORKING_DIR (path): working directory
        OUTPUT_DIR (path): output directory
        LABELS (path): file of the ground truth
        DETECTIONS (path): file of the detections
        EGIDS (list): EGIDs of interest
        ROOFS (string): file of the roofs. Defaults to None.
        method (string): method to use for the assessment of the results, either one-to-one, one-to-many or many-to-many. Defaults ot one-to-one.
        threshold (float): surface intersection threshold between label shape and detection shape to be considered as the same group. Defaults to 0.1.
        object_parameters (list): list of object parameter to be processed ('area', 'nearest_distance_border', 'nearest_distance_centroid')
        ranges (list): list of list of the bins to process by object_parameters.
        additional_metrics (bool): wheter or not to do the by-EGID, by-object, by-class metrics. Defaults to False.
        visualisation (bool): wheter or not to do and save the plots. Defaults to False.

    Returns:
        tuple:
            - float: f1 score of the labels and predictions
            - float: average of the intersections over unions over each building
            - dict: written files and layers
    """

    os.chdir(WORKING_DIR)

    # Create an output directory in case it doesn't exist
    if not visualisation and os.path.exists(os.path.join(OUTPUT_DIR, method)):
        os.system(f"rm -r {os.path.join(OUTPUT_DIR, method)}")
    output_dir = misc.ensure_dir_exists(os.path.join(OUTPUT_DIR, method))
    threshold_str = str(threshold).replace('.', 'dot')

    written_files = {}

    logger.info('Get input data...')

    # Get the EGIDS of interest
    egids = pd.read_csv(EGIDS)
    array_egids = egids.EGID.to_numpy()
    logger.info(f'    - {egids.shape[0]} selected EGIDs.')

    if ('EGID' in ROOFS) | ('egid' in ROOFS):
        roofs_gdf = gpd.read_file(ROOFS)
    else:
        # Get the rooftops shapes
        _, ROOFS_NAME = os.path.split(ROOFS)
        attribute = 'EGID'
        original_file_path = ROOFS
        desired_file_path = os.path.join(OUTPUT_DIR, ROOFS_NAME[:-4] + "_" + attribute + ".shp")

        roofs_gdf = misc.dissolve_by_attribute(desired_file_path, original_file_path, name=ROOFS_NAME[:-4], attribute=attribute)

    roofs_gdf['EGID'] = roofs_gdf['EGID'].astype(int)
    roofs_gdf = roofs_gdf[roofs_gdf.EGID.isin(array_egids)].copy()
    logger.info(f"    - {len(roofs_gdf)} roofs")

    # Read the shapefile for labels
    labels_gdf = gpd.read_file(LABELS)

    if labels_gdf.EGID.dtype != 'int64':
        labels_gdf['EGID'] = [round(float(egid)) for egid in labels_gdf.EGID.to_numpy()]
    if 'type' in labels_gdf.columns:
        labels_gdf['type'] = labels_gdf['type'].astype(int)
        labels_gdf = labels_gdf.rename(columns={'type':'obj_class'})
        # Type 12 corresponds to free surfaces, other classes are objects
        labels_gdf.loc[labels_gdf['obj_class'] == 4, 'descr'] = 'Aero'
        labels_gdf = labels_gdf[(labels_gdf['obj_class'] != 12) & (labels_gdf.EGID.isin(egids.EGID.to_numpy()))].copy()
    else:
        labels_gdf = labels_gdf[labels_gdf.EGID.isin(array_egids)].copy()
        
    # Clip labels to the corresponding roof
    for egid in array_egids:
        labels_egid_gdf = labels_gdf[labels_gdf.EGID==egid].copy()
        labels_egid_gdf = labels_egid_gdf.clip(roofs_gdf.loc[roofs_gdf.EGID==egid, 'geometry'].buffer(-0.01, join_style='mitre'), keep_geom_type=True)

        tmp_gdf = labels_gdf[labels_gdf.EGID!=egid].copy()
        labels_gdf = pd.concat([tmp_gdf, labels_egid_gdf], ignore_index=True)

    labels_gdf['label_id'] = labels_gdf.id
    labels_gdf['area'] = round(labels_gdf.area, 4)

    labels_gdf.drop(columns=['fid', 'type', 'layer', 'path'], inplace=True, errors='ignore')
    labels_gdf=labels_gdf.explode(ignore_index=True).reset_index(drop=True)

    nbr_labels=labels_gdf.shape[0]
    logger.info(f"    - {nbr_labels} labels")

    # Read the shapefile for detections
    if isinstance(DETECTIONS, str):
        detections_gdf = gpd.read_file(DETECTIONS) #, layer='occupation_for_all_EGIDS')
    elif isinstance(DETECTIONS, gpd.GeoDataFrame):
        detections_gdf = DETECTIONS.copy()
    else:
        logger.critical(f'Unrecognized variable type for the detections: {type(DETECTIONS)}.')
        sys.exit(1)
    if 'occupation' in detections_gdf.columns:
        detections_gdf = detections_gdf[detections_gdf['occupation'].astype(int) == 1].copy()
    detections_gdf['EGID'] = detections_gdf.EGID.astype(int)
    if 'det_id' in detections_gdf.columns:
        detections_gdf['ID_DET'] = detections_gdf.det_id.astype(int)
    else:
        detections_gdf['ID_DET'] = detections_gdf.index
    detections_gdf=detections_gdf.explode(index_parts=False).reset_index(drop=True)
    logger.info(f"    - {len(detections_gdf)} detections")
    

    if (len(object_parameters) > 0) and additional_metrics:

        ranges_dict = {object_parameters[i]: ranges[i] for i in range(len(object_parameters))}

        ## Nearest distance between polygons
        labels_gdf = misc.nearest_distance(labels_gdf, roofs_gdf, join_key='EGID', parameter='nearest_distance_centroid', lsuffix='_label', rsuffix='_roof')
        labels_gdf = misc.nearest_distance(labels_gdf, roofs_gdf, join_key='EGID', parameter='nearest_distance_border', lsuffix='_label', rsuffix='_roof')

        detections_gdf = misc.nearest_distance(detections_gdf, roofs_gdf, join_key='EGID', parameter='nearest_distance_centroid', lsuffix='_detection', rsuffix='_roof')
        detections_gdf = misc.nearest_distance(detections_gdf, roofs_gdf, join_key='EGID', parameter='nearest_distance_border', lsuffix='_detection', rsuffix='_roof')

    # Detections count
    logger.info(f"Method used for detections counting:")
    methods_list =  ['one-to-one', 'one-to-many', 'charges', 'fusion']
    if method in methods_list:
        logger.info(f'    {method} method')
    else:
        logger.warning('    Unknown method, default method = one-to-one.')


    metrics_egid_df = pd.DataFrame()
    metrics_objects_df = pd.DataFrame()

    if detections_gdf.shape[0]==0:
        logger.error('No detection is available, returning 0 as f1 score and IoU average.')
        return 0, 0, []
    
    elif method == 'charges' or method == 'fusion':

        logger.info("Geohash the labels and detections to use them in graphs...")
        GT_PREFIX= 'gt_'
        labels_gdf = misc.add_geohash(labels_gdf, prefix=GT_PREFIX)
        labels_gdf = misc.drop_duplicates(labels_gdf, subset='geohash')
        nbr_labels = labels_gdf.shape[0]

        DETS_PREFIX = "dt_"
        detections_gdf = misc.add_geohash(detections_gdf, prefix=DETS_PREFIX)
        detections_gdf = misc.drop_duplicates(detections_gdf, subset='geohash')

        logger.info(f"Metrics computation:")
        logger.info(f"     - Compute TP, FP and FN")

        tagged_gt_gdf, tagged_dets_gdf = metrics.tag(gt=labels_gdf, dets=detections_gdf,
                                                    threshold=threshold, method=method, buffer=-0.01, 
                                                    gt_prefix=GT_PREFIX, dets_prefix=DETS_PREFIX, group_attribute='EGID')
        feature_path = os.path.join(output_dir, 'tags.gpkg')
        
        if method=='fusion':
            tagged_final_gdf = pd.concat([tagged_dets_gdf, 
                                        tagged_gt_gdf[tagged_gt_gdf.FN_charge == 1]
                                        ]).reset_index(drop=True)

            tagged_final_gdf['tag'] = None
            tagged_final_gdf.loc[tagged_final_gdf['TP_charge'] >= 1, 'tag'] = 'TP' 
            tagged_final_gdf.loc[(tagged_final_gdf['FP_charge'] >= 1) & (tagged_final_gdf['TP_charge'] == 0), 'tag'] = 'FP' 
            tagged_final_gdf.loc[(tagged_final_gdf['FN_charge'] >= 1) & (tagged_final_gdf['TP_charge'] == 0), 'tag'] = 'FN' 
            tagged_final_gdf = tagged_final_gdf.reindex(columns=['id', 'EGID', 'geohash', 'label_id', 'detection_id', 'obj_class', 
                                                        'descr', 'area', 'nearest_distance_centroid', 'nearest_distance_border', 
                                                        'group_id', 'TP_charge', 'FN_charge', 'FP_charge', 'tag', 'geometry'])
            
            layer_name = 'tagged_final_' + method + '_thd_' + threshold_str
            tagged_final_gdf.astype({'TP_charge': 'str', 'FP_charge': 'str', 'FN_charge': 'str'}).to_file(feature_path, layer=layer_name, driver='GPKG')
            written_files[feature_path] = layer_name

        # Get output files 
        layer_name = 'tagged_labels_' + method + '_thd_' + threshold_str
        tagged_gt_gdf.astype({'TP_charge': 'str', 'FN_charge': 'str'}).to_file(feature_path, layer=layer_name, driver='GPKG')
        written_files[feature_path] = layer_name
        layer_name = 'tagged_detections_' + method + '_thd_' + threshold_str
        tagged_dets_gdf.astype({'TP_charge': 'str', 'FP_charge': 'str'}).to_file(feature_path, layer=layer_name, driver='GPKG')
        written_files[feature_path] = layer_name

        logger.info("    - Global metrics")
        TP, FP, FN = metrics.get_count(tagged_gt_gdf, tagged_dets_gdf)
        metrics_results = metrics.get_metrics(TP, FP, FN)
        metrics_df = pd.DataFrame(metrics_results, index=[0])

        if additional_metrics:
            logger.info("    - Metrics per egid")
            for egid in tqdm(sorted(labels_gdf.EGID.unique()), desc='Per-EGID metrics'):
                TP, FP, FN = metrics.get_count(
                    tagged_gt = tagged_gt_gdf[tagged_gt_gdf.EGID == egid],
                    tagged_dets = tagged_dets_gdf[tagged_dets_gdf.EGID == egid],
                )
                
                metrics_results = metrics.get_metrics(TP, FP, FN)
                tmp_df = pd.DataFrame.from_records([{'EGID': egid, **metrics_results}])
                metrics_egid_df = pd.concat([metrics_egid_df, tmp_df])

            logger.info("    - Metrics per object class")
            for object_class in sorted(labels_gdf.descr.unique()):
                filter_gt_gdf = tagged_gt_gdf[tagged_gt_gdf['descr']==object_class]
                    
                TP = float(filter_gt_gdf['TP_charge'].sum())
                FN = float(filter_gt_gdf['FN_charge'].sum())
                FP = 0

                metrics_results = metrics.get_metrics(TP, FP, FN)
                rem_list = ['FP', 'precision', 'f1']
                [metrics_results.pop(key) for key in rem_list]
                tmp_df = pd.DataFrame.from_records([{'attribute': 'object_class', 'value': object_class, **metrics_results}])
                metrics_objects_df = pd.concat([metrics_objects_df, tmp_df])

            if (len(object_class)>0) and isinstance(roofs_gdf, gpd.GeoDataFrame):
                logger.info("    - Metrics per object attributes")
                for parameter in object_parameters:
                    param_ranges = ranges_dict[parameter] 
                    for lim_inf, lim_sup in param_ranges:
                        filter_gt_gdf = tagged_gt_gdf[(tagged_gt_gdf[parameter] >= lim_inf) & (tagged_gt_gdf[parameter] <= lim_sup)]
                        filter_dets_gdf = tagged_dets_gdf[(tagged_dets_gdf[parameter] >= lim_inf) & (tagged_dets_gdf[parameter] <= lim_sup)]
                        
                        TP = float(filter_gt_gdf['TP_charge'].sum())
                        FP = float(filter_dets_gdf['FP_charge'].sum()) 
                        FN = float(filter_gt_gdf['FN_charge'].sum())

                        metrics_results = metrics.get_metrics(TP, FP, FN)
                        tmp_df = pd.DataFrame.from_records([{'attribute': parameter, 'value': f"{lim_inf}-{lim_sup}", **metrics_results}])
                        metrics_objects_df = pd.concat([metrics_objects_df, tmp_df])

    else:

        logger.info(f"Metrics computation:")
        logger.info(f"   - Compute TP, FP and FN")

        tp_gdf, fp_gdf, fn_gdf = metrics.get_fractional_sets(detections_gdf[['ID_DET', 'geometry']], labels_gdf, method=method, threhold=threshold)
        TP = len(tp_gdf)
        FP = len(fp_gdf)
        FN = len(fn_gdf)

        # Compute metrics
        logger.info("    - Global metrics")
        metrics_results = metrics.get_metrics(TP, FP, FN)
        metrics_df = pd.DataFrame(metrics_results, index=[0])

        if method=='one-to-many':
            tp_with_duplicates=tp_gdf.copy()
            dissolved_tp_gdf=tp_with_duplicates.dissolve(by=['ID_DET'], as_index=False)

            geom1 = dissolved_tp_gdf.geometry.values.tolist()
            geom2 = dissolved_tp_gdf['label_geometry'].values.tolist()
            iou = []
            for (i, ii) in zip(geom1, geom2):
                iou.append(metrics.intersection_over_union(i, ii))
            dissolved_tp_gdf['IOU'] = iou

            tp_gdf=dissolved_tp_gdf.copy()

            logger.info(f'{tp_with_duplicates.shape[0]-tp_gdf.shape[0]} labels are under a shared predictions with at least one other label.')

        # Set the final dataframe with tagged prediction
        tagged_dets_gdf = pd.concat([tp_gdf, fp_gdf, fn_gdf])

        tagged_dets_gdf.drop(['index_right', 'occupation_left', 'occupation_right', 'label_geometry', 'detection_geometry', 'ID_DET', 'area_DET', 'EGID_GT'], 
                             axis = 1, inplace=True, errors='ignore')
        tagged_dets_gdf=tagged_dets_gdf.round({'IOU': 2, 'detection_area': 4})
        tagged_dets_gdf.reset_index(drop=True, inplace=True)

        layer_name = 'tagged_detections_' + method + '_thd_' + threshold_str
        feature_path = os.path.join(output_dir, 'tagged_detections.gpkg')
        tagged_dets_gdf.to_file(feature_path, layer=layer_name, index=False)
        written_files[feature_path]=layer_name

        if additional_metrics:
            logger.info("    - Metrics per egid")
            for egid in tqdm(sorted(labels_gdf.EGID.unique()), desc='Per-EGID metrics'):
                tp_gdf, fp_gdf, fn_gdf = metrics.get_fractional_sets(detections_gdf, labels_gdf, method=method)
                TP = len(tp_gdf)
                FP = len(fp_gdf)
                FN = len(fn_gdf)
                metrics_results = metrics.get_metrics(TP, FP, FN)
                tmp_df = pd.DataFrame.from_records([{'EGID': egid, **metrics_results}])
                metrics_egid_df = pd.concat([metrics_egid_df, tmp_df])


            logger.info("    - Metrics per object class")
            for object_class in sorted(labels_gdf.descr.unique()):
                
                filter_gt_gdf = tagged_dets_gdf[tagged_dets_gdf['descr']==object_class].copy()
                
                TP = len(filter_gt_gdf[filter_gt_gdf['tag'] == 'TP'])
                FN = len(filter_gt_gdf[filter_gt_gdf['tag'] == 'FN'])
                FP = len(filter_gt_gdf[filter_gt_gdf['tag'] == 'FP'])

                metrics_results = metrics.get_metrics(TP, FP, FN)
                rem_list = ['FP', 'precision', 'f1']
                [metrics_results.pop(key) for key in rem_list]
                tmp_df = pd.DataFrame.from_records([{'attribute': 'object_class', 'value': object_class, **metrics_results}])
                metrics_objects_df = pd.concat([metrics_objects_df, tmp_df])

            if (len(object_class)>0) and isinstance(roofs_gdf, gpd.GeoDataFrame):
                logger.info("    - Metrics per object attributes")
                for parameter in object_parameters:
                    param_ranges = ranges_dict[parameter] 
                    for val in param_ranges:
                        filter_dets_gdf = tagged_dets_gdf[(tagged_dets_gdf[parameter] >= val[0]) & (tagged_dets_gdf[parameter] <= val[1])]
                        
                        TP = float(filter_dets_gdf.loc[filter_dets_gdf.tag=='TP'].shape[0])
                        FP = 0
                        FN = float(filter_dets_gdf.loc[filter_dets_gdf.tag=='TP'].shape[0])

                        metrics_results = metrics.get_metrics(TP, FP, FN)
                        tmp_df = pd.DataFrame.from_records([{'attribute': parameter, 'value': str(val).replace(",", " -"), **metrics_results}])
                        metrics_objects_df = pd.concat([metrics_objects_df, tmp_df])


    # Compute Jaccard index by EGID
    logger.info(f"    - Compute mean Jaccard index")

    labels_by_attr_gdf = metrics.get_jaccard_index(labels_gdf, detections_gdf)

    if 'EGID' not in metrics_egid_df.columns:
        metrics_egid_df['EGID'] = labels_by_attr_gdf.EGID

    metrics_egid_df['IoU_EGID'] = [
        labels_by_attr_gdf.loc[labels_by_attr_gdf.EGID==egid, 'IOU_EGID'].iloc[0]
        if egid in labels_by_attr_gdf.EGID.unique() else 0
        for egid in metrics_egid_df.EGID 
    ]
    feature_path = os.path.join(output_dir, 'metrics_per_EGID.csv')
    metrics_egid_df.round(3).to_csv(feature_path, index=False)
    written_files[feature_path] = ''
        
    # Compute Jaccard index for all buildings
    metrics_df['IoU_mean'] = round(metrics_egid_df['IoU_EGID'].mean(), 3)
    metrics_df['IoU_median'] = round(metrics_egid_df['IoU_EGID'].median(), 3)

    feature_path = os.path.join(output_dir, 'metrics.csv')
    metrics_df.to_csv(feature_path, index=False)
    written_files[feature_path] = ''

    # Concatenate roof attributes by EGID and get attributes keys
    metrics_egid_df = pd.merge(metrics_egid_df, egids, on='EGID')
    roof_attributes = egids.keys().tolist()
    roof_attributes.remove('EGID')
    if 'nbr_elem' in roof_attributes:
        roof_attributes.remove('nbr_elem')

    # Compute metrics by roof attributes 
    if additional_metrics:
        logger.info("    - Metrics per roof attributes")
        for attribute in roof_attributes:
            metrics_count_df = metrics_egid_df[[attribute, 'TP', 'FP', 'FN']].groupby([attribute], as_index=False).sum()
            metrics_iou_mean_df = metrics_egid_df[[attribute, 'IoU_EGID']].groupby([attribute], as_index=False).mean()
            metrics_iou_median_df = metrics_egid_df[[attribute, 'IoU_EGID']].groupby([attribute], as_index=False).median()
            
            for val in metrics_egid_df[attribute].unique():
                TP = metrics_count_df.loc[metrics_count_df[attribute] == val, 'TP'].iloc[0]  
                FP = metrics_count_df.loc[metrics_count_df[attribute] == val, 'FP'].iloc[0]
                FN = metrics_count_df.loc[metrics_count_df[attribute] == val, 'FN'].iloc[0]
                iou_mean = metrics_iou_mean_df.loc[metrics_iou_mean_df[attribute] == val, 'IoU_EGID'].iloc[0]
                iou_median = metrics_iou_median_df.loc[metrics_iou_median_df[attribute] == val, 'IoU_EGID'].iloc[0]    

                metrics_results = metrics.get_metrics(TP, FP, FN)
                tmp_df = pd.DataFrame.from_records([{'attribute': attribute, 'value': val, 
                                                    **metrics_results, 'IoU_mean': iou_mean, 'IoU_median': iou_median}])
                metrics_objects_df = pd.concat([metrics_objects_df, tmp_df])   
        
        feature_path = os.path.join(output_dir, 'per_attribute_metrics.csv')
        metrics_objects_df.to_csv(feature_path, index=False)
        written_files[feature_path] = ''

    # Sump-up results and save files
    TP = metrics_df.loc[0, 'TP']
    FP = metrics_df.loc[0, 'FP'] 
    FN = metrics_df.loc[0, 'FN']
    precision = metrics_df.loc[0, 'precision']
    recall = metrics_df.loc[0, 'recall']
    f1 = metrics_df.loc[0, 'f1']
    iou_mean = metrics_df.loc[0, 'IoU_mean']
    iou_median = metrics_df.loc[0, 'IoU_median']

    print()
    logger.info(f"TP = {TP}, FP = {FP}, FN = {FN}")
    logger.info(f"TP+FN = {TP+FN}, TP+FP = {TP+FP}")
    logger.info(f"precision = {precision:.2f}, recall = {recall:.2f}, f1 = {f1:.2f}")
    logger.info(f"Mean IoU for all EGIDs = {iou_mean:.2f}")
    logger.info(f"Median IoU for all EGIDs = {iou_median:.2f}")
    print()

    # Check if detection or labels have been lost in the process
    nbr_tagged_labels = TP + FN
    labels_diff = nbr_labels - nbr_tagged_labels
    filename = os.path.join(output_dir, 'problematic_objects.gpkg')
    if os.path.exists(filename):
        os.remove(filename)
    if (labels_diff != 0) and (method != 'fusion'):
        logger.error(f'There are {nbr_labels} labels in input and {nbr_tagged_labels} labels in output.')
        logger.info(f'The list of the problematic labels is exported to {filename}.')

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


    if visualisation and additional_metrics:
        logger.info('Save some figures...')
        
        xlabel_dict = {'EGID': '', 'roof_type': '', 'roof_inclination': '', 'building_type': '',
                    'object_class':'', 'area': r'Object area ($m^2$)', 
                    'nearest_distance_centroid': r'Object distance (m)'} 

        # _ = figures.plot_histo(output_dir, labels_gdf, detections_gdf, attribute=OBJECT_PARAMETERS, xlabel=xlabel_dict)
        for attr in metrics_objects_df.attribute.unique():
            if attr in xlabel_dict.keys():
                _ = figures.plot_groups(output_dir, metrics_objects_df, attribute=attr, xlabel=xlabel_dict[attr])
                _ = figures.plot_stacked_grouped_percent(output_dir, metrics_objects_df, attribute=attr, xlabel=xlabel_dict[attr])
                _ = figures.plot_metrics(output_dir, metrics_objects_df, attribute=attr, xlabel=xlabel_dict[attr])

            
    return f1, iou_median, written_files

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
    ROOFS = cfg['roofs']

    METHOD = cfg['method']
    THRESHOLD = cfg['threshold']

    ADDITIONAL_METRICS = cfg['additional_metrics'] if 'additional_metrics' in cfg.keys() else False
    OBJECT_PARAMETERS = cfg['object_attributes']['parameters']
    AREA_RANGES = cfg['object_attributes']['area_ranges']
    DISTANCE_RANGES = cfg['object_attributes']['distance_ranges']
    VISU = cfg['visualisation'] if 'visualisation' in cfg.keys() else False

    RANGES = [AREA_RANGES] + [DISTANCE_RANGES]

    f1, diff_in_labels, written_files = main(WORKING_DIR, OUTPUT_DIR, LABELS, DETECTIONS, EGIDS, ROOFS,
                                             method=METHOD, threshold=THRESHOLD,
                                             object_parameters=OBJECT_PARAMETERS, ranges=RANGES,
                                             additional_metrics=ADDITIONAL_METRICS, visualisation=VISU)

    logger.success("The following files were written. Let's check them out!")
    for path in written_files.keys():
        logger.success(f'  file: {path}{"" if written_files[path] == "" else f", layer: {written_files[path]}"}')
    if VISU and ADDITIONAL_METRICS:
        logger.success('Some figures were also written.')

    # Stop chronometer  
    toc = time.time()
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()