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

def main(WORKING_DIR, OUTPUT_DIR, LABELS, DETECTIONS, EGIDS, ROOFS, method='one-to-one', threshold=0.1, object_parameters=[], ranges=[], buffer=0.1,
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
        buffer (float): buffer to avoid the intersection of touching shapes.
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
    output_dir = misc.ensure_dir_exists(os.path.join(OUTPUT_DIR, method))
    threshold_str = str(threshold).replace('.', 'dot')

    written_files = {}

    logger.info("Get input data")

    egids, roofs_gdf, labels_gdf, detections_gdf = misc.get_inputs_for_assessment(EGIDS, ROOFS, LABELS, DETECTIONS)

    if (len(object_parameters) > 0) and additional_metrics:
  
        ranges_dict = {object_parameters[i]: ranges[i] for i in range(len(object_parameters))}

        ## Get nearest distance between polygons
        labels_gdf = misc.nearest_distance(labels_gdf, roofs_gdf, join_key='EGID', parameter='nearest_distance_centroid', lsuffix='_label', rsuffix='_roof')
        labels_gdf = misc.nearest_distance(labels_gdf, roofs_gdf, join_key='EGID', parameter='nearest_distance_border', lsuffix='_label', rsuffix='_roof')
        
        detections_gdf = misc.nearest_distance(detections_gdf, roofs_gdf, join_key='EGID', parameter='nearest_distance_centroid', lsuffix='_detection', rsuffix='_roof')
        detections_gdf = misc.nearest_distance(detections_gdf, roofs_gdf, join_key='EGID', parameter='nearest_distance_border', lsuffix='_detection', rsuffix='_roof')

        ## Get roundness of polygons
        labels_gdf = misc.roundness(labels_gdf)   
        detections_gdf = misc.roundness(detections_gdf)

    # Detections count
    logger.info(f"Method used for detections counting:")
    methods_list =  ['one-to-one', 'one-to-many', 'charges', 'fusion']
    if method in methods_list:
        logger.info(f'    {method} method')
    else:
        logger.warning('    Unknown method, default method = one-to-one.')

    metrics_egid_df = pd.DataFrame()
    metrics_objects_df = pd.DataFrame()

    logger.info("Geohash the labels and detections...")
    GT_PREFIX= 'gt_'
    labels_gdf = misc.add_geohash(labels_gdf, prefix=GT_PREFIX)
    labels_gdf = misc.drop_duplicates(labels_gdf, subset='geohash')
    nbr_labels = labels_gdf.shape[0]
    
    DETS_PREFIX = "dt_"
    detections_gdf = misc.add_geohash(detections_gdf, prefix=DETS_PREFIX)
    detections_gdf = misc.drop_duplicates(detections_gdf, subset='geohash')

    if detections_gdf.shape[0] == 0:
        logger.error('No detection is available, returning 0 as f1 score and IoU median.')
        metrics_df = pd.DataFrame({'attribute': ['EGID'], 'f1': [0], 'IoU_median': [0]})

        return metrics_df, []
    
    elif method == 'charges' or method == 'fusion':

        logger.info(f"Metrics computation:")
        logger.info(f"     - Compute TP, FP and FN")

        tagged_gt_gdf, tagged_dets_gdf = metrics.tag(gt=labels_gdf, dets=detections_gdf,
                                                    threshold=threshold, method=method, buffer=buffer, 
                                                    gt_prefix=GT_PREFIX, dets_prefix=DETS_PREFIX, group_attribute='EGID')
        feature_path = os.path.join(output_dir, 'tags.gpkg')
        
        if method == 'fusion':
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
        metrics_df = pd.DataFrame.from_records([{'attribute': 'EGID', 'value': 'ALL', **metrics_results}])

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
                filter_gt_gdf = tagged_gt_gdf[tagged_gt_gdf['descr'] == object_class].copy()
                    
                TP = float(filter_gt_gdf['TP_charge'].sum())
                FN = float(filter_gt_gdf['FN_charge'].sum())
                FP = 0

                metrics_results = metrics.get_metrics(TP, FP, FN)
                tmp_df = pd.DataFrame.from_records([{'attribute': 'object_class', 'value': object_class, **metrics_results}])
                metrics_objects_df = pd.concat([metrics_objects_df, tmp_df])

            if (len(object_class) > 0) and isinstance(roofs_gdf, gpd.GeoDataFrame):
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

        tp_gdf, fp_gdf, fn_gdf = metrics.get_fractional_sets(detections_gdf[['detection_id', 'geometry']], labels_gdf, method=method, iou_threshold=threshold)
        TP = len(tp_gdf)
        FP = len(fp_gdf)
        FN = len(fn_gdf)

        # Compute metrics
        logger.info("    - Global metrics")
        metrics_results = metrics.get_metrics(TP, FP, FN)
        metrics_df = pd.DataFrame.from_records([{'attribute': 'EGID', 'value': 'ALL', **metrics_results}])

        if method == 'one-to-many':
            tp_with_duplicates = tp_gdf.copy()
            dissolved_tp_gdf = tp_with_duplicates.dissolve(by=['detection_id'], as_index=False)

            geom1 = dissolved_tp_gdf.geometry.values.tolist()
            geom2 = dissolved_tp_gdf['label_geometry'].values.tolist()
            iou = []
            for (i, ii) in zip(geom1, geom2):
                iou.append(metrics.intersection_over_union(i, ii))
            dissolved_tp_gdf['IoU'] = iou

            tp_gdf = dissolved_tp_gdf.copy()

            logger.info(f'{tp_with_duplicates.shape[0] - tp_gdf.shape[0]} labels are under a shared detections with at least one other label.')

        # Set the final dataframe with tagged detections
        tagged_dets_gdf = pd.concat([tp_gdf, fp_gdf, fn_gdf])

        tagged_dets_gdf.drop(['index_right', 'occupation_left', 'occupation_right', 'label_geometry', 'detection_geometry', 'ID_DET', 'area_DET', 'EGID_GT'], 
                             axis=1, inplace=True, errors='ignore')
        tagged_dets_gdf = tagged_dets_gdf.round({'IoU': 2, 'detection_area': 4})
        tagged_dets_gdf.reset_index(drop=True, inplace=True)
        
        layer_name = 'tagged_detections_' + method + '_thd_' + threshold_str
        feature_path = os.path.join(output_dir, 'tagged_detections.gpkg')
        tagged_dets_gdf.to_file(feature_path, layer=layer_name, index=False)
        written_files[feature_path] = layer_name

        if additional_metrics:
            logger.info("    - Metrics per egid")
            for egid in tqdm(sorted(labels_gdf.EGID.unique()), desc='Per-EGID metrics'):
                tp_gdf, fp_gdf, fn_gdf = metrics.get_fractional_sets(detections_gdf[detections_gdf.EGID == egid], labels_gdf[labels_gdf.EGID == egid], method=method)
                TP = len(tp_gdf)
                FP = len(fp_gdf)
                FN = len(fn_gdf)
                metrics_results = metrics.get_metrics(TP, FP, FN)
                tmp_df = pd.DataFrame.from_records([{'EGID': egid, **metrics_results}])
                metrics_egid_df = pd.concat([metrics_egid_df, tmp_df])

            logger.info("    - Metrics per object class")
            for object_class in sorted(labels_gdf.descr.unique()):
                
                filter_gt_gdf = tagged_dets_gdf[tagged_dets_gdf['descr'] == object_class].copy()
                
                TP = len(filter_gt_gdf[filter_gt_gdf['tag'] == 'TP'])
                FN = len(filter_gt_gdf[filter_gt_gdf['tag'] == 'FN'])
                FP = len(filter_gt_gdf[filter_gt_gdf['tag'] == 'FP'])

                metrics_results = metrics.get_metrics(TP, FP, FN)
                tmp_df = pd.DataFrame.from_records([{'attribute': 'object_class', 'value': object_class, **metrics_results}])
                metrics_objects_df = pd.concat([metrics_objects_df, tmp_df])

            if (len(object_class) > 0) and isinstance(roofs_gdf, gpd.GeoDataFrame):
                logger.info("    - Metrics per object attributes")
                for parameter in object_parameters:
                    param_ranges = ranges_dict[parameter] 
                    for val in param_ranges:
                        filter_dets_gdf = tagged_dets_gdf[(tagged_dets_gdf[parameter] >= val[0]) & (tagged_dets_gdf[parameter] <= val[1])].copy()
                            
                        TP = float(filter_dets_gdf.loc[filter_dets_gdf.tag == 'TP'].shape[0])
                        FN = float(filter_dets_gdf.loc[filter_dets_gdf.tag == 'FN'].shape[0])
                        FP = 0

                        metrics_results = metrics.get_metrics(TP, FP, FN)
                        tmp_df = pd.DataFrame.from_records([{'attribute': parameter, 'value': str(val).replace(",", " -"), **metrics_results}])
                        metrics_objects_df = pd.concat([metrics_objects_df, tmp_df])

    # Compute Jaccard index by EGID
    logger.info(f"    - Compute mean Jaccard index")

    labels_by_attr_gdf = metrics.get_jaccard_index(labels_gdf, detections_gdf)

    if 'EGID' not in metrics_egid_df.columns:
        metrics_egid_df['EGID'] = labels_by_attr_gdf.EGID

    metrics_egid_df['IoU_EGID'] = [
        labels_by_attr_gdf.loc[labels_by_attr_gdf.EGID == egid, 'IoU_EGID'].iloc[0]
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
                metrics_df = pd.concat([metrics_df, tmp_df])

        metrics_df = pd.concat([metrics_df, metrics_objects_df]).reset_index(drop=True)

    # Sump-up results and save files
    TP = metrics_df.loc[0, 'TP']
    FP = metrics_df.loc[0, 'FP'] 
    FN = metrics_df.loc[0, 'FN']
    precision = metrics_df.loc[0, 'precision']
    recall = metrics_df.loc[0, 'recall']
    f1 = metrics_df.loc[0, 'f1']
    iou_mean = metrics_df.loc[0, 'IoU_mean']
    iou_median = metrics_df.loc[0, 'IoU_median']

    written_files[feature_path] = ''
    feature_path = os.path.join(output_dir, 'metrics.csv')
    metrics_df.to_csv(feature_path, sep=',', index=False, float_format='%.4f')

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
    if (labels_diff != 0) and (method != 'fusion') and (method != 'charges'):
        logger.warning(f'There are {int(nbr_labels)} labels in input and {int(nbr_tagged_labels)} labels in output.')
        logger.info(f'The list of the problematic labels is exported to {filename}.')

        if labels_diff > 0:
            tagged_labels = tp_gdf['label_id'].unique().tolist() + fn_gdf['label_id'].unique().tolist()

            untagged_labels_gdf = labels_gdf[~labels_gdf['label_id'].isin(tagged_labels)]
            untagged_labels_gdf.drop(columns=['label_geometry'], inplace=True)

            layer_name = 'missing_label_tags'
            untagged_labels_gdf.to_file(filename, layer=layer_name, index=False)

        elif labels_diff < 0:
            all_tagged_labels_gdf = pd.concat([tp_gdf, fn_gdf])

            duplicated_label_id = all_tagged_labels_gdf.loc[all_tagged_labels_gdf.duplicated(subset=['label_id']), 'label_id'].unique().tolist()
            duplicated_labels = all_tagged_labels_gdf[all_tagged_labels_gdf['label_id'].isin(duplicated_label_id)]
            duplicated_labels.drop(columns=['label_geometry', 'detection_geometry', 'index_right', 'EGID', 'occupation_left', 'occupation_right'], inplace=True)

            layer_name = 'duplicated_label_tags'
            duplicated_labels.to_file(filename, layer=layer_name, index=False)
            
        written_files[filename] = layer_name


    if visualisation and additional_metrics:
        logger.info('Save some figures...')

        xlabel_dict = {'EGID': '', 'building_type': '', 'roof_inclination': '',
                    'object_class':'', 'area': r'Object area ($m^2$)', 
                    'nearest_distance_border': r'Object distance (m)', 'roundness': r'Roundness'} 

        # _ = figures.plot_histo(output_dir, labels_gdf, detections_gdf, attribute=OBJECT_PARAMETERS, xlabel=xlabel_dict)
        for attr in metrics_df.attribute.unique():
            if attr in xlabel_dict.keys():
                _ = figures.plot_groups(output_dir, metrics_df, attribute=attr, xlabel=xlabel_dict[attr])
                # _ = figures.plot_stacked_grouped(output_dir, metrics_df, attribute=attr, xlabel=xlabel_dict[attr])
                _ = figures.plot_stacked_grouped_percent(output_dir, metrics_df, attribute=attr, xlabel=xlabel_dict[attr])
                _ = figures.plot_metrics(output_dir, metrics_df, attribute=attr, xlabel=xlabel_dict[attr])

    return metrics_df, written_files

# ------------------------------------------

if __name__ == "__main__":
    
    # Start chronometer
    tic = time.time()
    logger.info("Result assessment")
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

    DETECTIONS = cfg['detections']
    LABELS = cfg['ground_truth']
    EGIDS = cfg['egids']
    ROOFS = cfg['roofs']

    METHOD = cfg['method']
    THRESHOLD = cfg['threshold']
    BUFFER = cfg['buffer']  if 'buffer' in cfg.keys() else 0.01
    ADDITIONAL_METRICS = cfg['additional_metrics'] if 'additional_metrics' in cfg.keys() else False
    OBJECT_PARAMETERS = cfg['object_attributes']['parameters']
    AREA_RANGES = cfg['object_attributes']['area_ranges']
    DISTANCE_RANGES = cfg['object_attributes']['distance_ranges']
    ROUND_RANGES = cfg['object_attributes']['round_ranges']
    VISU = cfg['visualisation'] if 'visualisation' in cfg.keys() else False

    RANGES = [AREA_RANGES] + [DISTANCE_RANGES] + [ROUND_RANGES] 

    metrics_df, written_files = main(WORKING_DIR, OUTPUT_DIR, LABELS, DETECTIONS, EGIDS, ROOFS,
                                            method=METHOD, threshold=THRESHOLD, 
                                            object_parameters=OBJECT_PARAMETERS, ranges=RANGES, buffer=BUFFER,
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