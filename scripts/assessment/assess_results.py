#!/bin/python
# -*- coding: utf-8 -*-

#  proj-rooftops


import os
import sys
import time
import argparse
from loguru import logger
from yaml import load, FullLoader

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
import numpy as np
import pandas as pd
import geopandas as gpd

sys.path.insert(1, 'scripts')
import functions.fct_misc as misc
import functions.fct_metrics as metrics

logger = misc.format_logger(logger)


# Functions --------------------------


def plot_histo(dir_plots, df1, df2, attribute, xlabel):

    fig = plt.figure(figsize =(12, 8))

    df1 = df1.fillna(0)
    df2 = df2.fillna(0)
    
    for i in attribute:
        bins=np.histogram(np.hstack((df1[i],df2[i])), bins=10)[1]
        df1[i].plot.hist(bins=bins, alpha=0.5, label='GT')
        df2[i].plot.hist(bins=bins, alpha=0.5, label='Detections')

        plt.xlabel(xlabel[i] , fontweight='bold')

        plt.legend(frameon=False)  
        plt.title(f'Object distribution')

        plt.tight_layout() 
        plot_path = os.path.join(dir_plots, f'histo_{i}.png')  
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close(fig)


def plot_surface(dir_plots, df, attribute, xlabel):

    fig, ax = plt.subplots(1, 2, sharey= True, figsize=(16,8))

    color_list = ['limegreen', 'tomato']  

    df = df[df['attribute'] == attribute]  

    df.plot(ax=ax[0], x='value', y=['free_surface_label', 'occupied_surface_label',], kind='bar', stacked=True, rot=0, color = color_list)
    df.plot(ax=ax[1], x='value', y=['free_surface_det', 'occupied_surface_det',], kind='bar', stacked=True, rot=0, color = color_list)
    for b, c in zip(ax[0].containers, ax[1].containers):
        labels1 = [f'{"{0:.1f}".format(a)}' if a > 0 else "" for a in b.datavalues]
        labels2 = [f'{"{0:.1f}".format(a)}' if a > 0 else "" for a in c.datavalues]
        ax[0].bar_label(b, label_type='center', color = "black", labels=labels1, fontsize=10)
        ax[1].bar_label(c, label_type='center', color = "black", labels=labels2, fontsize=10)

    if attribute == 'object_class':
        plt.xticks(rotation=40, ha='right')  
    ax[0].set_xlabel(xlabel, fontweight='bold')
    ax[0].set_ylabel('Surface ($m^2$)', fontweight='bold')
    ax[1].set_xlabel(xlabel, fontweight='bold')

    ax[0].legend('', frameon=False)  
    ax[1].legend(['Free', 'Occupied'], bbox_to_anchor=(1.05, 1.0), loc='upper left', frameon=False)    
    ax[0].set_title(f'GT surfaces by {attribute.replace("_", " ")}')
    ax[1].set_title(f'Detection surfaces by {attribute.replace("_", " ")}')

    plt.tight_layout() 
    plot_path = os.path.join(dir_plots, f'surface_{attribute}.png')  
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)


def plot_stacked_grouped(dir_plots, df, attribute, xlabel):

    fig, ax = plt.subplots(figsize=(12,8))

    color_list = ['limegreen', 'orange', 'tomato']  
    counts_list = ['TP', 'FP', 'FN']    

    df = df[df['attribute'] == attribute]  
    df = df[['value', 'TP', 'FP', 'FN']].set_index('value')
    
    df[counts_list].plot(ax=ax, kind='bar', stacked=True, color=color_list, rot=0)

    for c in ax.containers:
        labels = [f'{"{0:.1f}".format(a)}' if a > 0 else "" for a in c.datavalues]
        ax.bar_label(c, label_type='center', color = "white", labels=labels, fontsize=10)

    if attribute == 'object_class':
        plt.xticks(rotation=40, ha='right')
    plt.xlabel(xlabel, fontweight='bold')

    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', frameon=False)    
    plt.title(f'Counts by {attribute.replace("_", " ")}')

    plt.tight_layout() 
    plot_path = os.path.join(dir_plots, f'counts_{attribute}.png')  
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)


def plot_stacked_grouped_percent(dir_plots, df, attribute, xlabel):

    fig, ax = plt.subplots(figsize=(12,8))

    color_list = ['limegreen', 'orange', 'tomato']  
    counts_list = ['TP', 'FP', 'FN']    

    df = df[df['attribute'] == attribute]  
    df = df[['value', 'TP', 'FP', 'FN']].set_index('value')
    df['sum'] = df.sum(axis=1)

    for count in counts_list:
        df[count] =  df[count] / df['sum']
    
    df[counts_list].plot(ax=ax, kind='bar', stacked=True, color=color_list, rot=0, width = 0.5)

    for c in ax.containers:
        labels = [f'{"{0:.1%}".format(a)}' if a > 0 else "" for a in c.datavalues]
        ax.bar_label(c, label_type='center', color = "white", labels=labels, fontsize=10)

    plt.ylim(0, 1)
    plt.gca().set_yticks(plt.gca().get_yticks().tolist())
    plt.gca().set_yticklabels([f'{"{0:.0%}".format(x)}' for x in plt.gca().get_yticks()]) 
    if attribute == 'object_class':
        plt.xticks(rotation=40, ha='right')
    plt.xlabel(xlabel, fontweight='bold')

    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', frameon=False)    
    plt.title(f'Counts by {attribute.replace("_", " ")}')

    plt.tight_layout() 
    plot_path = os.path.join(dir_plots, f'counts_{attribute}_percent.png')  
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)


def plot_metrics(dir_plots, df, attribute, xlabel):

    fig, ax = plt.subplots(figsize=(8,6))

    metrics_list = ['precision', 'recall', 'f1', 'IoU']    

    df = df[df['attribute'] == attribute] 
    
    for metric in metrics_list:
        if not df[metric].isnull().values.any():
            plt.scatter(df['value'], df[metric], label=metric, s=150)

    plt.ylim(-0.05, 1.05)
    if attribute == 'object_class':
        plt.xticks(rotation=40, ha='right')
    plt.xlabel(xlabel, fontweight='bold')

    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', frameon=False)
    plt.title(f'Metrics by {attribute.replace("_", " ")}')
 
    plt.tight_layout() 
    plot_path = os.path.join(dir_plots, f'metrics_{attribute}.png')  
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)


def main(WORKING_DIR, OUTPUT_DIR, LABELS, DETECTIONS, ROOFS, EGIDS, METHOD, THD, AREA_THD_FACTOR, OBJECT_PARAMETERS, RANGES):
    """Assess the results by calculating the precision, recall and f1-score.

    Args:
        WORKING_DIR (path): working directory
        OUTPUT_DIR (path): output directory
        DETECTIONS (path): file of the detections
        LABELS (path): file of the ground truth
        ROOFS (path): file of the roof border and main elements
        EGIDS (list): EGIDs of interest
        METHOD (string): method to use for the assessment of the results, either one-to-one, one-to-many or many-to-many.
        THD (float): surface fraction threshold under which the detection is not considered to overlap a label
        AREA_THD_FACTOR (float): factor apply to the minimum label area to define the area threshold under which detections are discarded
        OBJECT_PARAMETERS (list): list of object parameter to be processed ('area', 'nearest_distance_border', 'nearest_distance_centroid')
        RANGES (list): list of list of the bins to process by OBJECT_PARAMETERS

    Returns:
        df: metrics computed for different attribute.
        gdf: labels_diff, missing labels (lost during the process)
    """

    os.chdir(WORKING_DIR)

    # Create an output directory in case it doesn't exist
    output_dir = os.path.join(OUTPUT_DIR, METHOD)
    misc.ensure_dir_exists(output_dir)
    threshold_str = str(THD).replace('.', 'dot')
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
    desired_file_path = os.path.join(ROOFS_DIR, ROOFS_NAME[:-4] + "_" + attribute + ".shp")
    
    roofs = misc.dissolve_by_attribute(desired_file_path, original_file_path, name=ROOFS_NAME[:-4], attribute=attribute)
    roofs_gdf = roofs[roofs.EGID.isin(egids.EGID.to_numpy())]
    roofs_gdf['EGID'] = roofs_gdf['EGID'].astype(int)
    roofs_gdf['area'] = round(roofs_gdf['geometry'].area, 4)

    # Get labels shapefile
    logger.info("- GT")
    labels_gdf = gpd.read_file(LABELS)

    labels_gdf['type'] = labels_gdf['type'].astype(int)
    labels_gdf['EGID'] = labels_gdf['EGID'].astype(int)
    labels_gdf = labels_gdf.rename(columns={'type':'obj_class'})
    labels_gdf.loc[labels_gdf['obj_class'] == 4, 'descr'] = 'Aero'

    # Type 12 corresponds to free surfaces, other classes are ojects
    logger.info("  Filter objects and EGID")
    labels_gdf = labels_gdf[(labels_gdf['obj_class'] != 12) & (labels_gdf.EGID.isin(egids.EGID.to_numpy()))]
    labels_gdf['label_id'] = labels_gdf.index
    
    labels_gdf = labels_gdf.explode(index_parts=True)
    # labels_gdf = labels_gdf[labels_gdf['geometry'].geom_type.values == 'Polygon']

    # Create geohash to GT shapes
    logger.info("  Geohashing GT")
    GT_PREFIX= 'gt_'
    labels_gdf = misc.add_geohash(labels_gdf, prefix=GT_PREFIX)
    labels_gdf = misc.drop_duplicates(labels_gdf, subset='geohash')

    nbr_labels = labels_gdf.shape[0]
    logger.info(f"  Read labels file: {nbr_labels} shapes")

    # Add geometry attributes
    logger.info("  Add geometry attributes to GT")

    ## Area
    labels_gdf['area'] = round(labels_gdf.area, 4)

    ## Nearest distance between polygons
    labels_gdf = misc.nearest_distance(labels_gdf, roofs_gdf, join_key='EGID', parameter='nearest_distance_centroid', lsuffix='_label', rsuffix='_roof')
    labels_gdf = misc.nearest_distance(labels_gdf, roofs_gdf, join_key='EGID', parameter='nearest_distance_border', lsuffix='_label', rsuffix='_roof')

    # Save new labels file     
    labels_gdf = labels_gdf.drop(columns=['fid'], axis=1)

    # feature_path = os.path.join(output_dir, 'labels.gpkg')
    # labels_gdf.to_file(feature_path, index=False)

    # Get detections shapefile
    logger.info("- Detections")
    detections_gdf = gpd.read_file(DETECTIONS)

    # detections_gdf = detections_gdf[detections_gdf['geometry'].geom_type.values == 'Polygon']
    detections_gdf = detections_gdf.explode(index_parts=True)

    logger.info("  Geohashing detections")
    DETS_PREFIX = "dt_"
    detections_gdf = misc.add_geohash(detections_gdf, prefix=DETS_PREFIX)
    detections_gdf = misc.drop_duplicates(detections_gdf, subset='geohash')

    detections_gdf['EGID'] = detections_gdf['EGID'].astype(int)
   
    if 'value' in detections_gdf.columns:
        detections_gdf.rename(columns={'value': 'detection_id'}, inplace=True)
    detections_gdf['detection_id'] = detections_gdf['detection_id'].astype(int)

    logger.info(f"Read detection file: {len(detections_gdf)} shapes")

    # Filter detections by area
    area_threshold = AREA_THD_FACTOR * np.min(labels_gdf['area'])
    detections_gdf = detections_gdf[detections_gdf.area >= area_threshold]

    # Add geometry attributes
    logger.info("  Add geometry attributes to detection")

    ## Nearest distance between polygons
    detections_gdf = misc.nearest_distance(detections_gdf, roofs_gdf, join_key='EGID', parameter='nearest_distance_centroid', lsuffix='_detection', rsuffix='_roof')
    detections_gdf = misc.nearest_distance(detections_gdf, roofs_gdf, join_key='EGID', parameter='nearest_distance_border', lsuffix='_detection', rsuffix='_roof')


    # Detections count
    logger.info(f"Method used for detections counting")
    methods_list =  ['one-to-one', 'one-to-many', 'charges', 'fusion']
    if METHOD in methods_list:
        logger.info(f'Using the {METHOD} method')
    else:
        logger.warning('Unknown method, default method = one-to-one')

    logger.info(f"Metrics computation:")
    logger.info(f"- Count TP, FP and FN")

    metrics_df = pd.DataFrame()
    metrics_egid_df = pd.DataFrame()
    metrics_objects_df = pd.DataFrame()


    if METHOD == 'charges' or METHOD == 'fusion':

        tagged_gt_gdf, tagged_dets_gdf = metrics.tag(gt=labels_gdf, dets=detections_gdf, buffer=-0.05, gt_prefix=GT_PREFIX, dets_prefix=DETS_PREFIX, threshold=THD, method=METHOD)

        if METHOD=='fusion':
            unique_dets_gdf = tagged_dets_gdf[tagged_dets_gdf['group_id'].isna()] 
            dissolve_dets_gdf = tagged_dets_gdf.dissolve(by='group_id', as_index=False)
            tagged_dets_gdf = pd.concat([unique_dets_gdf, dissolve_dets_gdf]).reset_index(drop=True)
    
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

        # Get output files 
        feature_path = os.path.join(output_dir, 'tags.gpkg')
        layer_name = 'tagged_labels_' + METHOD + '_thd_' + threshold_str
        tagged_gt_gdf.astype({'TP_charge': 'str', 'FN_charge': 'str'}).to_file(feature_path, layer=layer_name, driver='GPKG')
        written_files[feature_path] = layer_name
        layer_name = 'tagged_detections_' + METHOD + '_thd_' + threshold_str
        tagged_dets_gdf.astype({'TP_charge': 'str', 'FP_charge': 'str'}).to_file(feature_path, layer=layer_name, driver='GPKG')
        written_files[feature_path] = layer_name
        if METHOD != 'charges':
            layer_name = 'tagged_final_' + METHOD + '_thd_' + threshold_str
            tagged_final_gdf.astype({'TP_charge': 'str', 'FP_charge': 'str', 'FN_charge': 'str'}).to_file(feature_path, layer=layer_name, driver='GPKG')
            written_files[feature_path] = layer_name


        # Compute metrics
        logger.info("- Global metrics")
        TP, FP, FN = metrics.get_count(tagged_gt_gdf, tagged_dets_gdf)
        metrics_results = metrics.get_metrics(TP, FP, FN)
        tmp_df = pd.DataFrame.from_records([{'attribute': 'EGID', 'value': 'ALL', **metrics_results}])
        metrics_df = pd.concat([metrics_df, tmp_df])
 
        logger.info("- Metrics per egid")
        for egid in sorted(labels_gdf.EGID.unique()):
            TP, FP, FN = metrics.get_count(
                tagged_gt = tagged_gt_gdf[tagged_gt_gdf.EGID == egid],
                tagged_dets = tagged_dets_gdf[tagged_dets_gdf.EGID == egid],
            )
            metrics_results = metrics.get_metrics(TP, FP, FN)
            tmp_df = pd.DataFrame.from_records([{'EGID': egid, **metrics_results}])
            metrics_egid_df = pd.concat([metrics_egid_df, tmp_df])

        ranges_dic = {OBJECT_PARAMETERS[i]: RANGES[i] for i in range(len(OBJECT_PARAMETERS))}

        logger.info("- Metrics per object's class")
        for object_class in sorted(labels_gdf.descr.unique()):
            filter_gt_gdf = tagged_gt_gdf[tagged_gt_gdf['descr']==object_class]
                
            TP = float(filter_gt_gdf['TP_charge'].sum())
            FN = float(filter_gt_gdf['FN_charge'].sum())
            FP = 0

            metrics_results = metrics.get_metrics(TP, FP, FN)
            rem_list = ['FP', 'TPplusFP', 'precision', 'f1']
            [metrics_results.pop(key) for key in rem_list]
            tmp_df = pd.DataFrame.from_records([{'attribute': 'object_class', 'value': object_class, **metrics_results}])
            metrics_objects_df = pd.concat([metrics_objects_df, tmp_df])
  
        logger.info("- Metrics per object attributes")
        for parameter in OBJECT_PARAMETERS:
            ranges = ranges_dic[parameter] 
            for val in ranges:
                filter_gt_gdf = tagged_gt_gdf[(tagged_gt_gdf[parameter] >= val[0]) & (tagged_gt_gdf[parameter] <= val[1])]
                filter_dets_gdf = tagged_dets_gdf[(tagged_dets_gdf[parameter] >= val[0]) & (tagged_dets_gdf[parameter] <= val[1])]
                
                TP = float(filter_gt_gdf['TP_charge'].sum())
                FP = float(filter_dets_gdf['FP_charge'].sum()) 
                FN = float(filter_gt_gdf['FN_charge'].sum())

                metrics_results = metrics.get_metrics(TP, FP, FN)
                tmp_df = pd.DataFrame.from_records([{'attribute': parameter, 'value': str(val).replace(",", " -"), **metrics_results}])
                metrics_objects_df = pd.concat([metrics_objects_df, tmp_df])

    else:
        # Count 

        tp_gdf, fp_gdf, fn_gdf = metrics.get_fractional_sets(detections_gdf, labels_gdf, method=METHOD)
        TP = len(tp_gdf)
        FP = len(fp_gdf)
        FN = len(fn_gdf)

        # Compute metrics
        logger.info("- Global metrics")
        metrics_results = metrics.get_metrics(TP, FP, FN)
        tmp_df = pd.DataFrame.from_records([{'attribute': 'EGID', 'value': 'ALL', **metrics_results}])
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

        tagged_dets_gdf.drop(['label_geometry', 'detection_geometry', 'detection_id'], axis=1, inplace=True)
        tagged_dets_gdf = tagged_dets_gdf.round({'IOU': 4})
        tagged_dets_gdf = tagged_dets_gdf.round({'detection_area': 4})
        tagged_dets_gdf.reset_index(drop=True, inplace=True)
        tagged_dets_gdf['fid'] = tagged_dets_gdf.index

        layer_name = 'tagged_detections_' + METHOD + '_thd_' + threshold_str
        feature_path = os.path.join(output_dir, 'detection_tags.gpkg')
        tagged_dets_gdf.to_file(feature_path, layer=layer_name, index=False)

        written_files[feature_path] = layer_name


        logger.info("- Metrics per egid")
        for egid in sorted(labels_gdf.EGID.unique()):
            tp_gdf, fp_gdf, fn_gdf = metrics.get_fractional_sets(detections_gdf, labels_gdf, method=METHOD)
            TP = len(tp_gdf)
            FP = len(fp_gdf)
            FN = len(fn_gdf)
            metrics_results = metrics.get_metrics(TP, FP, FN)
            tmp_df = pd.DataFrame.from_records([{'EGID': egid, **metrics_results}])
            metrics_egid_df = pd.concat([metrics_egid_df, tmp_df])

        ranges_dic = {OBJECT_PARAMETERS[i]: RANGES[i] for i in range(len(OBJECT_PARAMETERS))}

        logger.info("- Metrics per object's class")
        for object_class in sorted(labels_gdf.descr.unique()):
            
            filter_gt_gdf = tagged_dets_gdf[tagged_dets_gdf['descr']==object_class]
            
            TP = len(filter_gt_gdf[filter_gt_gdf['tag'] == 'TP'])
            FN = len(filter_gt_gdf[filter_gt_gdf['tag'] == 'FN'])
            FP = len(filter_gt_gdf[filter_gt_gdf['tag'] == 'FP'])

            metrics_results = metrics.get_metrics(TP, FP, FN)
            rem_list = ['FP', 'TPplusFP', 'precision', 'f1']
            [metrics_results.pop(key) for key in rem_list]
            tmp_df = pd.DataFrame.from_records([{'attribute': 'object_class', 'value': object_class, **metrics_results}])
            metrics_objects_df = pd.concat([metrics_objects_df, tmp_df])

    # Compute Jaccard index and free surface by EGID
    logger.info(f"- Compute mean Jaccard index")
    
    keys = ['IoU', 'occupied_surface_label', 'occupied_surface_det', 'free_surface_label', 'free_surface_det']
    for key in keys:
        metrics_egid_df[key] = 0
        metrics_df[key] = 0
    for egid in sorted(labels_gdf.EGID.unique()):
        labels_egid_gdf, detections_egid_gdf = metrics.get_jaccard_index(
            labels_gdf[labels_gdf.EGID == egid], 
            detections_gdf[detections_gdf.EGID == egid], attribute='EGID'
        )

        labels_free_gdf, detections_free_gdf = metrics.get_free_surface(
            labels_gdf[labels_gdf.EGID == egid], 
            detections_gdf[detections_gdf.EGID == egid],
            roofs_gdf[roofs_gdf.EGID == egid], attribute='EGID'
        )
        iou_average = detections_egid_gdf['IOU_EGID'].mean()
        metrics_egid_df['IoU'] = np.where(metrics_egid_df['EGID'] == egid, iou_average, metrics_egid_df['IoU'])
        
        occupied_surface_label = labels_free_gdf['occupied_surface'].sum()
        metrics_egid_df['occupied_surface_label'] = np.where(metrics_egid_df['EGID'] == egid, occupied_surface_label, metrics_egid_df['occupied_surface_label'])
        occupied_surface_det = detections_free_gdf['occupied_surface'].sum()
        metrics_egid_df['occupied_surface_det'] = np.where(metrics_egid_df['EGID'] == egid, occupied_surface_det, metrics_egid_df['occupied_surface_det'])
        free_surface_label = labels_free_gdf['free_surface'].sum()
        metrics_egid_df['free_surface_label'] = np.where(metrics_egid_df['EGID'] == egid, free_surface_label, metrics_egid_df['free_surface_label'])
        free_surface_det = detections_free_gdf['free_surface'].sum()
        metrics_egid_df['free_surface_det'] = np.where(metrics_egid_df['EGID'] == egid, free_surface_det, metrics_egid_df['free_surface_det'])
    
    # Compute Jaccard index and free surface for all buildings
    metrics_egid_df = metrics_egid_df.fillna(0)

    iou_average = metrics_egid_df['IoU'].mean()
    metrics_df['IoU'] = np.where(metrics_df['value'] == 'ALL', iou_average, metrics_df['IoU'])    

    occupied_label_sum = metrics_egid_df['occupied_surface_label'].sum()
    metrics_df['occupied_surface_label'] = np.where(metrics_df['value'] == 'ALL', occupied_label_sum, metrics_df['occupied_surface_label'])  
    occupied_det_sum = metrics_egid_df['occupied_surface_det'].sum()
    metrics_df['occupied_surface_det'] = np.where(metrics_df['value'] == 'ALL', occupied_det_sum, metrics_df['occupied_surface_det'])  
    free_label_sum = metrics_egid_df['free_surface_label'].sum()
    metrics_df['free_surface_label'] = np.where(metrics_df['value'] == 'ALL', free_label_sum, metrics_df['free_surface_label'])  
    free_det_sum = metrics_egid_df['free_surface_det'].sum()
    metrics_df['free_surface_det'] = np.where(metrics_df['value'] == 'ALL', free_det_sum, metrics_df['free_surface_det'])  

    # Concatenate roof attributes by EGID and get attributes keys
    metrics_egid_df = pd.merge(metrics_egid_df, egids, on='EGID')
    roof_attributes = egids.keys().tolist()
    roof_attributes.remove('EGID')

    # Compute metrics by roof attributes 
    logger.info("- Per roof attributes metrics")
    for attribute in roof_attributes:
        metrics_count_df = metrics_egid_df[[attribute, 'TP', 'FP', 'FN']].groupby([attribute], as_index=False).sum()
        metrics_iou_df = metrics_egid_df[[attribute, 'IoU']].groupby([attribute], as_index=False).mean()
        metrics_occupied_label_df = metrics_egid_df[[attribute, 'occupied_surface_label']].groupby([attribute], as_index=False).sum()
        metrics_occupied_det_df = metrics_egid_df[[attribute, 'occupied_surface_det']].groupby([attribute], as_index=False).sum()
        metrics_free_label_df = metrics_egid_df[[attribute, 'free_surface_label']].groupby([attribute], as_index=False).sum()
        metrics_free_det_df = metrics_egid_df[[attribute, 'free_surface_det']].groupby([attribute], as_index=False).sum()

        for val in metrics_egid_df[attribute].unique():
            TP = metrics_count_df['TP'][metrics_count_df[attribute] == val].iloc[0]  
            FP = metrics_count_df['FP'][metrics_count_df[attribute] == val].iloc[0]
            FN = metrics_count_df['FN'][metrics_count_df[attribute] == val].iloc[0]
            iou = metrics_iou_df['IoU'][metrics_iou_df[attribute] == val].iloc[0]    
            occupied_surface_label = metrics_occupied_label_df['occupied_surface_label'][metrics_occupied_label_df[attribute] == val].iloc[0]  
            occupied_surface_det = metrics_occupied_det_df['occupied_surface_det'][metrics_occupied_det_df[attribute] == val].iloc[0]  
            free_surface_label = metrics_free_label_df['free_surface_label'][metrics_free_label_df[attribute] == val].iloc[0] 
            free_surface_det = metrics_free_det_df['free_surface_det'][metrics_free_det_df[attribute] == val].iloc[0]   

            metrics_results = metrics.get_metrics(TP, FP, FN)
            tmp_df = pd.DataFrame.from_records([{'attribute': attribute, 'value': val, 
                                                **metrics_results, 'IoU': iou,
                                                'occupied_surface_label': occupied_surface_label, 'occupied_surface_det': occupied_surface_det,
                                                'free_surface_label': free_surface_label, 'free_surface_det': free_surface_det,}])
            metrics_df = pd.concat([metrics_df, tmp_df])   

    metrics_df = pd.concat([metrics_df, metrics_objects_df]).reset_index(drop=True)

    # Compute (1 - relative error) on occupied and free surfaces 
    metrics_df['occupied_re'] = abs(metrics_df['occupied_surface_det'] - metrics_df['occupied_surface_label']) / metrics_df['occupied_surface_label']
    metrics_df['free_re'] = abs(metrics_df['free_surface_det'] - metrics_df['free_surface_label']) / metrics_df['free_surface_label']

    # Sump-up results and save files
    TP = metrics_df['TP'][metrics_df.value == 'ALL'][0]
    FP = metrics_df['FP'][metrics_df.value == 'ALL'][0]
    FN = metrics_df['FN'][metrics_df.value == 'ALL'][0]
    precision = metrics_df['precision'][metrics_df.value == 'ALL'][0]
    recall = metrics_df['recall'][metrics_df.value == 'ALL'][0]
    f1 = metrics_df['f1'][metrics_df.value == 'ALL'][0]
    iou = metrics_df['IoU'][metrics_df.value == 'ALL'][0]
    occupied_surface_label = metrics_df['occupied_surface_label'][metrics_df.value == 'ALL'][0]
    occupied_surface_det = metrics_df['occupied_surface_det'][metrics_df.value == 'ALL'][0]
    free_surface_label = metrics_df['free_surface_label'][metrics_df.value == 'ALL'][0]
    free_surface_det = metrics_df['free_surface_det'][metrics_df.value == 'ALL'][0]

    written_files[feature_path] = layer_name
    feature_path = os.path.join(output_dir, 'metrics.csv')
    metrics_df.to_csv(feature_path, sep=',', index=False, float_format='%.4f')

    print('')
    logger.info(f"TP = {TP}, FP = {FP}, FN = {FN}")
    logger.info(f"TP+FN = {TP+FN}, TP+FP = {TP+FP}")
    logger.info(f"precision = {precision:.2f}, recall = {recall:.2f}, f1 = {f1:.2f}")
    logger.info(f"IoU for all EGIDs = {iou:.2f}")
    logger.info(f"Occupied surface relative error for all EGIDs = {(abs((occupied_surface_det - occupied_surface_label)/occupied_surface_label)):.2f}")
    logger.info(f"Free surface relative error for all EGIDs = {(abs((free_surface_det - free_surface_label)/free_surface_label)):.2f}")
    print('')


    # Check if detection or labels have been lost in the process
    nbr_tagged_labels = TP + FN
    labels_diff = nbr_labels - nbr_tagged_labels
    filename = os.path.join(output_dir, 'problematic_objects.gpkg')
    if os.path.exists(filename):
        os.remove(filename)
    if labels_diff != 0:
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

    logger.success("The following files were written. Let's check them out!")
    for path in written_files.keys():
        logger.success(f'  file: {path}, layer: {written_files[path]}')


    # Plots
    xlabel_dic = {'EGID': '', 'roof_type': '', 'roof_inclination': '',
                'object_class':'', 'area': r'Object area ($m^2$)', 
                'nearest_distance_border': r'Object distance (m)'} 

    plot_histo(output_dir, labels_gdf, detections_gdf, attribute=OBJECT_PARAMETERS, xlabel=xlabel_dic)
    for i in metrics_df.attribute.unique():
        plot_stacked_grouped(output_dir, metrics_df, attribute=i, xlabel=xlabel_dic[i])
        plot_stacked_grouped_percent(output_dir, metrics_df, attribute=i, xlabel=xlabel_dic[i])
        plot_metrics(output_dir, metrics_df, attribute=i, xlabel=xlabel_dic[i])
        if i in ['EGID', 'roof_type', 'roof_inclination']: 
            plot_surface(output_dir, metrics_df, attribute=i, xlabel=xlabel_dic[i])


    return metrics_df, labels_diff       # change for 1/(1 + diff_in_labels) if metrics can only be maximized.

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
    THD = cfg['filters']['threshold']
    AREA_THD_FACTOR = cfg['filters']['area_threshold_factor'] 
    OBJECT_PARAMETERS = cfg['object_attributes']['parameters']
    AREA_RANGES = cfg['object_attributes']['area_ranges'] 
    DISTANCE_RANGES = cfg['object_attributes']['distance_ranges'] 

    RANGES = [AREA_RANGES] + [DISTANCE_RANGES] 

    metrics_df, labels_diff = main(WORKING_DIR, OUTPUT_DIR, LABELS, DETECTIONS, ROOFS, EGIDS, METHOD, THD, AREA_THD_FACTOR, OBJECT_PARAMETERS, RANGES)

    # Stop chronometer  
    toc = time.time()
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()