#!/bin/python
# -*- coding: utf-8 -*-

#  proj-rooftops


import argparse
import os
import time
import sys
from loguru import logger
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

def main(WORKING_DIR, OUTPUT_DIR, LABELS, DETECTIONS, ROOFS, EGIDS, BINS, method='one-to-one', threshold=0.1, visualisation=False):
    """Assess the results by calculating the precision, recall and f1-score.

    Args:
        WORKING_DIR (path): working directory
        OUTPUT_DIR (path): output directory
        LABELS (path): file of the ground truth
        DETECTIONS (path): file of the detections
        ROOFS (path): file of the roof border and main elements
        EGIDS (list): EGIDs of interest
        method (string): method to use for the assessment of the results, either one-to-one, one-to-many or many-to-many. Defaults ot one-to-one.
        threshold (float): surface intersection threshold between label shape and detection shape to be considered as the same group. Defaults to 0.1.
        visualisation (bool): wheter or not to do and save the plots. Defaults to False.

    Returns:
        tuple:
            - DataFrame: metrics computed for different attribute.
            - GeoDataFrame: missing labels (lost during the process)
    """

    os.chdir(WORKING_DIR)

    # Create an output directory in case it doesn't exist
    output_dir = misc.ensure_dir_exists(os.path.join(OUTPUT_DIR, METHOD))
    threshold_str = str(THRESHOLD).replace('.', 'dot')

    written_files={}

    logger.info('Get input data...')

    # Get the EGIDS of interest
    egids = pd.read_csv(EGIDS)
    logger.info(f'Working on {egids.shape[0]} EGIDs.')

    # Open shapefiles
    
    labels_gdf = gpd.read_file(LABELS)

    if labels_gdf.EGID.dtype != 'int64':
        labels_gdf['EGID'] = [round(float(egid)) for egid in labels_gdf.EGID.to_numpy()]
    if 'occupation' in labels_gdf.columns:
        labels_gdf = labels_gdf[(labels_gdf.occupation.astype(int) == 1) & (labels_gdf.EGID.isin(egids.EGID.to_numpy()))].copy()
    if 'type' in labels_gdf.columns:
        labels_gdf['type'] = labels_gdf['type'].astype(int)
        labels_gdf = labels_gdf.rename(columns={'type':'obj_class'})
        # Type 12 corresponds to free surfaces, other classes are ojects
        labels_gdf.loc[labels_gdf['obj_class'] == 4, 'descr'] = 'Aero'
        logger.info("- Filter objects and EGID")
        labels_gdf = labels_gdf[(labels_gdf['obj_class'] != 12) & (labels_gdf.EGID.isin(egids.EGID.to_numpy()))].copy()
    else:
        labels_gdf = labels_gdf[labels_gdf.EGID.isin(egids.EGID.to_numpy())].copy()

    labels_gdf.drop(columns=['fid', 'layer', 'path'], inplace=True, errors='ignore')

    nbr_labels=labels_gdf.shape[0]
    logger.info(f"Read the file for labels: {nbr_labels} shapes")

    # Read detections shapefile 
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
        detections_gdf['detection_id'] = detections_gdf.det_id.astype(int)
    else:
        detections_gdf['detection_id'] = detections_gdf.index

    detections_gdf = detections_gdf.explode(ignore_index=True)
    logger.info(f"- {len(detections_gdf)} detection's shapes")

    # Get the rooftops shapes
    ROOFS_DIR, ROOFS_NAME = os.path.split(ROOFS)
    attribute = 'EGID'
    original_file_path = os.path.join(ROOFS_DIR, ROOFS_NAME)
    desired_file_path = os.path.join(ROOFS_DIR, ROOFS_NAME[:-4]  + "_" + attribute + ".shp")
    
    roofs = misc.dissolve_by_attribute(desired_file_path, original_file_path, name=ROOFS_NAME[:-4], attribute=attribute)
    roofs['EGID'] = roofs['EGID'].astype(int)
    roofs_gdf = roofs[roofs.EGID.isin(egids.EGID.to_numpy())].copy()
    roofs_gdf['area'] = round(roofs_gdf['geometry'].area, 4)

    logger.info('Get the free and occupied surface by EGID...')
    egid_surfaces_df = pd.DataFrame()
    labels_free_gdf, detections_free_gdf = metrics.get_free_surface(
        labels_gdf, 
        detections_gdf,
        roofs_gdf,
    )

    # Compute free and occupied surfaces of detection and GT by EGID 
    egid_surfaces_df['EGID'] = labels_gdf.EGID.unique()
    egid_surfaces_df['total_surface'] = [
        roofs_gdf.loc[roofs_gdf.EGID == egid, 'area'].iloc[0]
        if egid in roofs_gdf.EGID.unique() else 0
        for egid in egid_surfaces_df.EGID.unique() 
    ]
    egid_surfaces_df['occupied_surface_label'] = [
        labels_free_gdf.loc[labels_free_gdf.EGID == egid, 'occupied_surface'].iloc[0]
        if egid in labels_free_gdf.EGID.unique() else 0
        for egid in egid_surfaces_df.EGID.unique() 
    ]
    egid_surfaces_df['occupied_surface_det'] = [
        detections_free_gdf.loc[detections_free_gdf.EGID == egid, 'occupied_surface'].iloc[0]
        if egid in detections_free_gdf.EGID.unique() else 0
        for egid in egid_surfaces_df.EGID.unique()
    ]
    egid_surfaces_df['free_surface_label'] = [
        labels_free_gdf.loc[labels_free_gdf.EGID == egid, 'free_surface'].iloc[0]
        if egid in labels_free_gdf.EGID.unique() else 0
        for egid in egid_surfaces_df.EGID.unique() 
    ]
    egid_surfaces_df['free_surface_det'] = [
        detections_free_gdf.loc[detections_free_gdf.EGID == egid, 'free_surface'].iloc[0]
        if egid in detections_free_gdf.EGID.unique() else 0
        for egid in egid_surfaces_df.EGID.unique()
    ]

    # Compute relative error of detected surfaces 
    egid_surfaces_df['re_occupied_surface'] = misc.relative_error_df(egid_surfaces_df, target='occupied_surface_label', measure='occupied_surface_det')
    egid_surfaces_df['re_free_surface'] = misc.relative_error_df(egid_surfaces_df, target='free_surface_label', measure='free_surface_det') 

    # Compute surface ratio between detections and labels
    egid_surfaces_df['ratio_occupied_surface_label'] = egid_surfaces_df['occupied_surface_label'] / egid_surfaces_df['total_surface']
    egid_surfaces_df['ratio_occupied_surface_det'] = egid_surfaces_df['occupied_surface_det'] / egid_surfaces_df['total_surface']
    egid_surfaces_df['ratio_free_surface_label'] = egid_surfaces_df['free_surface_label'] / egid_surfaces_df['total_surface']
    egid_surfaces_df['ratio_free_surface_det'] = egid_surfaces_df['free_surface_det'] / egid_surfaces_df['total_surface']

    # Attribute bin to surface area
    bin_labels = []
    for i in range(len(BINS)-1):
        bin = f"{BINS[i]}-{BINS[i+1]}"
        bin_labels.append(bin)
    egid_surfaces_df['bin_occupied_surface_label (%)'] = pd.cut(egid_surfaces_df['ratio_occupied_surface_label'] * 100, BINS, right=True, labels=bin_labels)
    egid_surfaces_df['bin_occupied_surface_det (%)'] = pd.cut(egid_surfaces_df['ratio_occupied_surface_det'] * 100, BINS, right=True, labels=bin_labels)
    egid_surfaces_df['bin_free_surface_label (%)'] = pd.cut(egid_surfaces_df['ratio_free_surface_label'] * 100, BINS, right=True, labels=bin_labels)
    egid_surfaces_df['bin_free_surface_det (%)'] = pd.cut(egid_surfaces_df['ratio_free_surface_det'] * 100, BINS, right=True, labels=bin_labels)

    # Assess surface bins, 0: different bin, 1: same bin
    egid_surfaces_df['assess_occupied_surface_bins'] = np.where(egid_surfaces_df['bin_occupied_surface_det (%)']==egid_surfaces_df['bin_occupied_surface_label (%)'], 1, 0)
    egid_surfaces_df['assess_free_surface_bins'] = np.where(egid_surfaces_df['bin_free_surface_det (%)']==egid_surfaces_df['bin_free_surface_label (%)'], 1, 0)

    # Save EGID df 
    feature_path = os.path.join(output_dir, 'EGID_surfaces.csv')
    egid_surfaces_df.round(3).to_csv(feature_path, sep=',', index=False, float_format='%.4f')
    written_files[feature_path] = ''

    logger.info('Get the global free and occupied surface...')
    surfaces_df = pd.DataFrame()

    # Compute global surfaces
    surfaces_df.loc[0,'occupied_surface_label'] = egid_surfaces_df['occupied_surface_label'].sum()
    surfaces_df['free_surface_label'] = egid_surfaces_df['free_surface_label'].sum()
    surfaces_df['occupied_surface_det'] = egid_surfaces_df['occupied_surface_det'].sum()
    surfaces_df['free_surface_det'] = egid_surfaces_df['free_surface_det'].sum()

    # Compute the global surfaces ratio between detections and labels
    surfaces_df.loc[0,'ratio_occupied_surface_label'] = egid_surfaces_df['ratio_occupied_surface_label'].mean()
    surfaces_df['ratio_occupied_surface_det'] = egid_surfaces_df['ratio_occupied_surface_det'].mean()
    surfaces_df['ratio_free_surface_label'] = egid_surfaces_df['ratio_free_surface_label'].mean()
    surfaces_df['ratio_free_surface_det'] = egid_surfaces_df['ratio_free_surface_det'].mean()

    # Determine the global relative error of detected surfaces
    surfaces_df['occupied_rel_diff'] = abs(surfaces_df['occupied_surface_det'] - surfaces_df['occupied_surface_label']) / surfaces_df['occupied_surface_label']
    surfaces_df['free_rel_diff'] = abs(surfaces_df['free_surface_det'] - surfaces_df['free_surface_label']) / surfaces_df['free_surface_label']

    # Determine the global number of EGID surfaces correctly detected
    surfaces_df['TP_surface'] = len(egid_surfaces_df[egid_surfaces_df['assess_occupied_surface_bins']==1])
    surfaces_df['FP_surface'] = len(egid_surfaces_df[egid_surfaces_df['assess_occupied_surface_bins']==0])

    # Determine the global accuracy of detected surfaces
    surfaces_df['surface_accuracy'] = surfaces_df['TP_surface'] / len(egid_surfaces_df['EGID'])

    # Determine the accuracy of detected surfaces by surface bins
    for i in np.unique(egid_surfaces_df['bin_free_surface_label (%)']):
        surfaces_df[i] = len(egid_surfaces_df.loc[(egid_surfaces_df['bin_free_surface_label (%)']==i) & (egid_surfaces_df['assess_occupied_surface_bins']==1)]) \
         / len(egid_surfaces_df[egid_surfaces_df['bin_free_surface_label (%)']==i])

    # Concatenate roof attributes by EGID and get attributes keys
    egid_surfaces_df = pd.merge(egid_surfaces_df, egids, on='EGID')
    roof_attributes = egids.keys().tolist()
    roof_attributes.remove('EGID')

    # Compute free vs occupied surface by roof attributes 
    logger.info("- Free vs occupied surface per roof attribute")
    surface_types = ['occupied_surface_label', 'occupied_surface_det', 'free_surface_label', 'free_surface_det',
    'ratio_occupied_surface_label', 'ratio_occupied_surface_det', 'ratio_free_surface_label', 'ratio_free_surface_det']
    attribute_surface_dict = {'attribute': [], 'value': []}
    for var in surface_types: attribute_surface_dict[var] = []

    attribute_surface_df = pd.DataFrame()

    tmp_df = pd.DataFrame.from_records([{'attribute': 'EGID', 'value': 'ALL', **surfaces_df.iloc[0]}])
    attribute_surface_df = pd.concat([attribute_surface_df, tmp_df])

    for attribute in roof_attributes:
        for val in egid_surfaces_df[attribute].unique():
            attribute_surface_dict['attribute'] = attribute
            attribute_surface_dict['value'] = val
            for var in surface_types:
                surface = egid_surfaces_df.loc[egid_surfaces_df[attribute] == val, var].iloc[0]
                attribute_surface_dict[var] = surface
            
            attribute_surface_df = pd.concat([attribute_surface_df, pd.DataFrame(attribute_surface_dict, index=[0])], ignore_index=True)

    # Compute relative error on occupied and free surfaces 
    attribute_surface_df['occupied_rel_diff'] = abs(attribute_surface_df['occupied_surface_det'] - attribute_surface_df['occupied_surface_label']) \
        / attribute_surface_df['occupied_surface_label']
    attribute_surface_df['free_rel_diff'] = abs(attribute_surface_df['free_surface_det'] - attribute_surface_df['free_surface_label']) \
        / attribute_surface_df['free_surface_label']

    feature_path = os.path.join(output_dir, 'surfaces_by_attributes.csv')
    attribute_surface_df.round(3).to_csv(feature_path, sep=',', index=False, float_format='%.4f')
    written_files[feature_path] = ''
    

    print()
    logger.info(f"Occupied surface relative error for all EGIDs = {(surfaces_df.loc[0,'occupied_rel_diff'] ):.2f}")
    logger.info(f"Free surface relative error for all EGIDs = {(surfaces_df.loc[0, 'free_rel_diff']):.2f}")
    print()

    if visualisation:
        # Plots
        xlabel_dict = {'EGID': '', 'roof_type': '', 'roof_inclination': ''} 

        _ = figures.plot_surface_bin(output_dir, attribute_surface_df, bins=bin_labels, attribute='EGID')
        for i in attribute_surface_df.attribute.unique():
            _ = figures.plot_surface(output_dir, attribute_surface_df, attribute=i, xlabel=xlabel_dict[i])

    return written_files
    

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

    DETECTIONS = cfg['detections']
    LABELS = cfg['ground_truth']
    ROOFS = cfg['roofs']
    EGIDS = cfg['egids']

    METHOD = cfg['method']
    THRESHOLD = cfg['threshold']
    BINS = cfg['bins']
    VISUALISATION = cfg['visualisation']

    written_files = main(WORKING_DIR, OUTPUT_DIR, LABELS, DETECTIONS, ROOFS, EGIDS, BINS,
                         method=METHOD, threshold=THRESHOLD, visualisation=VISUALISATION)

    logger.success("The following files were written. Let's check them out!")
    for path in written_files.keys():
        logger.success(f'  file: {path}{"" if written_files[path] == "" else f", layer: {written_files[path]}"}')

    # Stop chronometer  
    toc = time.time()
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()