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


def main(WORKING_DIR, OUTPUT_DIR, LABELS, DETECTIONS, ROOFS, EGIDS, method='one-to-one', threshold=0.1, visualisation=False):
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
    egids=pd.read_csv(EGIDS)
    logger.info(f'Working on {egids.shape[0]} EGIDs.')

    # Open shapefiles
    
    labels_gdf = gpd.read_file(LABELS)

    if labels_gdf.EGID.dtype != 'int64':
        labels_gdf['EGID'] = [round(float(egid)) for egid in labels_gdf.EGID.to_numpy()]

    labels_gdf = labels_gdf[(labels_gdf.occupation.astype(int) == 1) & (labels_gdf.EGID.isin(egids.EGID.to_numpy()))].copy()
    labels_gdf.drop(columns=['fid', 'type', 'layer', 'path'], inplace=True, errors='ignore')

    nbr_labels=labels_gdf.shape[0]
    logger.info(f"Read the file for labels: {nbr_labels} shapes")

    if isinstance(DETECTIONS, str):
        detections_gdf = gpd.read_file(DETECTIONS, layer='occupation_for_all_EGIDS')
    elif isinstance(DETECTIONS, gpd.GeoDataFrame):
        detections_gdf = DETECTIONS.copy()
    else:
        logger.critical(f'Unrecognized variable type for the detections: {type(DETECTIONS)}.')
        sys.exit(1)
    detections_gdf = detections_gdf[detections_gdf['occupation'].astype(int) == 1]
    detections_gdf['EGID'] = detections_gdf.EGID.astype(int)
    detections_gdf['ID_DET'] = detections_gdf.det_id.astype(int)
    detections_gdf = detections_gdf.rename(columns={"area": "area_DET"})
    logger.info(f"Read the file for detections: {len(detections_gdf)} shapes")

    # Get the rooftops shapes
    ROOFS_DIR, ROOFS_NAME = os.path.split(ROOFS)
    attribute = 'EGID'
    original_file_path = os.path.join(ROOFS_DIR, ROOFS_NAME)
    desired_file_path = os.path.join(ROOFS_DIR, ROOFS_NAME[:-4]  + "_" + attribute + ".shp")
    
    roofs = misc.dissolve_by_attribute(desired_file_path, original_file_path, name=ROOFS_NAME[:-4], attribute=attribute)
    roofs_gdf = roofs[roofs.EGID.isin(egids.EGID.to_numpy())].copy()
    roofs_gdf['EGID'] = roofs_gdf['EGID'].astype(int)
    roofs_gdf['area'] = round(roofs_gdf['geometry'].area, 4)

    logger.info('Get the free and occupied surface by EGID...')
    egid_surfaces_df = pd.DataFrame()
    labels_free_gdf, detections_free_gdf = metrics.get_free_surface(
        labels_gdf, 
        detections_gdf,
        roofs_gdf,
    )
    
    egid_surfaces_df['EGID'] = labels_gdf.EGID.unique()
    egid_surfaces_df['occupied_surface_label'] = [
        labels_free_gdf.loc[labels_free_gdf.EGID==egid, 'occupied_surface'].iloc[0]
        if egid in labels_free_gdf.EGID.unique() else 0
        for egid in egid_surfaces_df.EGID.unique() 
    ]
    egid_surfaces_df['occupied_surface_det'] = [
        detections_free_gdf.loc[detections_free_gdf.EGID==egid, 'occupied_surface'].iloc[0]
        if egid in detections_free_gdf.EGID.unique() else 0
        for egid in egid_surfaces_df.EGID.unique()
    ]
    egid_surfaces_df['free_surface_label'] = [
        labels_free_gdf.loc[labels_free_gdf.EGID==egid, 'free_surface'].iloc[0]
        if egid in labels_free_gdf.EGID.unique() else 0
        for egid in egid_surfaces_df.EGID.unique() 
    ]
    egid_surfaces_df['free_surface_det'] = [
        detections_free_gdf.loc[detections_free_gdf.EGID==egid, 'free_surface'].iloc[0]
        if egid in detections_free_gdf.EGID.unique() else 0
        for egid in egid_surfaces_df.EGID.unique()
    ]


    logger.info('Get the global free and occupied surface...')
    surfaces_df=pd.DataFrame()
    surfaces_df.loc[0,'occupied_surface_label'] = egid_surfaces_df['occupied_surface_label'].sum()
    surfaces_df['free_surface_label'] = egid_surfaces_df['free_surface_label'].sum()
    surfaces_df['occupied_surface_det'] = egid_surfaces_df['occupied_surface_det'].sum()
    surfaces_df['free_surface_det'] = egid_surfaces_df['free_surface_det'].sum()

    # Determine relative results
    surfaces_df['occupied_rel_diff'] = abs(surfaces_df['occupied_surface_det'] - surfaces_df['occupied_surface_label']) / surfaces_df['occupied_surface_label']
    surfaces_df['free_rel_diff'] = abs(surfaces_df['free_surface_det'] - surfaces_df['free_surface_label']) / surfaces_df['free_surface_label']

    feature_path = os.path.join(output_dir, 'global_surfaces.csv')
    written_files[feature_path] = ''
 

    # Concatenate roof attributes by EGID and get attributes keys
    egid_surfaces_df = pd.merge(egid_surfaces_df, egids, on='EGID')
    roof_attributes = egids.keys().tolist()
    roof_attributes.remove('EGID')

    # Compute free vs occupied surface by roof attributes 
    logger.info("- Free vs occupied surface per roof attribute")
    surface_types = ['occupied_surface_label', 'occupied_surface_det', 'free_surface_label', 'free_surface_det']
    attribute_surface_dict = {'attribute': [], 'value': []}
    for var in surface_types: attribute_surface_dict[var] = []

    attribute_surface_df=pd.DataFrame()
    for attribute in roof_attributes:
        for val in egid_surfaces_df[attribute].unique():
            attribute_surface_dict['value'] = val
            attribute_surface_dict['attribute'] = attribute
            for var in surface_types:
                surface = egid_surfaces_df.loc[egid_surfaces_df[attribute]==val, var].iloc[0]
                attribute_surface_dict[var] = surface

            attribute_surface_df = pd.concat([attribute_surface_df, pd.DataFrame(attribute_surface_dict, index=[0])], ignore_index=True)


    # Compute (1 - relative error) on occupied and free surfaces 
    attribute_surface_df['occupied_rel_diff'] = abs(attribute_surface_df['occupied_surface_det'] - attribute_surface_df['occupied_surface_label']) \
        / attribute_surface_df['occupied_surface_label']
    attribute_surface_df['free_rel_diff'] = abs(attribute_surface_df['free_surface_det'] - attribute_surface_df['free_surface_label']) / attribute_surface_df['free_surface_label']

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

        for i in attribute_surface_df.attribute.unique():
            filepath = figures.plot_surface(output_dir, attribute_surface_df, attribute=i, xlabel=xlabel_dict[i])
            written_files[filepath] = ''

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

    DETECTIONS=cfg['detections']
    LABELS = cfg['ground_truth']
    ROOFS = cfg['roofs']
    EGIDS = cfg['egids']

    METHOD = cfg['method']
    THRESHOLD = cfg['threshold']
    VISUALISATION = cfg['visualisation']

    written_files = main(WORKING_DIR, OUTPUT_DIR, LABELS, DETECTIONS, ROOFS, EGIDS, 
                         method=METHOD, threshold=THRESHOLD, visualisation=VISUALISATION)

    logger.success("The following files were written. Let's check them out!")
    for path in written_files.keys():
        logger.success(f'  file: {path}{"" if written_files[path] == "" else f", layer: {written_files[path]}"}')

    # Stop chronometer  
    toc = time.time()
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()