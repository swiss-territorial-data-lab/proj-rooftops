#!/bin/python
# -*- coding: utf-8 -*-

#  proj-rooftops

import argparse
import os
import sys
import time
from loguru import logger
from tqdm import tqdm
from yaml import load, FullLoader

import geopandas as gpd
import pandas as pd

sys.path.insert(1, 'scripts')
import functions.fct_figures as figures
import functions.fct_misc as misc

logger = misc.format_logger(logger)

# Functions --------------------------

def main(WORKING_DIR, OUTPUT_DIR, LABELS, EGIDS_TRAINING, EGIDS_TEST, roofs=None, object_parameters=[], ranges=[], 
         additional_metrics=False, visualisation=False):
    """Assess the results by calculating the precision, recall and f1-score.

    Args:
        WORKING_DIR (path): working directory
        OUTPUT_DIR (path): output directory
        LABELS (path): file of the ground truth
        EGIDS_TRAINING (list): EGIDs of the training dataset
        EGIDS_TEST (list): EGIDs of the test dataset
        roofs (string): file of the roofs. Defaults to None.
        object_parameters (list): list of object parameter to be processed ('area', 'nearest_distance_border', 'nearest_distance_centroid')
        ranges (list): list of list of the bins to process by object_parameters.
        additional_metrics (bool): wheter or not to do the by-EGID, by-object, by-class metrics. Defaults to False.
        visualisation (bool): wheter or not to do and save the plots. Defaults to False.

    Returns:

    """

    os.chdir(WORKING_DIR)

    # Create an output directory in case it doesn't exist
    output_dir = misc.ensure_dir_exists(os.path.join(OUTPUT_DIR))

    written_files = {}

    logger.info("Get input data")

    # Get the EGIDS of interest
    egids_test = pd.read_csv(EGIDS_TEST)
    egids_test['dataset'] = 'test' 
    egids_training = pd.read_csv(EGIDS_TRAINING)
    egids_training['dataset'] = 'training' 
    egids = pd.concat([egids_training, egids_test])
    array_egids = egids.EGID.to_numpy()
    logger.info(f"- {egids.shape[0]} selected EGIDs")


    if ('EGID' in roofs) | ('egid' in roofs):
        roofs = gpd.read_file(roofs)
    else:  
        # Get the rooftops shapes
        logger.info("- Roof shapes")
        ROOFS_DIR, ROOFS_NAME = os.path.split(roofs)
        desired_file_path = roofs[:-4]  + "_EGID.shp"
        roofs = misc.dissolve_by_attribute(desired_file_path, roofs, name=ROOFS_NAME[:-4], attribute='EGID')

    roofs['EGID'] = roofs['EGID'].astype(int)
    roofs_gdf = roofs[roofs.EGID.isin(array_egids)].copy()
    logger.info(f"Read the file for roofs: {len(roofs_gdf)} shapes")

    # Open shapefiles
    labels_gdf = gpd.read_file(LABELS)

    if labels_gdf.EGID.dtype != 'int64':
        labels_gdf['EGID'] = [round(float(egid)) for egid in labels_gdf.EGID.to_numpy()]
    if 'occupation' in labels_gdf.columns:
        labels_gdf = labels_gdf[(labels_gdf.occupation.astype(int) == 1) & (labels_gdf.EGID.isin(array_egids))].copy()
    if 'type' in labels_gdf.columns:
        labels_gdf['type'] = labels_gdf['type'].astype(int)
        labels_gdf = labels_gdf.rename(columns={'type':'obj_class'})
        # Type 12 corresponds to free surfaces, other classes are objects
        labels_gdf.loc[labels_gdf['obj_class'] == 4, 'descr'] = 'Aero'
        logger.info("- Filter objects and EGID")
        labels_gdf = labels_gdf[(labels_gdf['obj_class'] != 12) & (labels_gdf.EGID.isin(array_egids))].copy()
    else:
        labels_gdf = labels_gdf[labels_gdf.EGID.isin(array_egids)].copy()

    # Clip labels to the corresponding roof
    for egid in array_egids:
        labels_egid_gdf = labels_gdf[labels_gdf.EGID == egid].copy()
        # labels_egid_gdf = labels_egid_gdf.clip(roofs_gdf.loc[roofs_gdf.EGID == egid, 'geometry'].buffer(-0.10, join_style='mitre'), keep_geom_type=True)

        tmp_gdf = labels_gdf[labels_gdf.EGID != egid].copy()
        labels_gdf = pd.concat([tmp_gdf, labels_egid_gdf], ignore_index=True)

    labels_gdf = labels_gdf.merge(egids, how='left', on='EGID').copy()
    labels_gdf['label_id'] = labels_gdf.id
    labels_gdf['area'] = round(labels_gdf.area, 4)

    labels_gdf.drop(columns=['fid', 'layer', 'path'], inplace=True, errors='ignore')
    # labels_gdf = labels_gdf.explode(ignore_index=True)
    
    nbr_labels = labels_gdf.shape[0]
    logger.info(f"- {nbr_labels} label shapes")

    if (len(object_parameters) > 0):
  
        ranges_dict = {object_parameters[i]: ranges[i] for i in range(len(object_parameters))}

        ## Get nearest distance between polygons
        labels_gdf = misc.nearest_distance(labels_gdf, roofs_gdf, join_key='EGID', parameter='nearest_distance_centroid', lsuffix='_label', rsuffix='_roof')
        labels_gdf = misc.nearest_distance(labels_gdf, roofs_gdf, join_key='EGID', parameter='nearest_distance_border', lsuffix='_label', rsuffix='_roof')      

        ## Get roundness of polygons
        labels_gdf = misc.roundness(labels_gdf)   

    # Plots
    xlabel_dict = {'EGID': '', 'roof_type': '', 'roof_inclination': '',
                    'object_class':'', 'area': r'Object area ($m^2$)', 
                    'nearest_distance_border': r'Object distance (m)', 
                    'nearest_distance_centroid': r'Object distance (m)',
                    'roundness': r'Roundness'} 

    _ = figures.plot_stacked_grouped_object(output_dir, labels_gdf, param_ranges=ranges_dict, param='roundness', attribute='object_class', label=xlabel_dict)
    _ = figures.plot_stacked_grouped_object(output_dir, labels_gdf, param_ranges=ranges_dict, param='area', attribute='object_class', label=xlabel_dict)
    _ = figures.plot_stacked_grouped_object(output_dir, labels_gdf, param_ranges=ranges_dict, param='nearest_distance_centroid', attribute='object_class', label=xlabel_dict)
    _ = figures.plot_histo_object(output_dir, labels_gdf, attribute='object_class', datasets=['training', 'test'])

    return written_files

# ------------------------------------------

if __name__ == "__main__":

    # Start chronometer
    tic = time.time()
    logger.info("Results assessment")
    logger.info('Starting...')

    # Argument and parameter specification
    parser = argparse.ArgumentParser(description="The script allows to plots GT characteristics (STDL.proj-rooftops)")
    parser.add_argument('config_file', type=str, help='Framework configuration file')
    args = parser.parse_args()

    logger.info(f"Using {args.config_file} as config file.")

    with open(args.config_file) as fp:
        cfg = load(fp, Loader=FullLoader)[os.path.basename(__file__)]

    # Load input parameters
    WORKING_DIR = cfg['working_dir']
    OUTPUT_DIR = cfg['output_dir']
    LABELS = cfg['ground_truth']
    EGIDS_TRAINING = cfg['egids_training']
    EGIDS_TEST = cfg['egids_test']
    ROOFS = cfg['roofs']

    ADDITIONAL_METRICS = cfg['additional_metrics'] if 'additional_metrics' in cfg.keys() else False
    OBJECT_PARAMETERS = cfg['object_attributes']['parameters']
    AREA_RANGES = cfg['object_attributes']['area_ranges']
    DISTANCE_RANGES = cfg['object_attributes']['distance_ranges']
    ROUND_RANGES = cfg['object_attributes']['round_ranges']
    VISU = cfg['visualisation'] if 'visualisation' in cfg.keys() else False

    RANGES = [AREA_RANGES] + [DISTANCE_RANGES] + [ROUND_RANGES] 

    written_files = main(WORKING_DIR, OUTPUT_DIR, LABELS, EGIDS_TRAINING, EGIDS_TEST,
                                             roofs=ROOFS,
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