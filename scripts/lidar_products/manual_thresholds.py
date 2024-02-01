#!/bin/python
# -*- coding: utf-8 -*-
# 
#  proj-rooftops

import argparse
import os
import sys
from loguru import logger
from yaml import load, FullLoader

import geopandas as gpd

sys.path.insert(1, 'scripts')
from functions.fct_misc import format_logger

logger = format_logger(logger)

# Define functions ---------------

def cause_occupation(df, condition, message='Undefined cause'):
    '''
    Set the status to “occupied” and write the reason behind this.

    - df: dataframe of the roof planes
    - condition: condition for the the "occupied" status
    - message: message to write

    return: df with the column 'status' and 'reason' filled where the condition matched
    '''

    df.loc[condition, 'status'] = 'occupied'
    df.loc[condition, 'reason'] = message

    return df

# Define parameters ---------------

# Argument and parameter specification
parser = argparse.ArgumentParser(description="The script classifies the roof planes by occupation degree.")
parser.add_argument('config_file', type=str, help='Framework configuration file')
args = parser.parse_args()

logger.info(f"Using {args.config_file} as config file.")
with open(args.config_file) as fp:
    cfg = load(fp, Loader=FullLoader)[os.path.basename(__file__)]

WORKING_DIR = cfg['working_dir']
OUTPUT_DIR = cfg['output_dir']

ROOF_FILE = cfg['roof_file']
ROOF_LAYER = cfg['roof_layer']

# Filtering parameters
NODATA_OVERLAP = 0.25
LIM_STD = 5500
LIM_MOE = 400
LIM_ROUGHNESS = 7.5

STAT_LIMITS = {'MOE_i': LIM_MOE, 'std_i': LIM_STD, 'median_r': LIM_ROUGHNESS}

# Main ----------------------------------

os.chdir(WORKING_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

logger.info('Import roofs...')

roofs_gdf = gpd.read_file(ROOF_FILE, layer=ROOF_LAYER)
logger.info(f'   - {roofs_gdf.shape[0]} roofs imported')
logger.info(f'{roofs_gdf[roofs_gdf.status.isna()].shape[0]} roofs still have to be classified.')


logger.info('Filter roofs based on the nodata percentage...')
message = f'More than {NODATA_OVERLAP*100}% of the area is not classified as building in the LiDAR point cloud.'
roofs_gdf = cause_occupation(roofs_gdf, (roofs_gdf['nodata_overlap'] > NODATA_OVERLAP) & roofs_gdf.status.isna(), message)

nbr_other_class_roofs = roofs_gdf[roofs_gdf.reason == message].shape[0]
logger.info(f'{nbr_other_class_roofs} roofs are classified as occupied. ' + message)


logger.info('Filter roof planes with statistical threshold values on LiDAR intensity and roughness rasters...')

# Get roofs with more than one stat over the limits.
index_over_lim = []
for attribute in STAT_LIMITS.keys():
    index_over_lim.extend(roofs_gdf[roofs_gdf[attribute] > STAT_LIMITS[attribute]].index.tolist())
seen = set()
dupes_index = [roof_index for roof_index in index_over_lim if roof_index in seen or seen.add(roof_index)]
index_over_lim = list(dict.fromkeys(index_over_lim))

# Get roofs with one stat over the limit.
roofs_gdf = cause_occupation(roofs_gdf, roofs_gdf.index.isin(dupes_index) & roofs_gdf.status.isna(), 'Several parameters are over thresholds.')
roofs_gdf = cause_occupation(roofs_gdf, (roofs_gdf['MOE_i'] > LIM_MOE) & roofs_gdf.status.isna(), f'The margin of error of the mean for the intensity is over {LIM_MOE}.')
roofs_gdf = cause_occupation(roofs_gdf, (roofs_gdf['std_i'] > LIM_STD) & roofs_gdf.status.isna(), f'The standard deviation for the intensity is over {LIM_STD}.')
roofs_gdf = cause_occupation(roofs_gdf, (roofs_gdf['median_r'] > LIM_ROUGHNESS) & roofs_gdf.status.isna(), f'The median of the roughness is over {LIM_ROUGHNESS}.')

logger.info(f'{roofs_gdf[roofs_gdf.index.isin(index_over_lim) & roofs_gdf.status.isna()].shape[0]} roof planes exceed at least one statistical threshold values')
logger.info('They have been classified as "occupied" surfaces.')

roofs_gdf.loc[roofs_gdf.status.isna(), 'status'] = 'potentially free'
logger.info(f'{roofs_gdf[roofs_gdf.status.isna()].shape[0]} roof planes do not exceed any threshold values')
logger.info('They have been classified as "potentially free" surfaces.')

logger.info('Save file...')
filepath = os.path.join(OUTPUT_DIR, 'roofs.gpkg')
layername = 'manually_filtered_roofs'
roofs_gdf.to_file(filepath, layer=layername)

logger.success(f'The files were written in the geopackage "{filepath}" in the layer {layername}.')