#!/bin/python
# -*- coding: utf-8 -*-
# 
#  proj-rooftops


import os
import sys
import argparse
from loguru import logger
from yaml import load, FullLoader

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(1, 'scripts')
from functions.fct_misc import format_logger

logger = format_logger(logger)

# Argument and parameter specification
parser = argparse.ArgumentParser(description="The script compares the occupation classification based on LiDAR products with that of the experts.")
parser.add_argument('config_file', type=str, help='Framework configuration file')
args = parser.parse_args()

logger.info(f"Using {args.config_file} as config file.")
with open(args.config_file) as fp:
    cfg = load(fp, Loader=FullLoader)[os.path.basename(__file__)]

# Define constants ----------------------------------

WORKING_DIR = cfg['working_dir']
OUTPUT_DIR = cfg['output_dir']

GT_PATH = cfg['gt_file']
OCEN_LAYER = cfg['layer_ocen']
OCAN_LAYER = cfg['layer_ocan']

PREDICTIONS_PATH = cfg['predictions_file']
PREDICTIONS_LAYER = cfg['predictions_layer']

os.chdir(WORKING_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Data processing --------------------------------------

logger.info('Read the files...')

ocen_gt = gpd.read_file(GT_PATH, layer=OCEN_LAYER)
ocan_gt = gpd.read_file(GT_PATH, layer=OCAN_LAYER)

predictions = gpd.read_file(PREDICTIONS_PATH, layer=PREDICTIONS_LAYER)
predictions = predictions[predictions['status']!='undefined']
logger.warning('The roofs classified as undefined are not considered.')

logger.info('Calculate the satisfaction rate...')

ocen_gt['OBJECTID'] = ocen_gt.OBJECTID.astype('int64')
ocen_gdf = pd.merge(
    predictions[['OBJECTID', 'status', 'reason', 'std_i', 'MOE_i', 'median_r', 'mean_r', 'nodata_overlap', 'geometry']], 
    ocen_gt[['OBJECTID', 'class']], 
    on='OBJECTID'
)
ocen_gdf.rename(columns={'status_ocen': 'status'}, inplace=True)
ocan_gt['OBJECTID'] = ocan_gt.OBJECTID.astype('int64')
ocan_gdf = pd.merge(
    predictions[['OBJECTID', 'status', 'reason', 'std_i', 'MOE_i', 'median_r', 'mean_r', 'nodata_overlap', 'geometry']], 
    ocan_gt[['OBJECTID', 'class']], 
    on='OBJECTID'
)
ocan_gdf.rename(columns={'status_ocan': 'status'}, inplace=True)

possible_classes = ocan_gdf['status'].unique()
agreement_dict = {'OCAN': [ocan_gdf, []], 'OCEN': [ocen_gdf, []]}
for key in agreement_dict.keys():
    gdf, agreement_list = agreement_dict[key]
    gdf.loc[gdf['class']=='not occupied', 'class'] = 'potentially free'
    gdf['agreement'] = [1 if gt == pred else 0 for gt, pred in zip(gdf['class'], gdf.status)]

    agreement_list.append(round(gdf.agreement.sum()/gdf.agreement.shape[0], 3))
    for occupation_class in possible_classes:
        condition = gdf['class']==occupation_class
        agreement_list.append(round(gdf.loc[condition, 'agreement'].sum()/gdf.loc[condition, 'agreement'].shape[0], 3))

    agreement_dict[key] = agreement_list

logger.info('Export the files...')
agreement_pd = pd.DataFrame.from_dict(agreement_dict, orient='index', columns=np.append(np.array('global'), possible_classes))
filepath_csv = os.path.join(OUTPUT_DIR, 'agreement_rates_manual_thrd.csv')
agreement_pd.to_csv(filepath_csv)

all_info_gdf = pd.merge(ocan_gdf, ocen_gdf[['OBJECTID', 'class', 'agreement', 'status', 'reason']], on='OBJECTID', suffixes=('_ocan', '_ocen'))
filepath = os.path.join(OUTPUT_DIR, 'comparison_occupation_classif.gpkg')
all_info_gdf[['OBJECTID', 'class_ocan', 'agreement_ocan', 'status_ocan', 'reason_ocan', 'class_ocen', 'agreement_ocen', 'status_ocen', 'reason_ocen',
              'std_i', 'MOE_i', 'median_r', 'mean_r', 'nodata_overlap', 'geometry']].to_file(filepath)

logger.success(f'Two files were written: "{filepath}" and {filepath_csv}.')