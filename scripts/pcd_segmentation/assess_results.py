#!/bin/python
# -*- coding: utf-8 -*-

#  proj-rooftops
#
#      Clemence Herny 
#      Gwenaelle Salamin
#      Alessandro Cerioni 


import os, sys
import time
import argparse
from loguru import logger
from tqdm import tqdm
from yaml import load, FullLoader

import numpy as np
import pandas as pd
import geopandas as gpd

sys.path.insert(1, 'scripts')
import functions.fct_com as fct_com

logger = fct_com.format_logger(logger)


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
GT = cfg['gt']
EGIDS = cfg['egids']

os.chdir(WORKING_DIR)

# Create an output directory in case it doesn't exist
_ = fct_com.ensure_dir_exists(OUTPUT_DIR)

# Get the EGIDS of interest
with open(EGIDS, 'r') as src:
    egids=src.read()
egid_list=[int(egid) for egid in egids.split("\n")]

# Open shapefiles
gdf_gt = gpd.read_file(GT)
if 'OBSTACLE' in gdf_gt.columns:
    gdf_gt.rename(columns={'OBSTACLE': 'occupation'}, inplace=True)
gdf_gt = gdf_gt[(gdf_gt.occupation.astype(int) == 1) & (gdf_gt.EGID.isin(egid_list))]
gdf_gt['ID_GT'] = gdf_gt.index
gdf_gt = gdf_gt.rename(columns={"area": "area_GT"})
logger.info(f"Read GT file: {len(gdf_gt)} shapes")

gdf_detec = gpd.read_file(DETECTIONS, layer='occupation_for_all_EGIDS')
gdf_detec = gdf_detec[gdf_detec['occupation'] == 1]
gdf_detec['ID_DET'] = gdf_detec.pred_id
gdf_detec = gdf_detec.rename(columns={"area": "area_DET"})
logger.info(f"Read detection file: {len(gdf_detec)} shapes")


logger.info(f"Metrics computation:")
logger.info(f" - Compute TP, FP and FN")

tp_gdf, fp_gdf, fn_gdf = fct_com.get_fractional_sets(gdf_detec, gdf_gt)

# Compute metrics
precision, recall, f1 = fct_com.get_metrics(tp_gdf, fp_gdf, fn_gdf)

TP = len(tp_gdf)
FP = len(fp_gdf)
FN = len(fn_gdf)

logger.info(f"   TP = {TP}, FP = {FP}, FN = {FN}")
logger.info(f"   precision = {precision:.2f}, recall = {recall:.2f}, f1 = {f1:.2f}")
logger.info(f" - Compute mean Jaccard index")
if TP!=0:
    iou_average = tp_gdf['IOU'].mean()
logger.info(f"   IOU average = {iou_average:.2f}")


# Set the final dataframe with tagged prediction
tagged_preds_gdf = []
tagged_preds_gdf_dict = pd.concat([tp_gdf, fp_gdf, fn_gdf])

tagged_preds_gdf_dict.drop(['index_right', 'occupation_left', 'occupation_right', 'geom_GT', 'geom_DET'], axis = 1, inplace=True)
tagged_preds_gdf_dict.reset_index(drop=True, inplace=True)

feature_path = os.path.join(OUTPUT_DIR, 'tagged_predictions.gpkg')
tagged_preds_gdf_dict.to_file(feature_path, driver='GPKG', index=False)
logger.success(f'The final results were written it the file {feature_path}')


# Stop chronometer  
toc = time.time()
logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

sys.stderr.flush()