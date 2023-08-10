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
from yaml import load, FullLoader

import numpy as np
import pandas as pd
import geopandas as gpd

sys.path.insert(1, 'scripts')
import functions.fct_misc as fct_misc
import functions.fct_metrics as metrics

logger = fct_misc.format_logger(logger)


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
METHOD = cfg['method']

os.chdir(WORKING_DIR)

# Create an output directory in case it doesn't exist
_ = fct_misc.ensure_dir_exists(OUTPUT_DIR)

written_files={}

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
nbr_labels=gdf_gt.shape[0]
logger.info(f"Read GT file: {nbr_labels} shapes")

gdf_detec = gpd.read_file(DETECTIONS, layer='occupation_for_all_EGIDS')
gdf_detec = gdf_detec[gdf_detec['occupation'].astype(int) == 1]
gdf_detec['ID_DET'] = gdf_detec.pred_id
gdf_detec = gdf_detec.rename(columns={"area": "area_DET"})
logger.info(f"Read detection file: {len(gdf_detec)} shapes")

logger.info(f"Metrics computation")
if METHOD=='one-to-one':
    logger.info('Using the one-to-one method.')
elif METHOD=='one-to-many':
    logger.info('Using one-to-many method.')
else:
    logger.warning('Unknown method, defaulting to one-to-one.')

logger.info(f"Metrics computation:")
logger.info(f" - Compute TP, FP and FN")

tp_gdf, fp_gdf, fn_gdf = metrics.get_fractional_sets(gdf_detec, gdf_gt, method=METHOD)

# Compute metrics
precision, recall, f1 = metrics.get_metrics(tp_gdf, fp_gdf, fn_gdf)

TP = tp_gdf.shape[0]
FP = fp_gdf.shape[0]
FN = fn_gdf.shape[0]

if METHOD=='one-to-many':
    tp_with_duplicates=tp_gdf.copy()
    dissolved_tp_gdf=tp_with_duplicates.dissolve(by=['ID_DET'], as_index=False)

    geom1 = dissolved_tp_gdf.geometry.values.tolist()
    geom2 = dissolved_tp_gdf['geom_GT'].values.tolist()
    iou = []
    for (i, ii) in zip(geom1, geom2):
        iou.append(metrics.intersection_over_union(i, ii))
    dissolved_tp_gdf['IOU'] = iou

    tp_gdf=dissolved_tp_gdf.copy()

    logger.info(f'{tp_with_duplicates.shape[0]-tp_gdf.shape[0]} labels are under a shared predictions with at least one other label.')

logger.info(f"   TP = {TP}, FP = {FP}, FN = {FN}")
logger.info(f"   precision = {precision:.2f}, recall = {recall:.2f}, f1 = {f1:.2f}")
logger.info(f" - Compute mean Jaccard index")
if TP!=0:
    iou_average = tp_gdf['IOU'].mean()
    logger.info(f"   IOU average = {iou_average:.2f}")


two_preds_one_label=len(tp_gdf.loc[tp_gdf.duplicated(subset=['ID_GT']), 'ID_GT'].unique().tolist())
if two_preds_one_label > 0:
    logger.warning(f'{two_preds_one_label} labels are associated with more than one prediction considered as TP.')

nbr_tagged_labels = TP + FN -two_preds_one_label
filename=os.path.join(OUTPUT_DIR, 'problematic_objects.gpkg')
if os.path.exists(filename):
    os.remove(filename)
if nbr_labels != nbr_tagged_labels:
    logger.error(f'There are {nbr_labels} labels in input and {nbr_tagged_labels} labels in output.')
    logger.info(f'The list of the problematic labels in exported to {filename}.')

    if nbr_labels > nbr_tagged_labels:
        tagged_labels=tp_gdf['ID_GT'].unique().tolist() + fn_gdf['ID_GT'].unique().tolist()

        untagged_gt_gdf=gdf_gt[~gdf_gt['ID_GT'].isin(tagged_labels)]
        untagged_gt_gdf.drop(columns=['geom_GT', 'OBSTACLE'], inplace=True)

        layer_name='missing_label_tags'
        untagged_gt_gdf.to_file(filename, layer=layer_name, index=False)

    elif nbr_labels < nbr_tagged_labels:
        all_tagged_labels_gdf=pd.concat([tp_gdf, fn_gdf])

        duplicated_id_gt=all_tagged_labels_gdf.loc[all_tagged_labels_gdf.duplicated(subset=['ID_GT']), 'ID_GT'].unique().tolist()
        duplicated_labels=all_tagged_labels_gdf[all_tagged_labels_gdf['ID_GT'].isin(duplicated_id_gt)]
        duplicated_labels.drop(columns=['geom_GT', 'OBSTACLE', 'geom_DET', 'index_right', 'fid', 'FID', 'fme_basena'], inplace=True)

        layer_name='duplicated_label_tags'
        duplicated_labels.to_file(filename, layer=layer_name, index=False)
    
    written_files[filename]=layer_name


# Set the final dataframe with tagged prediction
tagged_preds_gdf = []
tagged_preds_gdf_dict = pd.concat([tp_gdf, fp_gdf, fn_gdf])

tagged_preds_gdf_dict.drop(['index_right', 'occupation_left', 'occupation_right', 'geom_GT', 'geom_DET'], axis = 1, inplace=True)
tagged_preds_gdf_dict.reset_index(drop=True, inplace=True)

layer_name='tagged_predictions'
feature_path = os.path.join(OUTPUT_DIR, 'tagged_predictions.gpkg')
tagged_preds_gdf_dict.to_file(feature_path, layer=layer_name, index=False)
written_files[feature_path]=layer_name


logger.success("The following files were written. Let's check them out!")
for path in written_files.keys():
    logger.success(f'  file: {path}, layer: {written_files[path]}')

# Stop chronometer  
toc = time.time()
logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

sys.stderr.flush()