#!/bin/python
# -*- coding: utf-8 -*-

#  Proj rooftops
#
#      Clemence Herny 
#      Gwenaelle Salamin
#      Alessandro Cerioni 
#      Copyright (c) 2020 Republic and Canton of Geneva
#

import os, sys
from loguru import logger
from tqdm import tqdm
from yaml import load, FullLoader

import pandas as pd
import geopandas as gpd

sys.path.insert(1, 'scripts')
import functions.fct_metrics as metrics
import functions.fct_misc as fct_misc

logger=fct_misc.format_logger(logger)

# Argument and parameter specification
logger.info(f"Using config_flai.yaml as config file.")

with open('config/config_flai.yaml') as fp:
    cfg = load(fp, Loader=FullLoader)[os.path.basename(__file__)]


# Load input parameters
WORKING_DIR = cfg['working_dir']
METHOD='one-to-one'

GT = cfg['gt']
DETECTION = cfg['detection']

TILE_NAME=os.path.basename(DETECTION).split('.')[0]
TILES=cfg['tiles']

os.chdir(WORKING_DIR)

OUTPUT_DIR='final/flai_metrics'
fct_misc.ensure_dir_exists(OUTPUT_DIR)

written_files = {}


logger.info('Read and format input data')

gdf_detec = gpd.read_file(DETECTION)
gdf_detec['ID_DET'] = gdf_detec['FID']
gdf_detec = gdf_detec.rename(columns={"area": "area_DET"})
logger.info(f"Read detection file: {gdf_detec.shape[0]} shapes")

gdf_gt = gpd.read_file(GT)
gdf_gt['ID_GT'] = gdf_gt['fid']
gdf_gt = gdf_gt.rename(columns={"area": "area_GT"})

if gdf_gt['ID_GT'].isnull().values.any():
    logger.error('Some labels have a null identifier.')
    sys.exit(1)
elif gdf_detec['ID_DET'].isnull().values.any():
    logger.error('Some detections have a null identifier.')
    sys.exit(1)

if TILES:
    tiles=gpd.read_file(TILES)
    tile=tiles[tiles['fme_basena']==TILE_NAME]
    gdf_gt=gdf_gt.overlay(tile)

nbr_labels=gdf_gt.shape[0]
logger.info(f"Read GT file: {nbr_labels} shapes")


logger.info(f"Metrics computation")
if METHOD=='one-to-one':
    logger.info('Using the one-to-one method.')
elif METHOD=='one-to-many':
    logger.info('Using one-to-many method.')
else:
    logger.warning('Unknown method, defaulting to one-to-one.')

best_f1=0
for threshold in tqdm([i/100 for i in range(10 ,100, 5)], desc='Search for the best threshold on the IoU'):

    tp_gdf_loop, fp_gdf_loop, fn_gdf_loop = metrics.get_fractional_sets(gdf_detec, gdf_gt, iou_threshold=threshold, method=METHOD)

    # Tag predictions   
    tp_gdf_loop['tag'] = 'TP'
    fp_gdf_loop['tag'] = 'FP'
    fn_gdf_loop['tag'] = 'FN'

    # Compute metrics
    precision, recall, f1 = metrics.get_metrics(tp_gdf_loop, fp_gdf_loop, fn_gdf_loop)

    if f1 > best_f1 or threshold==0:
        tp_gdf=tp_gdf_loop
        fp_gdf=fp_gdf_loop
        fn_gdf=fn_gdf_loop

        best_precision=precision
        best_recall=recall
        best_f1=f1

        best_threshold=threshold

logger.info(f'The best threshold for the IoU is {best_threshold} in regard to the F1 score.')


TP = tp_gdf.shape[0]
FP = fp_gdf.shape[0]
FN = fn_gdf.shape[0]

logger.info(f"   TP = {TP}, FP = {FP}, FN = {FN}")
logger.info(f"   precision = {best_precision:.2f}, recall = {best_recall:.2f}, f1 = {best_f1:.2f}")
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

logger.info(f" - Compute mean Jaccard index")
iou_average = tp_gdf['IOU'].mean()
logger.info(f"   IOU average = {iou_average:.2f}")

if METHOD=='one-to-many':
    logger.info(f'{tp_with_duplicates.shape[0]-tp_gdf.shape[0]} labels are under a shared predictions with at least one other label.')


two_preds_one_label=len(tp_gdf.loc[tp_gdf.duplicated(subset=['ID_GT']), 'ID_GT'].unique().tolist())
if two_preds_one_label > 0:
    logger.warning(f'{two_preds_one_label} labels are associated with more than one prediction considered as TP.')

nbr_tagged_labels = TP + FN -two_preds_one_label
if nbr_labels != nbr_tagged_labels:
    logger.error(f'There are {nbr_labels} labels in input and {nbr_tagged_labels} labels in output.')
    filename=os.path.join(OUTPUT_DIR, 'problematic_objects.gpkg')
    logger.info(f'The list of the problematic labels in exported to {filename}.')

    if nbr_labels > nbr_tagged_labels:
        tagged_labels=tp_gdf['ID_GT'].unique().tolist() + fn_gdf['ID_GT'].unique().tolist()

        untagged_gt_gdf=gdf_gt[~gdf_gt['ID_GT'].isin(tagged_labels)]
        untagged_gt_gdf.drop(columns=['geom_GT', 'OBSTACLE'], inplace=True)

        layer_name='missing_label_tags'
        untagged_gt_gdf.to_file(filename, layer=layer_name, index=False)
        written_files[filename]=layer_name

    elif nbr_labels < nbr_tagged_labels:
        all_tagged_labels_gdf=pd.concat([tp_gdf, fn_gdf])

        duplicated_id_gt=all_tagged_labels_gdf.loc[all_tagged_labels_gdf.duplicated(subset=['ID_GT']), 'ID_GT'].unique().tolist()
        duplicated_labels=all_tagged_labels_gdf[all_tagged_labels_gdf['ID_GT'].isin(duplicated_id_gt)]
        duplicated_labels.drop(columns=['geom_GT', 'OBSTACLE', 'geom_DET', 'index_right', 'fid', 'FID', 'fme_basena'], inplace=True)

        layer_name='duplicated_label_tags'
        duplicated_labels.to_file(filename, layer=layer_name, index=False)
        written_files[filename]=layer_name


# Set the final dataframe with tagged prediction
logger.info(f"Set the final dataframe")

tagged_preds_gdf = pd.concat([tp_gdf, fp_gdf, fn_gdf])
tagged_preds_gdf = tagged_preds_gdf.drop(['index_right', 'geom_GT', 'FID', 'fid', 'OBSTACLE', 'geom_DET'], axis = 1)

feature_path = os.path.join(OUTPUT_DIR, f'tagged_predictions.gpkg')
tagged_preds_gdf.to_file(feature_path, driver='GPKG', index=False, layer=TILE_NAME + '_tags')
written_files[feature_path]=TILE_NAME + '_tags'

print()
logger.info("The following files were written. Let's check them out!")
for path in written_files.keys():
    logger.info(f'  file: {path}, layer: {written_files[path]}')
