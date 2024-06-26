import os
import sys
from argparse import ArgumentParser
from loguru import logger
from tqdm import tqdm
from yaml import load, FullLoader

import pandas as pd
import geopandas as gpd

sys.path.insert(1, 'scripts')
import functions.fct_metrics as metrics
import functions.fct_misc as misc

logger = misc.format_logger(logger)

# Argument and parameter specification
parser = ArgumentParser()
parser.add_argument('config_file', type=str, help='Framework configuration file')
args = parser.parse_args()

logger.info(f"Using {args.config_file} as config file.")
with open(args.config_file) as fp:
    cfg = load(fp, Loader=FullLoader)[os.path.basename(__file__)]

# Load input parameters
WORKING_DIR = cfg['working_dir']
METHOD = cfg['method']

GT = cfg['gt']
DETECTION = cfg['detection']
EPSG = cfg['epsg'] if 'epsg' in cfg.keys() else 2056

TILE_NAME = os.path.basename(DETECTION).split('.')[0]
TILES = cfg['tiles']

os.chdir(WORKING_DIR)

OUTPUT_DIR = 'final/flai_metrics'
misc.ensure_dir_exists(OUTPUT_DIR)

written_files = {}


logger.info('Read and format input data')

gdf_detec = gpd.read_file(DETECTION)
gdf_detec['ID_DET'] = gdf_detec['FID']
gdf_detec = gdf_detec.rename(columns={"area": "area_DET"})
gdf_detec = gdf_detec.to_crs(EPSG)
logger.info(f"Read detection file: {gdf_detec.shape[0]} shapes")

gdf_gt = gpd.read_file(GT)
gdf_gt['ID_GT'] = gdf_gt['fid']
gdf_gt = gdf_gt.rename(columns={"area": "area_GT"})
gdf_gt = gdf_gt.to_crs(EPSG)

if gdf_gt['ID_GT'].isnull().values.any():
    logger.error('Some labels have a null identifier.')
    sys.exit(1)
elif gdf_detec['ID_DET'].isnull().values.any():
    logger.error('Some detections have a null identifier.')
    sys.exit(1)

if TILES:
    tiles = gpd.read_file(TILES)
    tile = tiles[tiles['fme_basena'] == TILE_NAME]
    gdf_gt = gdf_gt.overlay(tile)

nbr_labels = gdf_gt.shape[0]
logger.info(f"Read GT file: {nbr_labels} shapes")


logger.info(f"Metrics computation")
if METHOD == 'one-to-one':
    logger.info('Using the one-to-one method.')
elif METHOD == 'one-to-many':
    logger.info('Using one-to-many method.')
else:
    logger.warning('Unknown method, the default value is one-to-one.')

best_f1 = 0
for threshold in tqdm([i / 100 for i in range(10, 100, 5)], desc='Search for the best threshold on the IoU'):

    tp_gdf_loop, fp_gdf_loop, fn_gdf_loop = metrics.get_fractional_sets(gdf_detec, gdf_gt, iou_threshold=threshold, method=METHOD)

    # Compute metrics
    TP = len(tp_gdf_loop)
    FP = len(fp_gdf_loop)
    FN = len(fn_gdf_loop)
    precision, recall, f1 = metrics.get_metrics(TP, FP, FN)

    if f1 > best_f1 or threshold == 0:
        tp_gdf = tp_gdf_loop
        fp_gdf = fp_gdf_loop
        fn_gdf = fn_gdf_loop

        best_precision = precision
        best_recall = recall
        best_f1 = f1

        best_threshold = threshold

logger.info(f'The best threshold for the IoU is {best_threshold} for the F1 score.')


TP = tp_gdf.shape[0]
FP = fp_gdf.shape[0]
FN = fn_gdf.shape[0]

if METHOD == 'one-to-many':
    tp_with_duplicates = tp_gdf.copy()
    dissolved_tp_gdf = tp_with_duplicates.dissolve(by=['ID_DET'], as_index=False)

    geom_DET = dissolved_tp_gdf.geometry.values.tolist()
    geom_GT = dissolved_tp_gdf['geom_GT'].values.tolist()
    iou = []
    for (i, ii) in zip(geom_DET, geom_GT):
        iou.append(metrics.intersection_over_union(i, ii))
    dissolved_tp_gdf['IOU'] = iou

    tp_gdf = dissolved_tp_gdf.copy()

    logger.info(f'{tp_with_duplicates.shape[0]-tp_gdf.shape[0]} labels share predictions with at least one other label.')

logger.info(f"   TP = {TP}, FP = {FP}, FN = {FN}")
logger.info(f"   precision = {best_precision:.2f}, recall = {best_recall:.2f}, f1 = {best_f1:.2f}")
logger.info(f" - Compute mean Jaccard index")

if TP != 0:
    iou_average = tp_gdf['IOU'].mean()
    logger.info(f"   IOU average = {iou_average:.2f}")


nbr_tagged_labels = TP + FN
filename = os.path.join(OUTPUT_DIR, 'problematic_objects.gpkg')

if os.path.exists(filename):
    os.remove(filename)
if nbr_labels != nbr_tagged_labels:
    logger.error(f'There are {nbr_labels} labels in input and {nbr_tagged_labels} labels in output.')
    logger.info(f'The list of the problematic labels in exported to {filename}.')

    if nbr_labels > nbr_tagged_labels:
        tagged_labels = tp_gdf['ID_GT'].unique().tolist() + fn_gdf['ID_GT'].unique().tolist()

        untagged_gt_gdf = gdf_gt[~gdf_gt['ID_GT'].isin(tagged_labels)]
        untagged_gt_gdf.drop(columns=['geom_GT', 'OBSTACLE'], inplace=True)

        layer_name = 'missing_label_tags'
        untagged_gt_gdf.to_file(filename, layer=layer_name, index=False)

    elif nbr_labels < nbr_tagged_labels:
        all_tagged_labels_gdf = pd.concat([tp_gdf, fn_gdf])

        duplicated_id_gt = all_tagged_labels_gdf.loc[all_tagged_labels_gdf.duplicated(subset=['ID_GT']), 'ID_GT'].unique().tolist()
        duplicated_labels = all_tagged_labels_gdf[all_tagged_labels_gdf['ID_GT'].isin(duplicated_id_gt)]
        duplicated_labels.drop(columns=['geom_GT', 'OBSTACLE', 'geom_DET', 'index_right', 'fid', 'FID', 'fme_basena'], inplace=True)

        layer_name = 'duplicated_label_tags'
        duplicated_labels.to_file(filename, layer=layer_name, index=False)
        
    written_files[filename] = layer_name


# Set the final dataframe with tagged prediction
logger.info(f"Set the final dataframe")

tagged_preds_gdf = pd.concat([tp_gdf, fp_gdf, fn_gdf])
tagged_preds_gdf = tagged_preds_gdf.drop(['index_right', 'geom_GT', 'FID', 'fid', 'geom_DET'], axis = 1)

feature_path = os.path.join(OUTPUT_DIR, f'tagged_predictions.gpkg')
tagged_preds_gdf.to_file(feature_path, driver='GPKG', index=False, layer=TILE_NAME + '_tags')
written_files[feature_path] = TILE_NAME + '_tags'

print()
logger.success("The following files were written. Let's check them out!")
for path in written_files.keys():
    logger.success(f'  file: {path}, layer: {written_files[path]}')
