#!/bin/python
# -*- coding: utf-8 -*-
# 
#  proj-rooftops

import os
import sys
from argparse import ArgumentParser
from glob import glob
from loguru import logger
from time import time
from tqdm import tqdm
from yaml import load, FullLoader

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.tree import export_graphviz

import pickle
import pydot

sys.path.insert(1, 'scripts')
import functions.fct_misc as misc

logger = misc.format_logger(logger)

# Start chronometer
tic = time()
logger.info('Starting...')

# Argument and parameter specification
parser = ArgumentParser(description="The script classify the surfaces by occupation with an random forest.")
parser.add_argument('config_file', type=str, help='Framework configuration file')
args = parser.parse_args()

logger.info(f"Using {args.config_file} as config file.")
with open(args.config_file) as fp:
    cfg = load(fp, Loader=FullLoader)[os.path.basename(__file__)]


# Define fuctions ----------------------------------

def prepare_features(features_df):

    features_df.loc[features_df.nodata_overlap.isna(), 'nodata_overlap'] = 0
    features_df.drop(columns=['OBJECTID'], inplace=True)
    features_list = features_df.columns.tolist()
    features_array = np.array(features_df)

    return features_list, features_array


def random_forest(labels_gdf, features_df, desc=None):
    # https://towardsdatascience.com/random-forest-in-python-24d0893d51c0

    written_files = []

    labels_gdf['OBJECTID'] = pd.to_numeric(labels_gdf.OBJECTID)

    proofed_features_df = features_df[features_df.OBJECTID.isin(labels_gdf.OBJECTID)].sort_values(by=['OBJECTID'])
    filtered_labels_gdf = labels_gdf[labels_gdf.OBJECTID.isin(proofed_features_df.OBJECTID)].sort_values(by=['OBJECTID'])

    filtered_labels_gdf.loc[filtered_labels_gdf['class']=='not occupied', 'class'] = 'potentially free'
    labels_array = filtered_labels_gdf['class'].to_numpy()

    features_list, features_array = prepare_features(proofed_features_df)

    train_features, test_features, train_labels, test_labels = train_test_split(features_array, labels_array, test_size = 0.25, random_state = 42)

    rf = RandomForestClassifier(random_state = 42, n_estimators=20)
    rf.fit(train_features, train_labels)

    filepath = os.path.join(OUTPUT_DIR, f'model_RF{"_" + desc if desc else ""}.pkl')
    pickle.dump(rf, open(filepath, 'wb'))
    written_files.append(filepath)

    results_df = pd.DataFrame(
        {
            'predicted_class': rf.predict(test_features),
            'real_class': test_labels,
            'predicted_probability': [max(proba) for proba in rf.predict_proba(test_features)]
        }
    )
    results_df['agreement'] = [1 if real == predicted else 0 
                                for real, predicted in zip(results_df.real_class, results_df.predicted_class)]
    
    filepath = os.path.join(OUTPUT_DIR, f'predicted_occupancy{"_" + desc if desc else ""}.csv')
    results_df.to_csv(filepath, index=False)
    written_files.append(filepath)

    agreement = {
        'global': round(results_df.agreement.sum()/results_df.shape[0], 3),
        'occupied': round(results_df.loc[results_df.real_class=='occupied', 'agreement'].sum()
                        /results_df.loc[results_df.real_class=='occupied', 'agreement'].shape[0], 3),
        'potentially free': round(results_df.loc[results_df.real_class=='potentially free', 'agreement'].sum()
                        /results_df.loc[results_df.real_class=='potentially free', 'agreement'].shape[0], 3),
    }
    logger.info(f'Global agreement rate: {agreement["global"]}')

    importance_dict = {
        'variable': features_list,
        'importance': [round(importance*100, 1) for importance in rf.feature_importances_],
    }
    importance_df = pd.DataFrame(importance_dict).sort_values(by=['importance'], ascending=False)
    filepath = os.path.join(OUTPUT_DIR, f'importance{"_" + desc if desc else ""}.csv')
    importance_df.to_csv(filepath, index=False)
    written_files.append(filepath)
    
    scores = cross_validate(rf, features_array, labels_array, cv=10,
                                scoring=('balanced_accuracy'),
                                return_train_score=True)
    scores_df = pd.DataFrame(scores)
    
    logger.info('Scores:')
    [print(f'    - Run: {run.Index}, training: {round(run.train_score, 2)}, test: {round(run.test_score, 2)}') for run in scores_df.itertuples()]


    return pd.DataFrame(agreement, index=[0]), written_files

# Define constants ----------------------------------

WORKING_DIR = cfg['working_dir']

GT_PATH = cfg['gt_file']
OCEN_LAYER = cfg['layer_ocen']
OCAN_LAYER = cfg['layer_ocan']

PREDICTIONS_PATH = cfg['predictions_file']
PREDICTIONS_LAYER = cfg['predictions_layer']

TRAIN = cfg['train']

written_files = []

os.chdir(WORKING_DIR)
OUTPUT_DIR = misc.ensure_dir_exists('processed/roofs')

# Data processing --------------------------------------

logger.info('Read the files')

ocen_gt = gpd.read_file(GT_PATH, layer=OCEN_LAYER)

ocan_gt = gpd.read_file(GT_PATH, layer=OCAN_LAYER)

all_features_gdf = gpd.read_file(PREDICTIONS_PATH, layer=PREDICTIONS_LAYER)
features_gdf = all_features_gdf[
    (all_features_gdf.area>2) 
    & (all_features_gdf.status!='undefined')
].copy()
features_gdf['area'] = features_gdf.area
features_df = features_gdf.drop(columns=['tile_id', 'joined_area', 'EGID', 'ALTI_MAX', 'DATE_LEVE', 'SHAPE_AREA', 'SHAPE_LEN', 'tilepath_intensity', 'tilepath_roughness',
                         'count_i', 'count_r',
                        #  'nodata_overlap', 'min_i', 'max_r', 'max_i', 'ALTI_MIN', 'mean_i', 'median_i',
                         'status', 'reason', 'geometry'])

if TRAIN:
    logger.info('Train and test a random forest for the OCEN')
    agreement_ocen_df, tmp = random_forest(ocen_gt, features_df, desc='OCEN')
    written_files.extend(tmp)

    logger.info('Train and test a random forest for the OCAN')
    agreement_ocan_df, tmp = random_forest(ocan_gt, features_df, desc='OCAN')
    written_files.extend(tmp)

    agreement_df = pd.concat([agreement_ocen_df, agreement_ocan_df], ignore_index=True)
    agreement_df['office'] = ['OCEN', 'OCAN']

    agreement_df.to_csv(os.path.join(OUTPUT_DIR, 'agreement_rates.csv'), index=False)

else:
    _, features_array = prepare_features(features_df)

    rf_model_ocan = pickle.load(open(os.path.join(OUTPUT_DIR, 'model_RF_OCAN.pkl'), 'rb'))
    predictions_ocan = rf_model_ocan.predict(features_array)

    features_gdf['pred_status_ocan'] = predictions_ocan

    rf_model_ocen = pickle.load(open(os.path.join(OUTPUT_DIR, 'model_RF_OCEN.pkl'), 'rb'))
    predictions_ocen = rf_model_ocen.predict(features_array)

    features_gdf['pred_status_ocen'] = predictions_ocen

    filepath = os.path.join(OUTPUT_DIR, 'roofs.gpkg')
    features_gdf.to_file(filepath, layer='roof_occupation_by_RF')
    written_files.append(filepath)


print()
logger.info("The following files were written. Let's check them out!")
for written_file in written_files:
    logger.info(written_file)

# Stop chronometer
toc = time()
logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")