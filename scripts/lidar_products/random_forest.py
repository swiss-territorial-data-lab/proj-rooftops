#!/bin/python
# -*- coding: utf-8 -*-
# 
#  proj-rooftops

import os
import sys
from argparse import ArgumentParser
from glob import glob
from loguru import logger
from tqdm import tqdm
from yaml import load, FullLoader

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_validate

sys.path.insert(1, 'scripts')
import functions.fct_misc as misc

logger = misc.format_logger(logger)

# Argument and parameter specification
parser = ArgumentParser(description="The script classify the surfaces by occupation with an random forest.")
parser.add_argument('config_file', type=str, help='Framework configuration file')
args = parser.parse_args()

logger.info(f"Using {args.config_file} as config file.")
with open(args.config_file) as fp:
    cfg = load(fp, Loader=FullLoader)[os.path.basename(__file__)]


# Define fuctions ----------------------------------

def random_forest(labels_gdf, features_df, desc=None):
    # https://towardsdatascience.com/random-forest-in-python-24d0893d51c0

    labels_gdf['OBJECTID'] = pd.to_numeric(labels_gdf.OBJECTID)

    proofed_features_df = features_df[features_df.OBJECTID.isin(labels_gdf.OBJECTID)].sort_values(by=['OBJECTID'])
    proofed_features_df.loc[proofed_features_df.nodata_overlap.isna(), 'nodata_overlap'] = 0
    filtered_labels_gdf = labels_gdf[labels_gdf.OBJECTID.isin(proofed_features_df.OBJECTID)].sort_values(by=['OBJECTID'])

    filtered_labels_gdf.loc[filtered_labels_gdf['class']=='not occupied', 'class'] = 'potentially free'
    labels_array = filtered_labels_gdf['class'].to_numpy()

    proofed_features_df.drop(columns=['OBJECTID'], inplace=True)
    features_list = proofed_features_df.columns.tolist()
    features_array = np.array(proofed_features_df)

    train_features, test_features, train_labels, test_labels = train_test_split(features_array, labels_array, test_size = 0.25, random_state = 42)

    rf = RandomForestClassifier(random_state = 42, n_estimators=10)
    rf.fit(train_features, train_labels)

    results_df = pd.DataFrame(
        {
            'predicted_class': rf.predict(test_features),
            'real_class': test_labels
        }
    )
    
    results_df['agreement'] = [1 if real == predicted else 0 
                                for real, predicted in zip(results_df.real_class, results_df.predicted_class)]

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
    importance_df.to_csv(os.path.join(OUTPUT_DIR, f'importance{"_" + desc if desc else ""}.csv'), index=False)
    
    
    scores = cross_validate(rf, features_array, labels_array, cv=10,
                                scoring=('balanced_accuracy'),
                                return_train_score=True)
    
    logger.info('Training scores:')
    [print(f'    - {round(score, 2)}') for score in scores["train_score"]]

    logger.info('Testing scores:')
    [print(f'    - {round(score, 2)}') for score in scores["test_score"]]

    return pd.DataFrame(agreement, index=[0])

# Define constants ----------------------------------

WORKING_DIR = cfg['working_dir']

GT_PATH = cfg['gt_file']
OCEN_LAYER = cfg['layer_ocen']
OCAN_LAYER = cfg['layer_ocan']

PREDICTIONS_PATH = cfg['predictions_file']
PREDICTIONS_LAYER = cfg['predictions_layer']

os.chdir(WORKING_DIR)
OUTPUT_DIR = misc.ensure_dir_exists('processed/roofs')

# Data processing --------------------------------------

logger.info('Read the files')

ocen_gt = gpd.read_file(GT_PATH, layer=OCEN_LAYER)

ocan_gt = gpd.read_file(GT_PATH, layer=OCAN_LAYER)

all_features = gpd.read_file(PREDICTIONS_PATH, layer=PREDICTIONS_LAYER)
features = all_features[
    (all_features.area>2) 
    & (all_features.status!='undefined')
].copy()
features['area'] = features.area
features.drop(columns=['tile_id', 'joined_area', 'EGID', 'ALTI_MAX', 'DATE_LEVE', 'SHAPE_AREA', 'SHAPE_LEN', 'tilepath_intensity', 'tilepath_roughness',
                         'count_i', 'count_r',
                        #  'nodata_overlap', 'min_i', 'max_r', 'max_i', 'ALTI_MIN', 'mean_i', 'median_i',
                         'status', 'reason', 'geometry'], inplace=True)

logger.info('Train and test a random forest for the OCEN')
agreement_ocen_df = random_forest(ocen_gt, features, desc='OCEN')
logger.info('Train and test a random forest for the OCAN')
agreement_ocan_df = random_forest(ocan_gt, features, desc='OCAN')

agreement_df = pd.concat([agreement_ocen_df, agreement_ocan_df], ignore_index=True)
agreement_df['office'] = ['OCEN', 'OCAN']

agreement_df.to_csv(os.path.join(OUTPUT_DIR, 'agreement_rates.csv'), index=False)