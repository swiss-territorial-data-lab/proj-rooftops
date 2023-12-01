#!/bin/python
# -*- coding: utf-8 -*-

#  proj-rooftops

import argparse
import os
import sys
from loguru import logger
from time import time
from yaml import load, FullLoader

import geopandas as gpd
import pandas as pd

import joblib
import optuna

sys.path.insert(1, 'scripts')
from functions.fct_misc import format_logger, ensure_dir_exists
import functions.fct_optimization as optimization
import pcd_segmentation
import vectorization
import assess_results

logger = format_logger(logger)

# Define functions --------------------------

def objective(trial):
    """Define the function to be optimized by the hyperparameters

    Args:
        trial (trial object): suggested hyperparametesr for the objective optimization
    """

    # Suggest value range to test (range value not taken into account for GridSampler method)
    # NUMBER_PLANES = trial.suggest_int('number_planes', 3, 25, step=1)
    DISTANCE_THERSHOLD = trial.suggest_float('distance_threshold', 0.05, 0.2, step=0.005)
    RANSAC = trial.suggest_int('ransac', 3, 5, step=1)
    ITERATIONS = trial.suggest_int('iterations', 3000, 15000, step=500)
    EPS_PLANES = trial.suggest_float('eps_planes', 3, 25, step=1)
    MIN_POINTS_PLANES = trial.suggest_int('min_points_planes', 500, 2500, step=100)
    EPS_CLUSTERS = trial.suggest_float('eps_clusters', 0.6, 1.2, step=0.01)
    MIN_POINTS_CLUSTERS = trial.suggest_int('min_points_clusters', 5, 35, step=1)
    AREA_MIN_PLANES = trial.suggest_int('min_plane_area', 1, 60, step=2)
    AREA_MAX_OBJECTS = trial.suggest_int('max_cluster_area', 100, 300, step=10)
    # ALPHA_SHAPE = trial.suggest_float('alpha_shape', 0.1, 3, step=0.05)

    dict_parameters_pcd_seg = {
        # 'number_planes':NUMBER_PLANES,
        'distance_threshold':DISTANCE_THERSHOLD,
        'ransac': RANSAC,
        'iterations': ITERATIONS,
        'eps_planes': EPS_PLANES,
        'min_points_planes': MIN_POINTS_PLANES,
        'eps_clusters': EPS_CLUSTERS,
        'min_points_clusters': MIN_POINTS_CLUSTERS,
    }

    dict_parameters_vect = {
        'min_plane_area': AREA_MIN_PLANES,
        'max_cluster_area': AREA_MAX_OBJECTS,
        # 'alpha_shape': ALPHA_SHAPE,
    }

    # print(dict_parameters_pcd_seg)
    pd.DataFrame(dict_parameters_pcd_seg, index=[0]).to_csv(os.path.join(OUTPUT_DIR, 'last_parameter.csv'), index=False)
    # print(dict_parameters_vect)


    _ = pcd_segmentation.main(WORKING_DIR, INPUT_DIR, OUTPUT_DIR,
                              EGIDS,
                              #   DISTANCE_THRESHOLD, RANSAC, ITERATIONS, EPS_PLANES, MIN_POINTS_PLANES, EPS_CLUSTERS, MIN_POINTS_CLUSTERS, number_planes=NUMBER_PLANES)
                              **dict_parameters_pcd_seg)
    all_occupation_gdf, _ = vectorization.main(WORKING_DIR, INPUT_DIR, OUTPUT_DIR,
                                                    EGIDS, ROOFS, EPSG,
                                                    alpha_shape=ALPHA_SHAPE,
                                                    **dict_parameters_vect
    )
    f1, averaged_iou, _ = assess_results.main(WORKING_DIR, OUTPUT_DIR,
                                                    LABELS, all_occupation_gdf,
                                                    EGIDS, ROOFS, method=METHOD)

    return f1, averaged_iou


def callback(study, trial):
   # cf. https://stackoverflow.com/questions/62144904/python-how-to-retrive-the-best-model-from-optuna-lightgbm-study/62164601#62164601https://stackoverflow.com/questions/62144904/python-how-to-retrive-the-best-model-from-optuna-lightgbm-study/62164601#62164601 
    if (trial.number%5) == 0:
        study_path=os.path.join(OUTPUT_DIR, 'study.pkl')
        joblib.dump(study, study_path)


# Optimization of the parameters --------------------------

    
tic = time()
logger.info('Starting...')

# Argument and parameter specification
parser = argparse.ArgumentParser(description="This script search the optimal parameters for the segmentation of the LiDAR point cloud.")
parser.add_argument('config_file', type=str, help='Framework configuration file')
args = parser.parse_args()

logger.info(f"Using {args.config_file} as config file.")

with open(args.config_file) as fp:
    cfg = load(fp, Loader=FullLoader)[os.path.basename(__file__)]

WORKING_DIR = cfg['working_dir']
INPUT_DIR = '.'
OUTPUT_DIR = '.'

EGIDS = cfg['egids']
LABELS = cfg['ground_truth']
ROOFS = cfg['roofs']
EPSG = cfg['epsg']

SEGMENTATION = cfg['parameters']['segmentation']

# NBR_PLANES = SEGMENTATION['planes']['number_planes']
# DISTANCE_THRESHOLD = SEGMENTATION['planes']['distance_threshold']
# RANSAC = SEGMENTATION['planes']['ransac']
# ITERATIONS = SEGMENTATION['planes']['iterations']
# EPS_PLANES = SEGMENTATION['planes']['eps']
# MIN_POINTS_PLANES = SEGMENTATION['planes']['min_points']
# EPS_CLUSTERS = SEGMENTATION['clusters']['eps']
# MIN_POINTS_CLUSTERS = SEGMENTATION['clusters']['min_points']
# MIN_AREA_PLANES = cfg['parameters']['area_threshold']['min']
# MAX_AREA_OBJECTS = cfg['parameters']['area_threshold']['max']
ALPHA_SHAPE = cfg['parameters']['alpha_shape']

METHOD = cfg['method']
VISUALISATION = cfg['visualisation']

os.chdir(WORKING_DIR)
_ = ensure_dir_exists(OUTPUT_DIR)
output_plots = ensure_dir_exists(os.path.join(OUTPUT_DIR, 'plots'))

written_files = []
study_path = os.path.join(OUTPUT_DIR, 'study.pkl')

logger.info('Optimization of Open3d hyperparameters')

study = optuna.create_study(directions=['maximize', 'maximize'], sampler=optuna.samplers.TPESampler(), study_name='Optimization of the Open3d hyperparameters')
# study = joblib.load(open(study_path, 'rb'))
study.optimize(objective, n_trials=5, callbacks=[callback])

joblib.dump(study, study_path)
written_files.append(study_path)


targets = {0: 'f1 score', 1: 'average IoU'}

logger.info('Plot results')
written_files.extend(optimization.plot_optimization_results(study, targets, output_path=output_plots))

logger.info('Save the best hyperparameters')
written_files.append(optimization.save_best_hyperparameters(study, targets))

print()
logger.success("The following files were written. Let's check them out!")
for written_file in written_files:
    logger.info(written_file)

# Stop chronometer  
toc = time()
logger.success(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

sys.stderr.flush()