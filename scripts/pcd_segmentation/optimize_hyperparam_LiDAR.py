import os, sys
import argparse
from loguru import logger
from time import time
from yaml import load, FullLoader

import pandas as pd
import geopandas as gpd

import optuna

sys.path.insert(1, 'scripts')
from functions.fct_misc import format_logger, ensure_dir_exists
import functions.fct_optimization as fct_opti
import pcd_segmentation
import vectorization
import assess_results

logger=format_logger(logger)

# Define functions --------------------------

def objective(trial):
    """Define the function to be optimized by the hyperparameters

    Args:
        trial (trail object): suggested hyperparametesr for the objective optimization
    """

    # Suggest value range to test (range value not taken into account for GridSampler method)
    NUMBER_PLANES = trial.suggest_int('number_planes', 1, 10, step=1)
    DISTANCE_THERSHOLD = trial.suggest_float('distance_threshold', 0.05, 0.35, step=0.1)
    RANSAC = trial.suggest_int('ransac', 3, 5, step=1)
    ITERATIONS = trial.suggest_int('iterations', 1000, 10000, step=1000)
    EPS_PLANES = trial.suggest_float('eps_planes', 0.5, 10, step=0.5)
    MIN_POINTS_PLANES = trial.suggest_int('min_points_planes', 100, 1000, step=100)
    EPS_CLUSTERS = trial.suggest_float('eps_clusters', 0.5, 10, step=0.5)
    MIN_POINTS_CLUSTERS = trial.suggest_int('min_points_clusters', 3, 21, step=2)
    # AREA_MIN_PLANES = trial.suggest_int('min_plane_area', 5, 10, step=1)
    # AREA_MAX_OBJECTS = trial.suggest_int('max_cluster_area', 10, 30, step=1)
    # ALPHA_SHAPE = trial.suggest_float('alpha_shape', 0, 4, step=0.1)

    dict_parameters_pcd_seg={
        'number_planes':NUMBER_PLANES,
        'distance_threshold':DISTANCE_THERSHOLD,
        'ransac': RANSAC,
        'iterations': ITERATIONS,
        'eps_planes': EPS_PLANES,
        'min_points_planes': MIN_POINTS_PLANES,
        'eps_clusters': EPS_CLUSTERS,
        'min_points_clusters': MIN_POINTS_CLUSTERS,
    }

    # dict_parameters_vect={
    #     'min_plane_area': AREA_MIN_PLANES,
    #     'max_cluster_area': AREA_MAX_OBJECTS,
    #     'alpha_shape': ALPHA_SHAPE,
    # }

    print(dict_parameters_pcd_seg)
    # print(dict_parameters_vect)


    pcd_segmentation.main(WORKING_DIR, INPUT_DIR, OUTPUT_DIR,
                                  EGIDS,
                                  **dict_parameters_pcd_seg)
    all_occupation_gdf = vectorization.main(WORKING_DIR, INPUT_DIR, OUTPUT_DIR,
                                                    EGIDS, EPSG,
                                                    alpha_shape=ALPHA_SHAPE)
    f1, diff_in_labels = assess_results.main(WORKING_DIR, OUTPUT_DIR,
                                                     all_occupation_gdf, GT,
                                                     EGIDS, METHOD)

    return f1

    
tic = time()
logger.info('Starting...')

# Argument and parameter specification
parser = argparse.ArgumentParser(description="This script search the optimal parameters for the segmentation of the LiDAR point cloud.")
parser.add_argument('config_file', type=str, help='Framework configuration file')
args = parser.parse_args()

logger.info(f"Using {args.config_file} as config file.")

with open(args.config_file) as fp:
    cfg = load(fp, Loader=FullLoader)[os.path.basename(__file__)]

WORKING_DIR=cfg['working_dir']
INPUT_DIR='.'
OUTPUT_DIR='.'

EGIDS=cfg['egids']
GT=cfg['gt']
DETECTIONS=cfg['detections']
EPSG=cfg['epsg']

SEGMENTATION=cfg['parameters']['segmentation']

NBR_PLANES=SEGMENTATION['planes']['number_planes']
DISTANCE_THRESHOLD=SEGMENTATION['planes']['distance_threshold']
RANSAC=SEGMENTATION['planes']['ransac']
ITERATIONS=SEGMENTATION['planes']['iterations']
EPS_PLANES=SEGMENTATION['planes']['eps']
MIN_POINTS_PLANES=SEGMENTATION['planes']['min_points']
EPS_CLUSTERS=SEGMENTATION['clusters']['eps']
MIN_POINTS_CLUSTERS=SEGMENTATION['clusters']['min_points']
MIN_AREA_PLANES=cfg['parameters']['area_threshold']['min']
MAX_AREA_OBJECTS=cfg['parameters']['area_threshold']['max']
ALPHA_SHAPE=cfg['parameters']['alpha_shape']

METHOD=cfg['method']
VISUALISATION=cfg['visualisation']

os.chdir(WORKING_DIR)
_ = ensure_dir_exists(OUTPUT_DIR)
output_plots = ensure_dir_exists(os.path.join(OUTPUT_DIR, 'plots'))

written_files = []

logger.info('Optimization of the hyperparameters for Open3d')

study=optuna.create_study(directions=['maximize'], sampler=optuna.samplers.TPESampler(), study_name='Optimization of the Open3d hyperparameters')
study.optimize(objective, n_trials=100)

logger.info('Plot results')
written_files.append(fct_opti.plot_optimization_results(study, output_plots))

logger.info('Save the best hyperparameters')
written_files.append(fct_opti.save_best_hyperparameters(study))

print()
logger.success("The following files were written. Let's check them out!")
for written_file in written_files:
    logger.info(written_file)

# Stop chronometer  
toc = time()
logger.success(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

sys.stderr.flush()