#!/bin/python
# -*- coding: utf-8 -*-

#  proj-rooftops

import argparse
import os
import sys
import time
import yaml
from loguru import logger

import pandas as pd
import matplotlib.pyplot as plt
import joblib
import optuna
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# the following allows us to import modules from within this file's parent folder
sys.path.insert(1, 'scripts')
import functions.fct_misc as misc
import functions.fct_optimization as optimization

import segment_images
import produce_vector_layer
import assessment.assess_results as assess_results

logger = misc.format_logger(logger)


## Functions

# Objective function for hyperparameters optimization
def objective(trial):
    """Define the function to be optimized by the hyperparameters

    Args:
        trial (trail object): suggested hyperparametesr for the objective optimization
    """

    logger.info(f"Call objective function for hyperparameters optimization")

    # Suggest value range to test (range value not taken into account for GridSampler method)
    PPS = trial.suggest_int('points_per_side', 32, 128, step=32)
    PPB = trial.suggest_int('points_per_batch', 32, 128, step=32)
    IOU_THD = trial.suggest_float('pred_iou_thresh', 0.6, 0.95, step=0.05)
    SST = trial.suggest_float('stability_score_thresh', 0.6, 0.95, step=0.05)
    SSO = trial.suggest_float('stability_score_offset', 0.0, 6.0, step=1.0)
    BOX_MNS_THD = trial.suggest_float('box_nms_thresh', 0.0, 0.9, step=0.1)
    CROP_N_LAYERS = trial.suggest_int('crop_n_layers', 0, 1, step=1)
    CROP_MNS_THD = trial.suggest_float('crop_nms_thresh', 0.0, 0.9, step=0.1)
    CROP_OVERLAP_RATIO = trial.suggest_float('crop_overlap_ratio', 0.0, 0.7, step=0.1)
    CROP_N_POINTS_DS_FACTOR = trial.suggest_int('crop_n_points_downscale_factor', 1, 10, step=1)
    MIN_MASK_REGION_AREA = trial.suggest_int('min_mask_region_area', 0, 300, step=50)

    # Create a dictionnary of the tested parameters value for a given trial
    SAM_parameters = {
            "points_per_side": PPS,
            "points_per_batch": PPB,
            "pred_iou_thresh": IOU_THD, 
            "stability_score_thresh": SST,
            "stability_score_offset": SSO, 
            "box_nms_thresh": BOX_MNS_THD,
            "crop_n_layers": CROP_N_LAYERS, 
            "crop_nms_thresh": CROP_MNS_THD,
            "crop_overlap_ratio": CROP_OVERLAP_RATIO,
            "crop_n_points_downscale_factor": CROP_N_POINTS_DS_FACTOR, 
            "min_mask_region_area": MIN_MASK_REGION_AREA
            }
    print('')
    print(SAM_parameters)
    pd.DataFrame(SAM_parameters, index=[0]).to_csv(os.path.join(OUTPUT_DIR, 'last_parameter.csv'), index=False)

    segment_images.main(WORKING_DIR, IMAGE_DIR, OUTPUT_DIR, SHP_EXT, CROP, DL_CKP, CKP_DIR, CKP, BATCH, FOREGROUND, UNIQUE, MASK_MULTI, CUSTOM_SAM, SHOW, SAM_parameters)
    produce_vector_layer.main(WORKING_DIR, EGIDS, ROOFS, OUTPUT_DIR, SHP_EXT, CRS)
    metrics_df, labels_diff = assess_results.main(WORKING_DIR, OUTPUT_DIR, LABELS, DETECTIONS, EGIDS,
                                             method=METHOD, threshold=THRESHOLD,
                                             roofs=ROOFS,
                                             object_parameters=OBJECT_PARAMETERS, ranges=RANGES,
                                             additional_metrics=ADDITIONAL_METRICS, visualisation=VISU)

    print('')
    # To Do: Config metrics choice in config file
    f1 = metrics_df['f1'].loc[(metrics_df['attribute']=='EGID') & (metrics_df['value']=='ALL')].values[0] 
    # iou = metrics_df['IoU_mean'].loc[(metrics_df['attribute']=='EGID') & (metrics_df['value']=='ALL')].values[0] 
    iou = metrics_df['IoU_median'].loc[(metrics_df['attribute']=='EGID') & (metrics_df['value']=='ALL')].values[0] 

    return f1, iou


def callback(study, trial):
   # cf. https://stackoverflow.com/questions/62144904/python-how-to-retrive-the-best-model-from-optuna-lightgbm-study/62164601#62164601https://stackoverflow.com/questions/62144904/python-how-to-retrive-the-best-model-from-optuna-lightgbm-study/62164601#62164601 
    if (trial.number%5) == 0:
        study_path=os.path.join(OUTPUT_DIR, 'study.pkl')
        joblib.dump(study, study_path)


## Main

if __name__ == "__main__":

# -------------------------------------
    # Start chronometer
    tic = time.time()
    logger.info('Starting...')

    # Argument and parameter specification
    parser = argparse.ArgumentParser(description="The script allows to optimize the hyperparameters of the algorithm to detect objects on rooftops.")
    parser.add_argument('config_file', type=str, help='Framework configuration file')
    args = parser.parse_args()

    logger.info(f"Using {args.config_file} as config file.")
 
    with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)[os.path.basename(__file__)]

    # Load input parameters
    WORKING_DIR = cfg['working_dir']
    OUTPUT_DIR = cfg['output_dir'] 
    IMAGE_DIR = cfg['image_dir']
    ROOFS = cfg['roofs']
    LABELS = cfg['ground_truth']
    EGIDS = cfg['egids']   
    DETECTIONS = cfg['detections']
    SHP_EXT = cfg['vector_extension']
    CRS = cfg['crs']
    METHOD = cfg['method']
    THRESHOLD = cfg['threshold']
    ADDITIONAL_METRICS = cfg['additional_metrics'] if 'additional_metrics' in cfg.keys() else False
    OBJECT_PARAMETERS = cfg['object_attributes']['parameters']
    AREA_RANGES = cfg['object_attributes']['area_ranges'] 
    DISTANCE_RANGES = cfg['object_attributes']['distance_ranges'] 
    RANGES = [AREA_RANGES] + [DISTANCE_RANGES] 
    VISU = cfg['visualisation'] if 'visualisation' in cfg.keys() else False
    CROP = cfg['image_crop']['enable']
    if CROP == True:
        SIZE = cfg['image_crop']['size']
    else:
        CROP = None
        SIZE = 0
    DL_CKP = cfg['SAM']['dl_checkpoints']
    CKP_DIR = cfg['SAM']['checkpoints_dir']
    CKP = cfg['SAM']['checkpoints']
    BATCH = cfg['SAM']['batch']
    FOREGROUND = cfg['SAM']['foreground']
    UNIQUE = cfg['SAM']['unique']
    # EK = cfg['SAM']['erosion_kernel']
    MASK_MULTI = cfg['SAM']['mask_multiplier']
    CUSTOM_SAM = cfg['SAM']['custom_SAM']
    SHOW = cfg['SAM']['show_masks']
    N_TRIALS = cfg['optimization']['n_trials']
    SAMPLER = cfg['optimization']['sampler']
    PPS = cfg['optimization']['param_grid']['points_per_side']
    PPB = cfg['optimization']['param_grid']['points_per_batch']
    IOU_THD = cfg['optimization']['param_grid']['pred_iou_thresh']
    SST = cfg['optimization']['param_grid']['stability_score_thresh']
    SSO = cfg['optimization']['param_grid']['stability_score_offset']
    BOX_MNS_THD = cfg['optimization']['param_grid']['box_nms_thresh']
    CROP_N_LAYERS = cfg['optimization']['param_grid']['crop_n_layers']
    CROP_MNS_THD = cfg['optimization']['param_grid']['crop_nms_thresh']
    CROP_OVERLAP_RATIO = cfg['optimization']['param_grid']['crop_overlap_ratio']
    CROP_N_POINTS_DS_FACTOR = cfg['optimization']['param_grid']['crop_n_points_downscale_factor']
    MIN_MASK_REGION_AREA = cfg['optimization']['param_grid']['min_mask_region_area']

    os.chdir(WORKING_DIR)

    # Create an output directory in case it doesn't exist
    misc.ensure_dir_exists(OUTPUT_DIR)
    output_plots = misc.ensure_dir_exists(os.path.join(OUTPUT_DIR, 'plots'))

    written_files = []
    study_path = os.path.join(OUTPUT_DIR, 'study.pkl')

    logger.info(f"Optimization of SAM hyperparemeters")

    # Set the parameter grid of hyperparameters to test
    # Define optuna study for hyperparameters optimisation
    if SAMPLER == 'GridSampler':
        logger.info(f"Set hyperparameters grid search")
        # The explicit value provided will be used for parameter optimization
        search_space = {"points_per_side": PPS,
                "points_per_batch": PPB,
                "pred_iou_thresh": IOU_THD,
                "stability_score_thresh": SST,
                "stability_score_offset": SSO,
                "box_nms_thresh": BOX_MNS_THD,
                "crop_n_layers": CROP_N_LAYERS,
                "crop_nms_thresh": CROP_MNS_THD,
                "crop_overlap_ratio": CROP_OVERLAP_RATIO,
                "crop_n_points_downscale_factor": CROP_N_POINTS_DS_FACTOR,
                "min_mask_region_area": MIN_MASK_REGION_AREA
                }
        study = optuna.create_study(directions=['maximize', 'maximize'], sampler=optuna.samplers.GridSampler(search_space), study_name='SAM hyperparameters optimization')   
    elif SAMPLER == 'TPESampler':
        study = optuna.create_study(directions=['maximize', 'maximize'], sampler=optuna.samplers.TPESampler(), study_name='SAM hyperparameters optimization') 
    # study.optimize(objective, n_trials=N_TRIALS, callbacks=[callback])
    study.optimize(objective, n_trials=N_TRIALS)

    joblib.dump(study, study_path)
    written_files.append(study_path)

    targets = {0: 'f1 score', 1: 'median IoU'}

    logger.info('Save the best hyperparameters')
    written_files.append(optimization.save_best_hyperparameters(study, targets, output_plots))

    logger.info('Plot results')
    written_files.extend(optimization.plot_optimization_results(study, targets, output_plots))

    print()
    logger.info("The following files were written. Let's check them out!")
    for written_file in written_files:
        logger.info(written_file)

    # Stop chronometer  
    toc = time.time()
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()