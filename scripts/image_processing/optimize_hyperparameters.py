##############################################################################
#!/bin/python
# -*- coding: utf-8 -*-
# 
#  proj-rooftops: automatic DETECTIONS of rooftops objects
#
#      Clemence Herny 
#      Gwenaelle Salamin
#      Alessandro Cerioni 
# 
# 
#  Script allowing to find the optimized hyperparameters values of the SAM model
#  Input: List of EGID, Ground Truth, hyperparameter values to explore
#  Output: Best hyperparameters values, Hyperparameters plots
#  Works along with scripts "/functions/common.py" and "/functions/fct_SAM.py"
##############################################################################


## Libraries

import os
import sys
import time
import argparse
import yaml
from loguru import logger

import pandas as pd
import matplotlib.pyplot as plt
import optuna
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# the following allows us to import modules from within this file's parent folder
sys.path.insert(1, 'scripts')
import functions.fct_misc as misc
import functions.fct_optimization as opti

import segment_images
import produce_vector_layer
import assess_results

logger = misc.format_logger(logger)
logger.remove()
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}", level="INFO")


## Functions

# Objective function for hyperparameters optimization
def objective(trial):
    """Define the function to be optimized by the hyperparameters

    Args:
        trial (trail object): suggested hyperparametesr for the objective optimization
    """

    logger.info(f"Call objective function for hyperparameters optimization")

    # Suggest value range to test (range value not taken into account for GridSampler method)
    PPS = trial.suggest_int('points_per_side', 32, 160, step=32)
    PPB = trial.suggest_int('points_per_batch', 32, 160, step=32)
    IOU_THD = trial.suggest_float('pred_iou_thresh', 0.6, 0.95, step=0.05)
    SST = trial.suggest_float('stability_score_thresh', 0.6, 0.95, step=0.05)
    SSO = trial.suggest_float('stability_score_offset', 0.0, 6.0, step=1.0)
    BOX_MNS_THD = trial.suggest_float('box_nms_thresh', 0.6, 0.95, step=0.05)
    CROP_N_LAYERS = trial.suggest_int('crop_n_layers', 0, 1, step=1)
    CROP_MNS_THD = trial.suggest_float('crop_nms_thresh', 0.6, 0.95, step=0.05)
    CROP_OVERLAP_RATIO = trial.suggest_float('crop_overlap_ratio', 0.3, 0.8, step=0.1)
    CROP_N_POINTS_DS_FACTOR = trial.suggest_int('crop_n_points_downscale_factor', 0, 10, step=1)
    MIN_MASK_REGION_AREA = trial.suggest_int('min_mask_region_area', 0, 200, step=50)

    # Create a dictionnary of the tested parameters value for a given trial
    dict_params = {
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
    print(dict_params)

    segment_images.main(WORKING_DIR, IMAGE_DIR, OUTPUT_DIR, SHP_EXT, CROP, DL_CKP, CKP_DIR, CKP, BATCH, FOREGROUND, UNIQUE, MASK_MULTI, CUSTOM_SAM, SHOW)
    produce_vector_layer.main(WORKING_DIR, ROOFS, OUTPUT_DIR, SHP_EXT, SRS)
    f1, labels_diff = assess_results.main(WORKING_DIR, OUTPUT_DIR, LABELS, DETECTIONS, EGIDS, METHOD)

    return f1

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
    IMAGE_DIR = cfg['image_dir']
    ROOFS = cfg['roofs']
    LABELS = cfg['ground_truth']
    EGIDS = cfg['egids']
    METHOD = cfg['method']
    OUTPUT_DIR = cfg['output_dir']    
    DETECTIONS = cfg['detections']
    SHP_EXT = cfg['vector_extension']
    SRS = cfg['srs']
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
        study = optuna.create_study(directions=['maximize'], sampler=optuna.samplers.GridSampler(search_space), study_name='SAM hyperparameters optimization')   
    elif SAMPLER == 'TPESampler':
        study = optuna.create_study(directions=['maximize'], sampler=optuna.samplers.TPESampler(), study_name='SAM hyperparameters optimization') 
    study.optimize(objective, n_trials=N_TRIALS)


    logger.info('Plot results')
    written_files.extend(opti.plot_optimization_results(study, output_plots))

    logger.info('Save the best hyperparameters')
    written_files.append(opti.save_best_hyperparameters(study, OUTPUT_DIR))

    print()
    logger.info("The following files were written. Let's check them out!")
    for written_file in written_files:
        logger.info(written_file)

    # Stop chronometer  
    toc = time.time()
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()