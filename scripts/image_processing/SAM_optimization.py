##############################################################################
#!/bin/python
# -*- coding: utf-8 -*-
# 
#  proj-rooftops: automatic detection of rooftops objects
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

import os, sys
import time
import argparse
import yaml
from loguru import logger

import pandas as pd
import matplotlib.pyplot as plt
import optuna

# the following allows us to import modules from within this file's parent folder
sys.path.insert(1, 'scripts')
import functions.fct_com as fct_com
import functions.fct_SAM as fct_SAM

logger=fct_com.format_logger(logger)
logger.remove()
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}", level="INFO")


## Functions

# Objective function for hyperparameters optimization
def objective(trial):

    logger.info(f"Call objective function for hyperparameters optimization")

    # Suggest value range to test (range value not taken into account for GridSampler method)
    PPS = trial.suggest_int('points_per_side', 0, 200, step=20)
    # PPB = trial.suggest_int('points_per_batch', 0, 200, step=20)
    IOU_THD = trial.suggest_float('pred_iou_thresh', 0, 1, step=0.5)
    SST = trial.suggest_float('stability_score_thresh', 0, 1, step=0.5)
    SSO = trial.suggest_float('stability_score_offset', 0, 1, step=0.5)
    # BOX_MNS_THD = trial.suggest_float('box_nms_thresh', 0, 5, step=1.0)
    # CROP_N_LAYERS = trial.suggest_int('crop_n_layers', 0, 10, step=1)
    # CROP_MNS_THD = trial.suggest_float('crop_nms_thresh', 0, 5, step=0.5)
    # CROP_N_POINTS_DS_FACTOR = trial.suggest_int('crop_n_points_downscale_factor', 0, 10, step=1)
    # MIN_MASK_REGION_AREA = trial.suggest_int('min_mask_region_area', 0, 500, step=20)

    # Create a dictionnary of the tested parameters value for a given trial
    dict_params = {
            "points_per_side": PPS,
            # "points_per_batch": PPB,
            "pred_iou_thresh": IOU_THD, 
            "stability_score_thresh": SST,
            "stability_score_offset": SSO, 
            # "box_nms_thresh": BOX_MNS_THD,
            # "crop_n_layers": CROP_N_LAYERS, 
            # "crop_nms_thresh": CROP_MNS_THD,
            # "crop_n_points_downscale_factor": CROP_N_POINTS_DS_FACTOR, 
            # "min_mask_region_area": MIN_MASK_REGION_AREA
            }
    
    # SAM mask + vectorization + filtering + assessment
    fct_SAM.SAM_mask(IMAGE_DIR, OUTPUT_DIR, SIZE, CROP, SHP_ROOFS, DL_CKP, CKP_DIR, CKP, BATCH, FOREGROUND, UNIQUE, MASK_MULTI, SHP_EXT, dict_params, written_files)
    fct_SAM.filter(OUTPUT_DIR, SHP_ROOFS, SRS, DETECTION, SHP_EXT, written_files)
    f1 = fct_SAM.assessment(OUTPUT_DIR, DETECTION, GT, SHP_EXT, written_files)

    return f1


## Main

if __name__ == "__main__":

# -------------------------------------
    # Start chronometer
    tic = time.time()
    logger.info('Starting...')

    # Argument and parameter specification
    parser = argparse.ArgumentParser(description="The script allows to transform 3D segmented point clouds to 2D polygons (STDL.proj-rooftops)")
    parser.add_argument('config_file', type=str, help='Framework configuration file')
    args = parser.parse_args()

    logger.info(f"Using {args.config_file} as config file.")
 
    with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)[os.path.basename(__file__)]

    # Load input parameters
    WORKING_DIR = cfg['working_dir']
    IMAGE_DIR = cfg['image_dir']
    SHP_ROOFS = cfg['shp_roofs_dir']
    # SHP_ROOFS_EGID = cfg['shp_roofs_egid_dir']
    GT = cfg['gt']
    OUTPUT_DIR = cfg['output_dir']    
    DETECTION = cfg['detection']
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
    CROP_N_POINTS_DS_FACTOR = cfg['optimization']['param_grid']['crop_n_points_downscale_factor']
    MIN_MASK_REGION_AREA = cfg['optimization']['param_grid']['min_mask_region_area']

    os.chdir(WORKING_DIR)

    # Create an output directory in case it doesn't exist
    fct_com.ensure_dir_exists(OUTPUT_DIR)

    written_files = []


    logger.info(f"Optimization of SAM hyperparemeters")
    # Set the parameter grid of hyperparameters to test


    # Define optuna study for hyperparameters optimisation
    if SAMPLER == 'GridSampler':
        logger.info(f"Set hyperparameters grid search")
        # The explicit value provided will be used for parameter optimization
        search_space = {"points_per_side": PPS,
                # "points_per_batch": PPB,
                "pred_iou_thresh": IOU_THD,
                "stability_score_thresh": SST,
                "stability_score_offset": SSO,
                # "box_nms_thresh": BOX_MNS_THD,
                # "crop_n_layers": CROP_N_LAYERS,
                # "crop_nms_thresh": CROP_MNS_THD,
                # "crop_n_points_downscale_factor": CROP_N_POINTS_DS_FACTOR,
                # "min_mask_region_area": MIN_MASK_REGION_AREA
                }
        study = optuna.create_study(directions=['maximize'], sampler=optuna.samplers.GridSampler(search_space), study_name='SAM hyperparameter optimization')   
    elif SAMPLER == 'TPESampler':
        study = optuna.create_study(directions=['maximize'], sampler=optuna.samplers.TPESampler(), study_name='SAM hyperparameter optimization') 
    study.optimize(objective, n_trials=N_TRIALS)


    # Opitmization plots
    OUTPUT_PLOTS = os.path.join(OUTPUT_DIR, 'plots')
    fct_com.ensure_dir_exists(OUTPUT_PLOTS)

    fig_importance = optuna.visualization.matplotlib.plot_param_importances(study)
    # optuna.visualization.plot_param_importances(study).show(renderer="browser")
    feature_path = os.path.join(OUTPUT_PLOTS, 'importance.png')
    plt.tight_layout()
    plt.savefig(feature_path)
    written_files.append(feature_path)

    fig_contour = optuna.visualization.matplotlib.plot_contour(study)
    feature_path = os.path.join(OUTPUT_PLOTS, 'contour.png')
    # plt.tight_layout()
    plt.savefig(feature_path)
    written_files.append(feature_path)

    fig_edf = optuna.visualization.matplotlib.plot_edf(study)
    feature_path = os.path.join(OUTPUT_PLOTS, 'edf.png')
    plt.tight_layout()
    plt.savefig(feature_path)
    written_files.append(feature_path)

    fig_history = optuna.visualization.matplotlib.plot_optimization_history(study)
    feature_path = os.path.join(OUTPUT_PLOTS, 'history.png')
    plt.tight_layout()
    plt.savefig(feature_path)
    written_files.append(feature_path)

    fig_slice = optuna.visualization.matplotlib.plot_slice(study)
    feature_path = os.path.join(OUTPUT_PLOTS, 'slice.png')
    # plt.tight_layout()
    plt.savefig(feature_path)
    written_files.append(feature_path)

    # plt.show()
    

    # Save the best hyprparameters
    logger.info(f"Best hyperparameters")
    best_trial = study.best_trial
    best_params = study.best_params
    best_val = study.best_value
    best_points_per_side = study.best_params['points_per_side']
    # best_points_per_batch = study.best_params['points_per_batch']
    best_pred_iou_thresh = study.best_params['pred_iou_thresh']
    best_stability_score_thresh = study.best_params['stability_score_thresh']
    # best_box_nms_thresh = study.best_params['box_nms_thresh']
    # best_crop_n_layers = study.best_params['crop_n_layers']
    # best_crop_nms_thresh = study.best_params['crop_nms_thresh']
    # best_crop_n_points_downscale_factor = study.best_params['crop_n_points_downscale_factor']
    # best_min_mask_region_area = study.best_params['min_mask_region_area']

    #     
    logger.info('Create dictionary of the best hyperparameters')
    best_hp_dict = {'best_trial' : best_trial,
                    'best_param' : best_params,
                    'best_value' : best_val,
                    'best_points_per_side' : best_points_per_side,
                    # 'best_points_per_batch': best_points_per_batch,
                    'best_pred_iou_thresh': best_pred_iou_thresh,
                    'best_stability_score_thresh': best_stability_score_thresh,
                    # 'best_box_nms_thresh': best_box_nms_thresh,
                    # 'best_crop_n_layers': best_crop_n_layers,
                    # 'best_crop_nms_thresh': best_crop_nms_thresh,
                    # 'best_crop_n_points_downscale_factor': best_crop_n_points_downscale_factor,
                    # 'best_min_mask_region_area': best_min_mask_region_area
                    }    

    df = pd.DataFrame(best_hp_dict, index=[0])
    feature_path = os.path.join(OUTPUT_DIR, 'best_hyperparams.txt')
    df.to_csv(feature_path, index=False, header=True)  
    written_files.append(feature_path) 


    print()
    logger.info("The following files were written. Let's check them out!")
    for written_file in written_files:
        logger.info(written_file)

    # Stop chronometer  
    toc = time.time()
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()