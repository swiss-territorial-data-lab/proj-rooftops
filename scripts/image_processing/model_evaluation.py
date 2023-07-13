#!/bin/python
# -*- coding: utf-8 -*-

#  Proj rooftops
#
#      Clemence Herny 
#      Gwenaelle Salamin
#      Alessandro Cerioni 



import os, sys
import time
import argparse
import yaml
import re
import torch
from glob import glob
from loguru import logger
from tqdm import tqdm

import geopandas as gpd
import optuna
from samgeo import SamGeo

# the following allows us to import modules from within this file's parent folder
sys.path.insert(1, 'scripts')
import functions.common as fct_com

logger=fct_com.format_logger(logger)
logger.remove()
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}", level="INFO")


####### Functions ####### 
# Objective function for hyperparameters optimization
def objective(trial):

    logger.info(f"Call objective function for hyperparameters optimization")

    PPS = trial.suggest_float('points_per_side', 0, 200)
    PPB = trial.suggest_float('points_per_batch', 0, 200)
    IOU_THD = trial.suggest_float('pred_iou_thresh', 0, 1)
    SST = trial.suggest_float('stability_score_thresh', 0, 1)
    SSO = trial.suggest_float('stability_score_offset', 0, 1)
    BOX_MNS_THD = trial.suggest_float('box_nms_thresh', 0, 1)
    CROP_N_LAYERS = trial.suggest_int('crop_n_layers', 0, 1)
    CROP_MNS_THD = trial.suggest_float('crop_nms_thresh', 0, 1)
    CROP_N_POINTS_DS_FACTOR = trial.suggest_int('crop_n_points_downscale_factor', 0, 50)
    MIN_MASK_REGION_AREA = trial.suggest_int('min_mask_region_area', 0, 10)

    logger.info(f"Read images file name")
    tiles=glob(os.path.join(IMAGE_DIR, '*.tif'))

    if '\\' in tiles[0]:
        tiles=[tile.replace('\\', '/') for tile in tiles]
      
    for tile in tqdm(tiles, desc='Applying SAM to tiles', total=len(tiles)):

        logger.info(f"Read images: {os.path.basename(tile)}") 
        image = tile

        # Crop the input image by pixel value
        if CROP:
            logger.info(f"Crop image with size {SIZE}") 
            image = fct_com.crop(image, SIZE, IMAGE_DIR)
            written_files.append(image)  
            logger.info(f"...done. A file was written: {image}") 

        egid = float(re.sub('[^0-9]','', os.path.basename(tile)))
        roofs=gpd.read_file(SHP_ROOF)
        shp_egid = roofs[roofs['EGID'] == egid]

        logger.info(f"Perform image segmentation with SAM")  
        if DL_CKP == True:
            dl_dir = os.path.join(CKP_DIR)
            if not os.path.exists(dl_dir):
                os.makedirs(dl_dir)
            ckp_dir = os.path.join(os.path.expanduser('~'), dl_dir)
        elif DL_CKP == False:
            ckp_dir = CKP_DIR
        checkpoint = os.path.join(ckp_dir, CKP)
        logger.info(f"Select pretrained model: {CKP}")   

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        logger.info("Test SAM hyperparameters values")

        param_grid = {
            "points_per_side": PPS,
            'points_per_batch':PPB,
            'pred_iou_thresh': IOU_THD,
            'stability_score_thresh': SST, 
            'stability_score_offset': SSO,
            'box_nms_thresh': BOX_MNS_THD,
            'crop_n_layers': CROP_N_LAYERS,
            'crop_nms_thresh': CROP_MNS_THD,
            'crop_n_points_downscale_factor': CROP_N_POINTS_DS_FACTOR,
            'min_mask_region_area': MIN_MASK_REGION_AREA
            }

        sam = SamGeo(
            checkpoint=checkpoint,
            model_type='vit_h',
            device=device,
            sam_kwargs=param_grid,
            )

        logger.info(f"Produce and save mask")  
        file_path=os.path.join(fct_com.ensure_dir_exists(os.path.join(OUTPUT_DIR, 'segmented_images')),
                                tile.split('/')[-1].split('.')[0] + '_segment.tif')       
        
        mask = file_path
        sam.generate(image, mask, batch=BATCH, foreground=FOREGROUND, unique=UNIQUE, erosion_kernel=(3,3), mask_multiplier=MASK_MULTI)
        written_files.append(file_path)  
        logger.info(f"...done. A file was written: {file_path}")

        file_path=os.path.join(fct_com.ensure_dir_exists(os.path.join(OUTPUT_DIR, 'segmented_images')),
                                tile.split('/')[-1].split('.')[0] + '_colormask.tif')   
        # sam.show_masks(cmap="binary_r")
        # sam.show_anns(axis="off", alpha=0.7, output=file_path)

        logger.info(f"Convert segmentation mask to vector layer")  
        if SHP_EXT == 'gpkg': 
            file_path=os.path.join(fct_com.ensure_dir_exists(os.path.join(OUTPUT_DIR, 'segmented_images')),
                    tile.split('/')[-1].split('.')[0] + '_segment.gpkg')       
            sam.tiff_to_gpkg(mask, file_path, simplify_tolerance=None)

            written_files.append(file_path)  
            logger.info(f"...done. A file was written: {file_path}")
        elif SHP_EXT == 'shp': 
            file_path=os.path.join(fct_com.ensure_dir_exists(os.path.join(OUTPUT_DIR, 'segmented_images')),
                    tile.split('/')[-1].split('.')[0] + '_segment.shp')        
            sam.tiff_to_vector(mask, file_path)
            written_files.append(file_path)  
            logger.info(f"...done. A file was written: {file_path}")


   ############################################# 
    # Vectorization 
    #  
    # Get the rooftops shapes
    ROOFS_DIR, ROOFS_NAME = os.path.split(SHP_ROOF_EGID)
    feature_path = os.path.join(ROOFS_DIR, ROOFS_NAME[:-4]  + "_EGID.shp")

    if os.path.exists(feature_path):
        logger.info(f"File {ROOFS_NAME[:-4]}_EGID.shp already exists")
        rooftops = gpd.read_file(feature_path)
    else:
        logger.info(f"File {ROOFS_NAME[:-4]}_EGID.shp does not exist")
        logger.info(f"Create it")
        gdf_roofs = gpd.read_file(WORKING_DIR  + '/' + ROOFS_DIR  + '/' + ROOFS_NAME)
        logger.info(f"Dissolved shapes by EGID number")
        rooftops = gdf_roofs.dissolve('EGID', as_index=False)
        rooftops.drop(['OBJECTID', 'ALTI_MAX', 'DATE_LEVE', 'SHAPE_AREA', 'SHAPE_LEN'], axis=1)
        rooftops.to_file(feature_path)
        written_files.append(feature_path)  
        logger.info(f"...done. A file was written: {feature_path}")

    # Read all the shapefile produced, filter them with rooftop extension and merge them into a single layer  
    logger.info(f"Read shapefiles' name")
    tiles=glob(os.path.join(OUTPUT_DIR + 'segmented_images',  '*.' + SHP_EXT))
    if '\\' in tiles[0]:
        tiles=[tile.replace('\\', '/') for tile in tiles]

    vector_layer = gpd.GeoDataFrame() 

    for tile in tqdm(tiles, desc='Read detection shapefiles', total=len(tiles)):

        logger.info(f"Read shapefile: {os.path.basename(tile)}")
        objects = gpd.read_file(tile)

        # Set CRS
        objects.crs = SRS
        shape_objects = objects.dissolve('value', as_index=False)
        egid = float(re.sub('[^0-9]','', os.path.basename(tile)))
        shape_egid  = rooftops[rooftops['EGID'] == egid]
        shape_egid.buffer(0)
        # shape_egid.geometry = shape_egid.geometry.buffer(1)

        fct_com.test_crs(shape_objects, shape_egid)

        logger.info(f"Filter detection by EGID location")
        selection = shape_objects.sjoin(shape_egid, how='inner', predicate="within")
        selection['area'] = selection.area 
        final_gdf = selection.drop(['index_right', 'OBJECTID', 'ALTI_MIN', 'ALTI_MAX', 'DATE_LEVE','SHAPE_AREA', 'SHAPE_LEN'], axis=1)
        feature_path = os.path.join(OUTPUT_DIR, f"tile_EGID_{int(egid)}_segment_selection.gpkg")
        final_gdf.to_file(feature_path)
        written_files.append(feature_path)  
        logger.info(f"...done. A file was written: {feature_path}")
        
        # Merge/Combine multiple shapefiles into one
        logger.info(f"Merge shapefiles together in a single vector layer")
        vector_layer = gpd.pd.concat([vector_layer, final_gdf])
    feature_path = os.path.join(OUTPUT_DIR, DETECTION + '.' + SHP_EXT)

    vector_layer.to_file(feature_path)
    written_files.append(feature_path)  
    logger.info(f"...done. A file was written: {feature_path}")


    ############################################# 
    # Assessments  
   
    # Open shapefiles
    gdf_gt = gpd.read_file(GT)
    gdf_gt = gdf_gt[gdf_gt['occupation'] == 1]
    gdf_gt['ID_GT'] = gdf_gt.index
    gdf_gt = gdf_gt.rename(columns={"area": "area_GT"})
    logger.info(f"Read GT file: {len(gdf_gt)} shapes")

    feature_path = os.path.join(OUTPUT_DIR, DETECTION + '.' + SHP_EXT)
    gdf_detec = gpd.read_file(feature_path)
    gdf_detec = gdf_detec# [gdf_detec['occupation'] == 1]
    gdf_detec['ID_DET'] = gdf_detec.index
    gdf_detec = gdf_detec.rename(columns={"area": "area_DET"})
    logger.info(f"Read detection file: {len(gdf_detec)} shapes")


    logger.info(f"Metrics computation:")
    logger.info(f" - Compute TP, FP and FN")

    tp_gdf, fp_gdf, fn_gdf = fct_com.get_fractional_sets(gdf_detec, gdf_gt)

    # Tag predictions   
    tp_gdf['tag'] = 'TP'
    fp_gdf['tag'] = 'FP'
    fn_gdf['tag'] = 'FN'

    # Compute metrics
    precision, recall, f1 = fct_com.get_metrics(tp_gdf, fp_gdf, fn_gdf)

    TP = len(tp_gdf)
    FP = len(fp_gdf)
    FN = len(fn_gdf)

    logger.info(f"   TP = {TP}, FP = {FP}, FN = {FN}")
    logger.info(f"   precision = {precision:.2f}, recall = {recall:.2f}, f1 = {f1:.2f}")
    logger.info(f" - Compute mean Jaccard index")
    iou_average = tp_gdf['IOU'].mean()
    logger.info(f"   IOU average = {iou_average:.2f}")


    return f1




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
    SHP_ROOF = cfg['shp_roofs_dir']
    SHP_ROOF_EGID = cfg['shp_roofs_egid_dir']
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
    DL_CKP = cfg['SAM']['dl_checkpoints']
    CKP_DIR = cfg['SAM']['checkpoints_dir']
    CKP = cfg['SAM']['checkpoints']
    BATCH = cfg['SAM']['batch']
    FOREGROUND = cfg['SAM']['foreground']
    UNIQUE = cfg['SAM']['unique']
    # EK = cfg['SAM']['erosion_kernel']
    MASK_MULTI = cfg['SAM']['mask_multiplier']
    N_TRIALS = cfg['SAM']['n_trials']
    PPS = cfg['SAM']['param_grid']['points_per_side']
    PPB = cfg['SAM']['param_grid']['points_per_batch']
    IOU_THD = cfg['SAM']['param_grid']['pred_iou_thresh']
    SST = cfg['SAM']['param_grid']['stability_score_thresh']
    SSO = cfg['SAM']['param_grid']['stability_score_offset']
    BOX_MNS_THD = cfg['SAM']['param_grid']['box_nms_thresh']
    CROP_N_LAYERS = cfg['SAM']['param_grid']['crop_n_layers']
    CROP_MNS_THD = cfg['SAM']['param_grid']['crop_nms_thresh']
    CROP_N_POINTS_DS_FACTOR = cfg['SAM']['param_grid']['crop_n_points_downscale_factor']
    MIN_MASK_REGION_AREA = cfg['SAM']['param_grid']['min_mask_region_area']


    os.chdir(WORKING_DIR)

    # Create an output directory in case it doesn't exist
    fct_com.ensure_dir_exists(OUTPUT_DIR)

    written_files = []

    # Set the parameter grid of hyperparameters to test
    logger.info(f"Set hyperparemeters grid")
    param_grid = {"points_per_side": PPS,
                  "points_per_batch": PPB,
                  "pred_iou_thresh": IOU_THD,
                  "stability_score_thresh": SST,
                  "stability_score_offset": SSO,
                   "box_nms_thresh": BOX_MNS_THD,
                   "crop_n_layers": CROP_N_LAYERS,
                   "crop_nms_thresh": CROP_MNS_THD,
                   "crop_n_points_downscale_factor": CROP_N_POINTS_DS_FACTOR,
                   "min_mask_region_area": MIN_MASK_REGION_AREA
                 }

    # Define optuna study for hyperparameters optimisation
    study = optuna.create_study(directions=['maximize'],sampler=optuna.samplers.GridSampler(param_grid))
    study.optimize(objective, n_trials=N_TRIALS)
    logger.info(f"Best parameters")
    study.best_params

    print()
    logger.info("The following files were written. Let's check them out!")
    for written_file in written_files:
        logger.info(written_file)

    # Stop chronometer  
    toc = time.time()
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()
