#!/bin/python
# -*- coding: utf-8 -*-

#  proj-rooftops


import os
import sys
import time
import argparse
from glob import glob
from loguru import logger
from tqdm import tqdm
from yaml import load, FullLoader

import torch
from rasterio.mask import mask
from samgeo import SamGeo

# the following allows us to import modules from within this file's parent folder
sys.path.insert(1, 'scripts')
import functions.fct_misc as misc

logger = misc.format_logger(logger)


def main(WORKING_DIR, IMAGE_DIR, OUTPUT_DIR, SHP_EXT, CROP, DL_CKP, CKP_DIR, CKP, BATCH, FOREGROUND, UNIQUE, MASK_MULTI, CUSTOM_SAM, SHOW, dic={}):

    os.chdir(WORKING_DIR)

    # Create an output directory in case it doesn't exist
    misc.ensure_dir_exists(OUTPUT_DIR)

    written_files = []

    logger.info(f"Read the image file names")
    tiles = glob(os.path.join(IMAGE_DIR, '*.tif'))

    if '\\' in tiles[0]:
        tiles = [tile.replace('\\', '/') for tile in tiles]

    if CROP:
        logger.info(f"Images will be cropped with size {SIZE} and written to {IMAGE_DIR}.")

    if DL_CKP == True:
        dl_dir = os.path.join(CKP_DIR)
        if not os.path.exists(dl_dir):
            os.makedirs(dl_dir)
        ckp_dir = os.path.join(os.path.expanduser('~'), dl_dir)
    elif DL_CKP == False:
        ckp_dir = CKP_DIR
    checkpoint = os.path.join(ckp_dir, CKP)
    logger.info(f"Select pretrained model: {CKP}")

    if CUSTOM_SAM == True:
        logger.info("Use of customed SAM parameters")
        # sam_kwargs = {
        #     "points_per_side": 64,
        #     "pred_iou_thresh": 0.86,
        #     "stability_score_thresh": 0.92,
        #     "crop_n_layers": 1,
        #     "crop_n_points_downscale_factor": 1,
        #     "min_mask_region_area": 100,
        # }
        sam_kwargs = dic
    else:
        sam_kwargs = None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('kwarg', sam_kwargs)
    sam = SamGeo(
        checkpoint=checkpoint,
        model_type='vit_h',
        device=device,
        sam_kwargs=sam_kwargs,
        )

    logger.info(f"Process images:") 
    logger.info(f"- Object detection and mask saving")    
    logger.info(f"- Convert mask to vector")  

    for tile in tqdm(tiles, desc='Applying SAM to tiles', total=len(tiles)):

        # Read image 
        tilepath = tile
        
        # Crop the input image by pixel value
        if CROP:
            cropped_tilepath = misc.crop(tilepath, SIZE, IMAGE_DIR)
            written_files.append(cropped_tilepath)  
            tilepath = cropped_tilepath

        # Produce and save mask
        file_path = os.path.join(misc.ensure_dir_exists(os.path.join(OUTPUT_DIR, 'segmented_images')),
                                tile.split('/')[-1].split('.')[0] + '_segment.tif')       
        
        mask = file_path
        sam.generate(tilepath, mask, batch=BATCH, foreground=FOREGROUND, unique=UNIQUE, erosion_kernel=(3,3), mask_multiplier=MASK_MULTI)
        written_files.append(file_path)  

        if SHOW:
            file_path = os.path.join(misc.ensure_dir_exists(os.path.join(OUTPUT_DIR, 'segmented_images')),
                        tile.split('/')[-1].split('.')[0] + '_annotated.tif')   
            sam.show_masks(cmap="binary_r")
            sam.show_anns(axis="off", alpha=0.7, output=file_path)
            written_files.append(file_path)

        # Convert segmentation mask to vector layer 
        file_path = os.path.join(misc.ensure_dir_exists(os.path.join(OUTPUT_DIR, 'segmented_images')),
                    tile.split('/')[-1].split('.')[0] + '_segment')  
        if SHP_EXT == 'gpkg': 
            sam.tiff_to_gpkg(mask, file_path, simplify_tolerance=None)
        elif SHP_EXT == 'shp':       
            sam.tiff_to_vector(mask, file_path)
        written_files.append(file_path)  

    print()
    logger.info("The following files were written. Let's check them out!")
    for written_file in written_files:
        logger.info(written_file)
    print()


if __name__ == "__main__":

# Define contants ------------------------------
    # Start chronometer
    tic = time.time()
    logger.info('Starting...')

    # Argument and parameter specification
    parser = argparse.ArgumentParser(description="The script allows to transform 3D segmented point clouds to 2D polygons (STDL.proj-rooftops)")
    parser.add_argument('config_file', type=str, help='Framework configuration file')
    args = parser.parse_args()

    logger.info(f"Using {args.config_file} as config file.")
 
    with open(args.config_file) as fp:
        cfg = load(fp, Loader=FullLoader)[os.path.basename(__file__)]

    # Load input parameters
    WORKING_DIR = cfg['working_dir']
    IMAGE_DIR = cfg['image_dir']
    OUTPUT_DIR = cfg['output_dir']
    SHP_EXT = cfg['vector_extension']
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
    CUSTOM_SAM = cfg['SAM']['custom_SAM']
    SHOW = cfg['SAM']['show_masks']

    main(WORKING_DIR, IMAGE_DIR, OUTPUT_DIR, SHP_EXT, CROP, DL_CKP, CKP_DIR, CKP, BATCH, FOREGROUND, UNIQUE, MASK_MULTI, CUSTOM_SAM, SHOW)

    # Stop chronometer  
    toc = time.time()
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()