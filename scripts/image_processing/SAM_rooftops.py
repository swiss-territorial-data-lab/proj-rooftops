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
from glob import glob
from loguru import logger
from tqdm import tqdm
from yaml import load, FullLoader

import torch
import matplotlib.pyplot as plt
from rasterio.mask import mask
from rasterio.features import rasterize
from shapely.geometry import Polygon, mapping
from samgeo import SamGeo, tms_to_geotiff, get_basemaps


# the following allows us to import modules from within this file's parent folder
# sys.path.insert(0, '.')
sys.path.insert(1, 'scripts')
import functions.fct_misc as fct_misc
logger=fct_misc.format_logger(logger)
logger.remove()
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}", level="INFO")

# Define functions ------------------------------


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
        cfg = yaml.load(fp, Loader=yaml.FullLoader)[os.path.basename(__file__)]

    # Load input parameters
    WORKING_DIR=cfg['working_dir']
    IMAGE_DIR=cfg['image_dir']
    OUTPUT_DIR=cfg['output_dir']
    SHP_EXT=cfg['vector_extension']
    DL_CKP=cfg['SAM']['dl_checkpoints']
    CKP_DIR=cfg['SAM']['checkpoints_dir']
    CKP=cfg['SAM']['checkpoints']

    os.chdir(WORKING_DIR)

    # Create an output directory in case it doesn't exist
    output_dir = os.path.join(OUTPUT_DIR)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    written_files = []

    logger.info(f"Read images file name")
    tiles=glob(os.path.join(IMAGE_DIR, '*.tif'))

    if '\\' in tiles[0]:
        tiles=[tile.replace('\\', '/') for tile in tiles]
      
    for tile in tqdm(tiles, desc='Applying SAM to tiles', total=len(tiles)):
        print(os.path.basename(tile))

        logger.info(f"Read images: {os.path.basename(tile)}") 
        image = tile

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
        sam = SamGeo(
            checkpoint=checkpoint,
            model_type='vit_h',
            device=device,
            # erosion_kernel=(3, 3),
            # mask_multiplier=255,
            sam_kwargs=None,
        )

        logger.info(f"Produce and save mask")  
        file_path=os.path.join(fct_misc.ensure_dir_exists(os.path.join(OUTPUT_DIR, 'segmented_images')),
                                tile.split('/')[-1].split('.')[0] + '_segment.tif')       
        
        mask = file_path
        sam.generate(image, mask, batch=False, foreground=False, unique=True, erosion_kernel=None, mask_multiplier=255)
        written_files.append(file_path)  
        logger.info(f"...done. A file was written: {file_path}")

        file_path=os.path.join(fct_misc.ensure_dir_exists(os.path.join(OUTPUT_DIR, 'segmented_images')),
                                tile.split('/')[-1].split('.')[0] + '_colormask.tif')   
        sam.show_masks(cmap="binary_r")
        sam.show_anns(axis="off", alpha=0.7, output=file_path)
        plt.show()
        
        logger.info(f"Convert segmentation mask to vector layer")  
        if SHP_EXT == 'gpkg': 
            file_path=os.path.join(fct_misc.ensure_dir_exists(os.path.join(OUTPUT_DIR, 'segmented_images')),
                    tile.split('/')[-1].split('.')[0] + '_segment.gpkg')       
            sam.tiff_to_gpkg(mask, file_path, simplify_tolerance=None)
            written_files.append(file_path)  
            logger.info(f"...done. A file was written: {file_path}")
        elif SHP_EXT == 'shp': 
            file_path=os.path.join(fct_misc.ensure_dir_exists(os.path.join(OUTPUT_DIR, 'segmented_images')),
                    tile.split('/')[-1].split('.')[0] + '_segment.shp')        
            sam.tiff_to_vector(mask, file_path)
            written_files.append(file_path)  
            logger.info(f"...done. A file was written: {file_path}")

    print()
    logger.info("The following files were written. Let's check them out!")
    for written_file in written_files:
        logger.info(written_file)
    print()

    # Stop chronometer  
    toc = time.time()
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()