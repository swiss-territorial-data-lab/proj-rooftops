#!/bin/python
# -*- coding: utf-8 -*-
# 
#  proj-rooftops: automatic DETECTIONS of rooftops objects
#
#      Clemence Herny 
#      Gwenaelle Salamin
#      Alessandro Cerioni 
# 


import argparse
import os
import sys
from loguru import logger
from glob import glob
from yaml import load, FullLoader

import whitebox
# whitebox.download_wbt(linux_musl=True, reset=True)        # Uncomment if issue with GLIBC library
wbt = whitebox.WhiteboxTools()

sys.path.insert(1, 'scripts')
import functions.fct_misc as misc

logger = misc.format_logger(logger)

# Argument and parameter specification
parser = argparse.ArgumentParser(description="The script makes rasters from the intensity values of the point clouds in a folder.")
parser.add_argument('config_file', type=str, help='Framework configuration file')
args = parser.parse_args()

logger.info(f"Using {args.config_file} as config file.")
with open(args.config_file) as fp:
    cfg = load(fp, Loader=FullLoader)[os.path.basename(__file__)]


WORKING_DIR = cfg['working_dir']
INPUT_DIR = cfg['input_dir']

OVERWRITE = cfg['overwrite'] if 'overwrite' in cfg.keys() else False

PARAMETERS = cfg['parameters']
METHOD = PARAMETERS['method'].lower()
RES = PARAMETERS['resolution']
RADIUS = PARAMETERS['radius']
RETURNS = PARAMETERS['returns']

OUTPUT_DIR_TIF = misc.ensure_dir_exists(os.path.join(WORKING_DIR,'processed/lidar/rasterized_lidar/intensity'))

logger.info('Getting the list of files...')
lidar_files = glob(os.path.join(WORKING_DIR, INPUT_DIR, '*.las'))

logger.info('Processing files...')
for file in lidar_files:

    output_path_tif = os.path.join(OUTPUT_DIR_TIF, 
                                 os.path.basename(file.rstrip('.las')) + f'_{METHOD}_{str(RES).replace(".", "pt")}_{str(RADIUS).replace(".", "pt")}_{RETURNS}.tif')
    
    if (not os.path.isfile(output_path_tif)) | OVERWRITE:
        if METHOD == 'idw':
            wbt.lidar_idw_interpolation(
                i=file, 
                output=output_path_tif, 
                parameter="intensity", 
                returns=RETURNS,
                exclude_cls='1,2,3,5,7,9,13,15,16,19',
                radius=RADIUS,
                resolution=RES,
            )
        elif METHOD == 'nnb':
            wbt.lidar_nearest_neighbour_gridding(
                i=file, 
                output=output_path_tif, 
                parameter="intensity", 
                returns=RETURNS,
                exclude_cls='1,2,3,5,7,9,13,15,16,19',
                radius=RADIUS,
                resolution=RES,
            )
        else:
            logger.error('This method of interpolation is not supported. Please, pass "idw" or "nnb" as parameter.')

logger.success(f'The files were saved in the folder "{OUTPUT_DIR_TIF}".')