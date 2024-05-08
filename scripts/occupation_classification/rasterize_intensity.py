#!/bin/python
# -*- coding: utf-8 -*-
# 
#  proj-rooftops


import argparse
import os
import sys
from loguru import logger
from glob import glob
from yaml import load, FullLoader

import whitebox
# whitebox.download_wbt(linux_musl=True, reset=True)        # Uncomment if issue with GLIBC library
wbt = whitebox.WhiteboxTools()

from joblib import Parallel, delayed

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
# WBT needs absolute paths
OUTPUT_DIR = os.path.join(WORKING_DIR, cfg['output_dir'])
LIDAR_DIR = os.path.join(WORKING_DIR, cfg['lidar_dir'])

OVERWRITE = cfg['overwrite'] if 'overwrite' in cfg.keys() else False

PARAMETERS = cfg['parameters']
METHOD = PARAMETERS['method'].lower()
RES = PARAMETERS['resolution']
RADIUS = PARAMETERS['radius']
RETURNS = PARAMETERS['returns']
EXCLUDED_CLASSES = PARAMETERS['excluded_classes']

LIDAR_PROPERTY = "intensity"

N_JOBS = 20

os.makedirs(OUTPUT_DIR, exist_ok=True)

logger.info('Getting the list of files...')
lidar_files = glob(os.path.join(WORKING_DIR, LIDAR_DIR, '*.las'))

logger.info('Processing files...')
if len(lidar_files) == 0:
    logger.critical('The list of LiDAR files is empty. Please, check that you provided the right folder path.')
    sys.exit(1)

job_dict = {}
for file in lidar_files:

    output_path_tif = os.path.join(OUTPUT_DIR, 
                                 os.path.basename(file.rstrip('.las')) + f'_{METHOD}_{str(RES).replace(".", "pt")}_{str(RADIUS).replace(".", "pt")}_{RETURNS}.tif')
    
    if (not os.path.isfile(output_path_tif)) | OVERWRITE:
        if METHOD == 'idw':
            job_dict[os.path.basename(file.rstrip('.las'))] = {
                'i': file, 
                'output': output_path_tif, 
                'parameter': "intensity", 
                'returns': RETURNS,
                'exclude_cls': EXCLUDED_CLASSES,
                'radius': RADIUS,
                'resolution': RES,
            }
        elif METHOD == 'nnb':
            wbt.lidar_nearest_neighbour_gridding(
                i=file, 
                output=output_path_tif, 
                parameter="intensity", 
                returns=RETURNS,
                exclude_cls=EXCLUDED_CLASSES,
                radius=RADIUS,
                resolution=RES,
            )
        else:
            logger.error('This method of interpolation is not supported. Please, pass "idw" or "nnb" as parameter.')

if METHOD == 'idw':
    job_outcome = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(wbt.lidar_idw_interpolation)(**v) for k, v in sorted(list(job_dict.items()) )
        )

logger.success(f'The files were saved in the folder "{OUTPUT_DIR}".')