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
parser = argparse.ArgumentParser(description="The script gets the info of the LiDAR point clouds.")
parser.add_argument('config_file', type=str, help='Framework configuration file')
args = parser.parse_args()

logger.info(f"Using {args.config_file} as config file.")
with open(args.config_file) as fp:
    cfg = load(fp, Loader=FullLoader)[os.path.basename(__file__)]


WORKING_DIR = cfg['working_dir']
INPUT_DIR = cfg['input_dir']

OVERWRITE = cfg['overwrite'] if 'overwrite' in cfg.keys() else False

OUTPUT_DIR_HTML = misc.ensure_dir_exists(os.path.join(WORKING_DIR,'processed/lidar/lidar_info'))

logger.info('Getting the list of files...')
lidar_files = glob(os.path.join(WORKING_DIR, INPUT_DIR, '*.las'))

logger.info('Processing files...')
for file in lidar_files:
    
    if '\\' in file:
        filename = file.split('\\')[-1].rstrip('.las')       
    else:
        filename = file.split('/')[-1].rstrip('.las')

    output_path_html = os.path.join(OUTPUT_DIR_HTML, filename + '.html')

    if (not os.path.isfile(output_path_html)) | OVERWRITE:
        wbt.lidar_info(
            file, 
            output_path_html, 
            density=True, 
            vlr=True, 
            geokeys=True,
        )

logger.success(f'The files were saved in the folder "{OUTPUT_DIR_HTML}".')