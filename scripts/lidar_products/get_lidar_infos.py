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

sys.path.insert(1, 'scripts')
from functions.fct_misc import format_logger

logger = format_logger(logger)

# Argument and parameter specification
parser = argparse.ArgumentParser(description="The script gets the info of the LiDAR point clouds.")
parser.add_argument('config_file', type=str, help='Framework configuration file')
args = parser.parse_args()

logger.info(f"Using {args.config_file} as config file.")
with open(args.config_file) as fp:
    cfg = load(fp, Loader=FullLoader)[os.path.basename(__file__)]


WORKING_DIR = cfg['working_dir']
# WBT needs absolute paths
OUTPUT_DIR = os.path.join(WORKING_DIR, cfg['output_dir'])
INPUT_DIR = os.path.join(WORKING_DIR, cfg['input_dir'])

OVERWRITE = cfg['overwrite'] if 'overwrite' in cfg.keys() else False

os.makedirs(OUTPUT_DIR, exist_ok=True)

logger.info('Getting the list of files...')
lidar_files = glob(os.path.join(INPUT_DIR, '*.las'))

logger.info('Processing files...')
for file in lidar_files:
    
    if '\\' in file:
        filename = file.split('\\')[-1].rstrip('.las')       
    else:
        filename = file.split('/')[-1].rstrip('.las')

    output_path_html = os.path.join(OUTPUT_DIR, filename + '.html')

    if (not os.path.isfile(output_path_html)) | OVERWRITE:
        wbt.lidar_info(
            file, 
            output_path_html, 
            density=True, 
            vlr=True, 
            geokeys=True,
        )

logger.success(f'The files were saved in the folder "{OUTPUT_DIR}".')