#!/bin/python
# -*- coding: utf-8 -*-

#  proj-rooftops


import os
import sys
from argparse import ArgumentParser
from loguru import logger
from time import time
from tqdm import tqdm
from yaml import FullLoader, load

import pandas as pd
import geopandas as gpd

# the following allows us to import modules from within this folder script of the current working directory.
sys.path.insert(1, 'scripts')
import functions.fct_misc as misc

logger = misc.format_logger(logger)

# Start chronometer
tic = time()
logger.info('Generate tiles')
logger.info('Starting...')

# Argument and parameter specification
parser = ArgumentParser(description="The script prepares the image dataset to process the rooftops project (STDL.proj-rooftops)")
parser.add_argument('config_file', type=str, help='Framework configuration file')
args = parser.parse_args()

logger.info(f"Using {args.config_file} as config file.")

with open(args.config_file) as fp:
    cfg = load(fp, Loader=FullLoader)[os.path.basename(__file__)]

# Load input parameters
WORKING_DIR = cfg['working_dir']
OUTPUT_DIR = cfg['output_dir']
ROOFS = cfg['roofs']
EGIDS = cfg['egids']
GROUND_TRUTH = cfg['ground_truth']

FILTERS=cfg['filters']
BUILDING_TYPE = FILTERS['building_type']
ROOF_INCLINATION = FILTERS['roof_inclination']
PREPARE_LABELS = FILTERS['prepare_labels']
PREPARE_TILES = FILTERS['prepare_tiles']

BUFFER = cfg['buffer']

os.chdir(WORKING_DIR)
# Create an output directory in case it doesn't exist
_ = misc.ensure_dir_exists(OUTPUT_DIR)

written_files = []

logger.info('Get the input data')

# Get the EGIDS of interest
egids = pd.read_csv(EGIDS)

if BUILDING_TYPE in ['administrative', 'industrial', 'residential']:
    logger.info(f'Only the building with the type "{BUILDING_TYPE}" are considered.')
    egids = egids[egids.roof_type==BUILDING_TYPE].copy()
elif BUILDING_TYPE != 'all':
    logger.critical('Unknown building type passed.')
    sys.exit(1)
if ROOF_INCLINATION in ['flat', 'pitched', 'mixed']:
    logger.info(f'Only the roofs with the type "{ROOF_INCLINATION}" are considered.')
    egids = egids[egids.roof_inclination == ROOF_INCLINATION].copy()
elif ROOF_INCLINATION != 'all':
    logger.critical('Unknown roof type passed.')
    sys.exit(1) 

array_egids = egids.EGID.to_numpy()
logger.info(f'    - {egids.shape[0]} selected EGIDs.')


if ('EGID' in ROOFS) | ('egid' in ROOFS):
    roofs_gdf = gpd.read_file(ROOFS)
else:
    # Get the rooftops shapes
    _, ROOFS_NAME = os.path.split(ROOFS)
    attribute = 'EGID'
    original_file_path = ROOFS
    desired_file_path = os.path.join(OUTPUT_DIR, ROOFS_NAME[:-4] + "_" + attribute + ".shp")

    roofs_gdf = misc.dissolve_by_attribute(desired_file_path, original_file_path, name=ROOFS_NAME[:-4], attribute=attribute)

roofs_gdf['EGID'] = roofs_gdf['EGID'].astype(int)

labelled_roofs_gdf = gpd.read_file(GROUND_TRUTH)
if not labelled_roofs_gdf.empty:
    roof_objects_gdf = misc.format_labels(labelled_roofs_gdf, roofs_gdf, array_egids)

logger.info('Format the labels')
supercategory = {'Aero': 'Outlets', 'Antenna': 'Antenna', 'Balcony / terrace': 'Balconies and terraces', 'Chimney': 'Outlets', 'Extensive vegetation': 'Vegetation',
                    'Intensive vegetation': 'Vegetation', 'Lawn': 'Vegetation', 'Other obstacle': 'Miscellaneous obstacles', 'Pipes': 'Pipes',
                    'Solar PV': 'Solar installations', 'Solar Thermal': 'Solar installations', 'Solar unknown': 'Solar installations', 'Window':'Window'}
roof_objects_gdf.rename(columns={'descr': 'CATEGORY'}, inplace=True)
roof_objects_gdf.drop(columns=['obj_class', 'EGID'], inplace=True)
roof_objects_gdf['SUPERCATEGORY'] = [supercategory[category] for category in roof_objects_gdf.CATEGORY.to_numpy()]

filepath = os.path.join(OUTPUT_DIR, 'ground_truth_labels.gpkg')
roof_objects_gdf.to_file(filepath)
written_files.append(filepath)

logger.success('The labels were written to file.')


print()
logger.info("The following files were written. Let's check them out!")
for written_file in written_files:
    logger.info(written_file)
print()

# Stop chronometer  
toc = time()
logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

sys.stderr.flush()