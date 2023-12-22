import os
import sys
import warnings
from argparse import ArgumentParser
from loguru import logger
from time import time
from tqdm import tqdm
from yaml import load, FullLoader

import geopandas as gpd
import pandas as pd

sys.path.insert(1, 'scripts')
import functions.fct_pcdseg as pcdseg
from functions.fct_misc import ensure_dir_exists, format_logger

# Start chronometer
tic = time()
logger.info('Starting...')

# Argument and parameter specification
parser = ArgumentParser(description='The script allows to transform 3D segmented point cloud to 2D polygon (STDL.proj-rooftops)')
parser.add_argument('config_file', type=str, help='Framework configuration file')
args = parser.parse_args()

logger.info(f"Using {args.config_file} as config file.")

with open(args.config_file) as fp:
    cfg = load(fp, Loader=FullLoader)[os.path.basename(__file__)]

# Load input parameters
WORKING_DIR = cfg['working_dir']
OUTPUT_DIR = cfg['output_dir']

PARTITION = cfg['partition']

os.chdir(WORKING_DIR)
_ = ensure_dir_exists(OUTPUT_DIR)

partition_gdf = gpd.read_file(PARTITION)
obstacles_gdf = partition_gdf[partition_gdf.occupation.astype(int) == 1].copy()

# Fuse overlapping detections
dissolved_obstacles_gdf = obstacles_gdf[['EGID', 'geometry']].dissolve('EGID').explode(index_parts=False).reset_index()
dissolved_obstacles_gdf['det_id'] = dissolved_obstacles_gdf.index
dissolved_obstacles_gdf['area'] = dissolved_obstacles_gdf.area

# Export
filepath = os.path.join(OUTPUT_DIR, 'dissolved_occupation.gpkg')
dissolved_obstacles_gdf.to_file(filepath)

logger.success(f'Done! A file was written: {filepath}')

# Stop chronometer  
toc = time()
logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

sys.stderr.flush()