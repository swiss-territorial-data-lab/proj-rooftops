import os
import sys
from argparse import ArgumentParser
from loguru import logger
from time import time
from tqdm import tqdm
from yaml import load, FullLoader

import geopandas as gpd
import pandas as pd

from urllib.request import urlretrieve

sys.path.insert(1, 'scripts')
import functions.fct_misc as misc

logger = misc.format_logger(logger)

# Start chronometer
tic = time()
logger.info('Starting...')

# Argument and parameter specification
parser = ArgumentParser(description="The script prepares the point cloud dataset to be processed")
parser.add_argument('config_file', type=str, help='Framework configuration file')
args = parser.parse_args()

logger.info(f"Using {args.config_file} as config file.")

with open(args.config_file) as fp:
    cfg = load(fp, Loader=FullLoader)[os.path.basename(__file__)]

WORKING_DIR = cfg['working_dir']
OUTPUT_DIR = cfg['output_dir']

SELECTED_EGIDS = cfg['selected_egids']
INPUT_TILES = cfg['input_tiles']
ROOFS = cfg['roofs']

OVERWRITE = cfg['overwrite']

os.chdir(WORKING_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

logger.info('Read dataframes')
roofs_gdf = gpd.read_file(ROOFS)
egids_df = pd.read_csv(SELECTED_EGIDS)
tiles_gdf = gpd.read_file(INPUT_TILES)

logger.info('Determine the tiles to download')
egid_list = egids_df.EGID.unique()
roof_subset_gdf = roofs_gdf.loc[roofs_gdf.EGID.isin(egid_list), ['EGID', 'geometry']].dissolve(by='EGID', as_index=False)
tile_subset_gdf = gpd.sjoin(tiles_gdf, roof_subset_gdf, lsuffix="_tile", rsuffix="_roof")

for tile_name in tqdm(tile_subset_gdf.fme_basena.unique(), desc="Download the tiles"):
    filepath = os.path.join(OUTPUT_DIR, tile_name + '.las.zip')

    if (not OVERWRITE) and (os.path.isfile(filepath) or os.path.isfile(filepath[:-4])):
        continue

    url = 'https://ge.ch/sitg/geodata/SITG/TELECHARGEMENT/LIDAR_2019/' + tile_name + '.las.zip'

    _ = urlretrieve(url, filepath)
