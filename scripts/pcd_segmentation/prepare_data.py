#!/bin/python
# -*- coding: utf-8 -*-

#  proj-rooftops


import argparse
import os
import sys
import time
from loguru import logger
from tqdm import tqdm
from yaml import load, FullLoader

import laspy
import geopandas as gpd
import numpy as np
import open3d as o3d
import pandas as pd

import whitebox
wbt = whitebox.WhiteboxTools()

sys.path.insert(1, 'scripts')
import functions.fct_misc as misc

logger = misc.format_logger(logger)

# Start chronometer
tic = time.time()
logger.info('Starting...')

# Argument and parameter specification
parser = argparse.ArgumentParser(description="The script prepares the point cloud dataset to be processed (STDL.proj-rooftops)")
parser.add_argument('config_file', type=str, help='Framework configuration file')
args = parser.parse_args()

logger.info(f"Using {args.config_file} as config file.")

with open(args.config_file) as fp:
    cfg = load(fp, Loader=FullLoader)[os.path.basename(__file__)]

# Load input parameters
WORKING_DIR = cfg['working_dir']
PCD_DIR = cfg['pcd_dir']
OUTPUT_DIR = cfg['output_dir']

INPUTS=cfg['inputs']
FILTERS=cfg['filters']

PCD_TILES=INPUTS['pcd_tiles']
SHP_ROOFS = INPUTS['shp_roofs']
EGIDS = INPUTS['egids']
BUILDING_TYPE = FILTERS['building_type'] if 'building_type' in FILTERS.keys() else 'all'
ROOF_INCLINATION = FILTERS['roof_inclination'] if 'roof_inclination' in FILTERS.keys() else 'all'
FILTER_CLASS = FILTERS['filter_class']
CLASS_NUMBER = FILTERS['class_number']
FILTER_ROOF = FILTERS['filter_roof']
DISTANCE_BUFFER = FILTERS['distance_buffer']

VISU = cfg['visualisation']
OVERWRITE=cfg['overwrite'] if 'overwrite' in cfg.keys() else True

PCD_EXT = '.las'

if FILTER_CLASS:
    logger.info(f"The point cloud data will be filtered by class number: {CLASS_NUMBER}") 
if FILTER_ROOF:
    logger.info(f"The points below the min roof altitude will be filter. A buffer of {DISTANCE_BUFFER} is considered.")


os.chdir(WORKING_DIR) # WARNING: wbt requires absolute paths as input

# Create an output directory in case it doesn't exist
output_dir = misc.ensure_dir_exists(os.path.join(WORKING_DIR, OUTPUT_DIR))
per_egid_dir = misc.ensure_dir_exists(os.path.join(output_dir, 'per_EGID_data'))

written_files = []

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
    egids = egids[egids.roof_inclination==ROOF_INCLINATION].copy()
elif ROOF_INCLINATION != 'all':
    logger.critical('Unknown roof type passed.')
    sys.exit(1) 

logger.info(f'{egids.shape[0]} egids will be processed.')

# Get the per-EGID rooftops shapes
ROOFS_DIR, ROOFS_NAME = os.path.split(SHP_ROOFS)
feature_path = os.path.join(OUTPUT_DIR, ROOFS_NAME[:-4]  + "_EGID.shp")

rooftops = misc.dissolve_by_attribute(feature_path, SHP_ROOFS, name=ROOFS_NAME[:-4], attribute='EGID', buffer=0.05)

# Produce light files of the selected EGIDs with the essential per-EGID information for the workflow 
completed_egids = pd.merge(egids, rooftops[['EGID', 'nbr_elem']], on='EGID')

subset_rooftops = rooftops[rooftops.EGID.isin(completed_egids.EGID.tolist())]
feature_path = os.path.join(OUTPUT_DIR, ROOFS_NAME[:-4]  + "_EGID_subset.shp")
subset_rooftops.to_file(feature_path)
written_files.append(feature_path)

feature_path = os.path.join(output_dir, 'completed_egids.csv')
completed_egids.to_csv(feature_path, index=False)
written_files.append(feature_path)  
logger.info(f"...done. A file was written: {feature_path}")

# Select LiDAR tiles
tile_delimitation = gpd.read_file(PCD_TILES)
rooftops_on_tiles = tile_delimitation.sjoin(subset_rooftops, how='right', lsuffix='tile', rsuffix='roof')

# Get the per-EGID point cloud
for egid in tqdm(egids.EGID.to_numpy()):
    # Select the building shape  
    file_name = 'EGID_' + str(egid)
    final_path = os.path.join(per_egid_dir, file_name + '.csv')

    if (not OVERWRITE) & os.path.exists(final_path):
        continue

    shape = subset_rooftops.loc[subset_rooftops['EGID'] == int(egid)].copy()

    # Write it to use it with WBT
    shape_path = os.path.join(per_egid_dir, file_name + ".shp")
    shape.to_file(shape_path)
   
    # Select corresponding tiles
    useful_tiles = rooftops_on_tiles.loc[rooftops_on_tiles.EGID == int(egid), tile_delimitation.columns].copy()
    useful_tiles['filepath'] = [os.path.join(PCD_DIR, name + PCD_EXT) for name in useful_tiles.fme_basena.to_numpy()]

    clipped_inputs = str()
    for tile in useful_tiles.itertuples():
        pcd_path = os.path.join(WORKING_DIR, tile.filepath)

        # Perform .las clip with shapefile    
        clip_path = os.path.join(per_egid_dir, file_name + '_' + tile.fme_basena + PCD_EXT)
        wbt.clip_lidar_to_polygon(pcd_path, shape_path, clip_path)  

        try:
            with laspy.open(clip_path) as src:
                nbr_points = len(src.read().points)
        except FileNotFoundError:
            whitebox.download_wbt(linux_musl=True, reset=True)              # in case of issues with the GLIBC library on Linux
            wbt.clip_lidar_to_polygon(pcd_path, shape_path, clip_path)  
            with laspy.open(clip_path) as src:
                nbr_points = len(src.read().points)

        if nbr_points > 10:
            clipped_inputs = clipped_inputs + ', ' + clip_path
 
    # Join the PCD if the EGID expands over serveral tiles
    if useful_tiles.shape[0] == 1:
        whole_pcd_path = clip_path
    else:
        clipped_inputs = clipped_inputs.lstrip(', ')
        whole_pcd_path = os.path.join(per_egid_dir, file_name + PCD_EXT)
        wbt.lidar_join(
            clipped_inputs, 
            whole_pcd_path,
        )
    
    # Open and read clipped .las file
    try:
        las = laspy.read(whole_pcd_path)
    except FileNotFoundError:
        if useful_tiles.shape[0] == 1:
            logger.error(f"Problem with the tile {tile.fme_basena} and the EGID {egid}.")
        else:
            logger.error(f"Problem with the EGID {egid}.")
        continue

    # Filter point cloud data by class value 
    if FILTER_CLASS:
        las.points = las.points[las.classification == CLASS_NUMBER]
        
    # Convert point cloud data to numpy array
    pcd_points = np.stack((las.x, las.y, las.z)).transpose()

    # Filter point cloud with min roof altitude (remove points below the roof) 
    if FILTER_ROOF:  
        alti_roof = rooftops.loc[rooftops['EGID'] == int(egid), 'ALTI_MIN'].iloc[0] - DISTANCE_BUFFER
        pcd_filter = pcd_points[pcd_points[:, 2] > alti_roof].copy()

    # Conversion of numpy array to Open3D format + visualisation
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_filter)
    if VISU:
        o3d.visualization.draw_geometries([pcd])

    # Save the processed point cloud data
    pcd_df = pd.DataFrame(pcd_filter, columns = ['X', 'Y', 'Z'] )
    pcd_df.to_csv(final_path)
    written_files.append(final_path)  

    os.remove(shape_path)
    for extension in ['.cpg', '.dbf', '.prj', '.shx']:
        os.remove(shape_path.replace('.shp', extension))
    if clip_path == whole_pcd_path:
        os.remove(clip_path)
    else:
        for path in clipped_inputs.split(', '):
            os.remove(path)
        os.remove(whole_pcd_path)

print()
logger.info("The following files were written. Let's check them out!")
for written_file in written_files:
    logger.info(written_file)

# Stop chronometer
toc = time.time()
logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

sys.stderr.flush()