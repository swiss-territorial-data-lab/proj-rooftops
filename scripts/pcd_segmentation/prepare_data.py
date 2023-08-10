#!/bin/python
# -*- coding: utf-8 -*-

#  proj-rooftops
#
#      Clemence Herny 
#      Gwenaelle Salamin
#      Alessandro Cerioni 


import os, sys
import time
import argparse
from loguru import logger
from tqdm import tqdm
from yaml import load, FullLoader

import numpy as np
import pandas as pd
import geopandas as gpd
import laspy
import open3d as o3d
import whitebox
# whitebox.download_wbt(linux_musl=True, reset=True)        # Uncomment if issue with GLIBC library
wbt = whitebox.WhiteboxTools()

sys.path.insert(1, 'scripts')
import functions.fct_misc as fct_misc

logger = fct_misc.format_logger(logger)


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
FILTER_CLASS = FILTERS['filter_class']
CLASS_NUMBER = FILTERS['class_number']
FILTER_ROOF = FILTERS['filter_roof']
DISTANCE_BUFFER = FILTERS['distance_buffer']

VISU = cfg['visualisation']

PCD_EXT='.las'

if FILTER_CLASS:
    logger.info(f"The point cloud data will be filtered by class number: {CLASS_NUMBER}") 
if FILTER_ROOF:
    logger.info(f"The points below the min roof altitude will be filter. A buffer of {DISTANCE_BUFFER} is considered.")


os.chdir(WORKING_DIR) # WARNING: wbt requires absolute paths as input

# Create an output directory in case it doesn't exist
output_dir = fct_misc.ensure_dir_exists(os.path.join(WORKING_DIR, OUTPUT_DIR))

written_files = []

# Get the EGIDS of interest
with open(EGIDS, 'r') as src:
    egids=src.read()
egid_list=egids.split("\n")

# Get the rooftops shapes
ROOFS_DIR, ROOFS_NAME = os.path.split(SHP_ROOFS)
feature_path = os.path.join(ROOFS_DIR, ROOFS_NAME[:-4]  + "_EGID.shp")

if os.path.exists(feature_path):
    logger.info(f"File {ROOFS_NAME[:-4]}_EGID.shp already exists")
    rooftops = gpd.read_file(feature_path)
else:
    logger.info(f"File {ROOFS_NAME[:-4]}_EGID.shp does not exist")
    logger.info(f"Create it")
    gdf_roofs = gpd.read_file(os.path.join(ROOFS_DIR, ROOFS_NAME))
    gdf_roofs.drop(['OBJECTID', 'ALTI_MAX', 'DATE_LEVE', 'SHAPE_AREA', 'SHAPE_LEN'], axis=1, inplace=True)
    logger.info(f"Dissolved shapes by EGID number")
    rooftops = gdf_roofs.dissolve('EGID', as_index=False, aggfunc='min')
    rooftops.to_file(feature_path)
    written_files.append(feature_path)  
    logger.info(f"...done. A file was written: {feature_path}")

tile_delimitation=gpd.read_file(PCD_TILES)
rooftops_on_tiles=tile_delimitation.sjoin(rooftops, how='right', lsuffix='tile', rsuffix='roof')

# Get the per-EGID point cloud
for egid in tqdm(egid_list):
# Select the building shape  
    file_name='EGID_' + str(egid)
    
    shape = rooftops.loc[rooftops['EGID'] == int(egid)]

    # Write it to use it with WBT
    shape_path = os.path.join(output_dir, file_name + ".shp")
    shape.to_file(shape_path)

    # Select corresponding tiles
    useful_tiles=rooftops_on_tiles.loc[rooftops_on_tiles.EGID == int(egid), tile_delimitation.columns]
    useful_tiles['filepath']=[os.path.join(PCD_DIR, name + PCD_EXT) for name in useful_tiles.fme_basena.to_numpy()]

    clipped_inputs=str()
    for tile in useful_tiles.itertuples():
        pcd_path = os.path.join(WORKING_DIR, tile.filepath)

        # Perform .las clip with shapefile    
        clip_path = os.path.join(output_dir, tile.fme_basena + PCD_EXT)
        wbt.clip_lidar_to_polygon(pcd_path, shape_path, clip_path)  

        clipped_inputs=clipped_inputs + ', ' + clip_path

    # Join the PCD for EGID expanding over serveral tiles
    if useful_tiles.shape[0]==1:
        whole_pcd_path=clip_path
    else:
        clipped_inputs=clipped_inputs.lstrip(', ')
        whole_pcd_path=os.path.join(output_dir, file_name + PCD_EXT)
        wbt.lidar_join(
            clipped_inputs, 
            whole_pcd_path,
        )
    
    # Open and read clipped .las file 
    las = laspy.read(whole_pcd_path)

    # Filter point cloud data by class value 
    if FILTER_CLASS:
        las.points = las.points[las.classification == CLASS_NUMBER]
        
    # Convert point cloud data to numpy array
    pcd_points = np.stack((las.x, las.y, las.z)).transpose()

    # Filter point cloud with min roof altitude (remove points below the roof) 
    if FILTER_ROOF:  
        alti_roof = rooftops.loc[rooftops['EGID'] == int(egid), 'ALTI_MIN'].iloc[0] - DISTANCE_BUFFER
        pcd_filter = pcd_points[pcd_points[:, 2] > alti_roof]

    # Conversion of numpy array to Open3D format + visualisation
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_filter)
    if VISU:
        o3d.visualization.draw_geometries([pcd])

    # Save the processed point cloud data
    pcd_df = pd.DataFrame(pcd_filter, columns = ['X', 'Y', 'Z'] )
    feature_path = os.path.join(output_dir, file_name + '.csv')
    pcd_df.to_csv(feature_path)
    written_files.append(feature_path)  

    os.remove(shape_path)
    for extension in ['.cpg', '.dbf', '.prj', '.shx']:
        os.remove(shape_path.replace('.shp', extension))
    if clip_path==whole_pcd_path:
        os.remove(clip_path)
    else:
        os.remove(clip_path)
        os.remove(whole_pcd_path)

print()
logger.info("The following files were written. Let's check them out!")
for written_file in written_files:
    logger.info(written_file)

# Stop chronometer
toc = time.time()
logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

sys.stderr.flush()