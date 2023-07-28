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
import yaml
from loguru import logger

import numpy as np
import pandas as pd
import geopandas as gpd
import laspy
import open3d as o3d
import whitebox
# whitebox.download_wbt(linux_musl=True, reset=True)        # Uncomment if issue with GLIBC library
wbt = whitebox.WhiteboxTools()

sys.path.insert(1, 'scripts')
import functions.fct_com as fct_com

logger = fct_com.format_logger(logger)


if __name__ == "__main__":

    # Start chronometer
    tic = time.time()
    logger.info('Starting...')

    # Argument and parameter specification
    parser = argparse.ArgumentParser(description="The script prepares the point cloud dataset to be processed (STDL.proj-rooftops)")
    parser.add_argument('config_file', type=str, help='Framework configuration file')
    args = parser.parse_args()

    logger.info(f"Using {args.config_file} as config file.")
 
    with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)[os.path.basename(__file__)]

    # Load input parameters
    INPUTS=cfg['inputs']
    FILTERS=cfg['filters']

    WORKING_DIR = cfg['working_dir']
    PCD_DIR = cfg['pcd_dir']
    OUTPUT_DIR = cfg['output_dir']

    PCD_FILENAME=INPUTS['pcd_filename']
    PCD_NAME = os.path.splitext(PCD_FILENAME)[0]
    PCD_EXT = os.path.splitext(PCD_FILENAME)[1]
    SHP_ROOFS = INPUTS['shp_roofs']
    EGID = FILTERS['egid']
    FILTER_CLASS = FILTERS['filter_class']
    CLASS_NUMBER = FILTERS['class_number']
    FILTER_ROOF = FILTERS['filter_roof']
    DISTANCE_BUFFER = FILTERS['distance_buffer']

    VISU = cfg['visualisation']

    os.chdir(WORKING_DIR) # WARNING: wbt requires absolute paths as input

    # Create an output directory in case it doesn't exist
    file_name = PCD_NAME + '_EGID' + str(EGID)
    output_dir = fct_com.ensure_dir_exists(os.path.join(WORKING_DIR, OUTPUT_DIR, file_name))

    written_files = []


    # Get info on the pcd !!! Not mandatory, can be deleted !!!
    # Open and read las file 
    pcd_path = os.path.join(WORKING_DIR, PCD_DIR, PCD_FILENAME)
    logger.info('Read the point cloud data...')
    las = laspy.read(pcd_path)
    # las.header
    logger.info("   - 3D Point cloud name: " + PCD_FILENAME)
    logger.info("   - Number of points: " + str(las.header.point_count))
    logger.info("   - Point Cloud available infos: " + str(list(las.point_format.dimension_names)))
    # logger.info("   - Classes: " + str(set(list(las.classification))))


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

    # Select the building shape  
    logger.info(f"Select the shape for EGID {EGID}")        
    shape = rooftops.loc[rooftops['EGID'] == EGID]
    shape_path = os.path.join(output_dir, file_name + ".shp")
    shape.to_file(shape_path)
    written_files.append(shape_path)  
    logger.info(f"...done. A file was written: {shape_path}")

    # Perform .las clip with shapefile    
    logger.info(f"Clip point cloud data with shapefile")   
    clip_path = os.path.join(output_dir, file_name + PCD_EXT)
    wbt.clip_lidar_to_polygon(pcd_path, shape_path, clip_path)
    written_files.append(clip_path)  
    logger.info(f"...done. A file was written: {clip_path}")

    # Open and read clipped .las file 
    logger.info(f"Read the point cloud data for EGID {EGID}")  
    las = laspy.read(clip_path)

    # Filter point cloud data by class value 
    if FILTER_CLASS:
        logger.info(f"Filter the point cloud data by class number: {CLASS_NUMBER}")  
        las.points = las.points[las.classification == CLASS_NUMBER]
        # logger.info("Classes: ")
        # logger.info(set(list(las.classification)))
        
    # Convert point cloud data to numpy array
    pcd_points = np.stack((las.x, las.y, las.z)).transpose()

    # Conversion of numpy array to Open3D format + visualisation
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_points)
    if VISU:
        o3d.visualization.draw_geometries([pcd])

    # Filter point cloud with min roof altitude (remove points below the roof) 
    if FILTER_ROOF:
        logger.info(f"Filter points below the min roof altitude (by EGID)")  
        alti_roof = rooftops.loc[rooftops['EGID'] == EGID, 'ALTI_MIN'].iloc[0] - DISTANCE_BUFFER
        logger.info(f"Min altitude of the roof (+ buffer): {(alti_roof):.2f} m") 
        pcd_filter = pcd_points[pcd_points[:, 2] > alti_roof]

    # Conversion of numpy array to Open3D format + visualisation
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_filter)
    if VISU:
        o3d.visualization.draw_geometries([pcd])

    # Save the processed point cloud data
    logger.info(f"Save the processed point cloud data")  
    pcd_df = pd.DataFrame(pcd_filter, columns = ['X', 'Y', 'Z'] )
    feature_path = os.path.join(output_dir, file_name + '.csv')
    pcd_df.to_csv(feature_path)
    written_files.append(feature_path)  
    logger.info(f"...done. A file was written: {feature_path}")

    print()
    logger.info("The following files were written. Let's check them out!")
    for written_file in written_files:
        logger.info(written_file)

    # Stop chronometer  
    toc = time.time()
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()