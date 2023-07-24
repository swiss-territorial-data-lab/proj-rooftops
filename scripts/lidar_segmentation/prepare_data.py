#!/bin/python
# -*- coding: utf-8 -*-

#  Proj rooftops
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
wbt = whitebox.WhiteboxTools()

# # the following allows us to import modules from within this file's parent folder
# sys.path.insert(0, '.')

logger.remove()
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}", level="INFO")


if __name__ == "__main__":

    # Start chronometer
    tic = time.time()
    logger.info('Starting...')

    # Argument and parameter specification
    parser = argparse.ArgumentParser(description="The script prepares dataset to process the rooftops project (STDL.proj-rooftops)")
    parser.add_argument('config_file', type=str, help='Framework configuration file')
    args = parser.parse_args()

    logger.info(f"Using {args.config_file} as config file.")
 
    with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)[os.path.basename(__file__)]

    # Load input parameters
    WORKING_DIR = cfg['working_dir']
    PCD_DIR = cfg['pcd_dir']
    SHP_DIR = cfg['shp_dir']
    OUTPUT_DIR = cfg['output_dir']
    PCD_NAME = cfg['pcd_name']
    PCD_EXT = cfg['pcd_ext']
    EGID = cfg['egid']
    SHP_ROOFS = cfg['shp_roofs']
    FILTER_CLASS = cfg['filters']['filter_class']
    CLASS_NUMBER = cfg['filters']['class_number']
    FILTER_ROOF = cfg['filters']['filter_roof']

    os.chdir(WORKING_DIR)

    # Create an output directory in case it doesn't exist
    output_dir = os.path.join(WORKING_DIR  + '/' + OUTPUT_DIR + '/' + PCD_NAME + '/')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    written_files = []

    # Prepare the point cloud 

    data = PCD_NAME + "." + PCD_EXT
    data_path = os.path.join(output_dir, data)
    
    if PCD_EXT == 'las':    
        # Open and read las file 
        input_data = os.path.join(WORKING_DIR  + '/' + PCD_DIR + '/' + PCD_NAME + '/' + PCD_NAME + "." + PCD_EXT)

        las = laspy.read(input_data)
        # las.header
        logger.info("3D Point cloud name: " + data)
        logger.info("Number of points: " + str(las.header.point_count))
        logger.info("Point Cloud available infos: ")
        logger.info(list(las.point_format.dimension_names))
        logger.info("Classes: ")
        logger.info(set(list(las.classification)))

        # Clip point cloud with shapefile 
        logger.info('Read shapefile...')

        feature_path = os.path.join(WORKING_DIR  + '/' + SHP_DIR + '/'  + SHP_ROOFS[:-4]  + "_EGID.shp")

        if os.path.exists(feature_path):
            logger.info(f"File {SHP_ROOFS[:-4]}_EGID.shp already exists")
            dissolved = gpd.read_file(feature_path)
        else:
            logger.info(f"File {SHP_ROOFS[:-4]}_EGID.shp does not exist")
            logger.info(f"Create it")
            gdf_roofs = gpd.read_file(WORKING_DIR  + '/' + SHP_DIR  + '/' + SHP_ROOFS)
            logger.info(f"Dissolved shapes by EGID number")
            dissolved = gdf_roofs.dissolve('EGID', as_index=False)
            dissolved.drop(['OBJECTID', 'ALTI_MAX', 'DATE_LEVE', 'SHAPE_AREA', 'SHAPE_LEN'], axis=1)
            dissolved.to_file(feature_path)
            written_files.append(feature_path)  
            logger.info(f"...done. A file was written: {feature_path}")

        logger.info(f"Select the shape for EGID {EGID}")        
        shape = dissolved.loc[dissolved['EGID'] == EGID]
        shape_data = os.path.join(WORKING_DIR  + '/' + OUTPUT_DIR + '/' + PCD_NAME + '/' + SHP_ROOFS[: -4]  + "_EGID" + str(EGID) + ".shp")
        shape.to_file(shape_data)
        written_files.append(shape_data)  
        logger.info(f"...done. A file was written: {shape_data}")

        output_data = os.path.join(WORKING_DIR  + '/' + OUTPUT_DIR + '/'  + PCD_NAME + '/' + PCD_NAME + "_clip." + PCD_EXT)

        # las clip
        logger.info(f"Clip LiDAR point cloud with shapefile")   
        wbt.clip_lidar_to_polygon(input_data, shape_data, output_data)
        written_files.append(output_data)  
        logger.info(f"...done. A file was written: {output_data}")

        # las altitude filter
        logger.info(f"Filter LiDAR points above the min altitude of the roof (by EGID)")  
        alti_roof = dissolved.loc[dissolved['EGID'] == EGID, 'ALTI_MIN'].iloc[0] - 2.0             # -1 as a below buffer 
        logger.info(f"Min altitude of the roof (+ buffer): {(alti_roof):.2f} m")

        # open las file with laspy
        las = laspy.read(output_data)

        # Filter point cloud by class value 
        if FILTER_CLASS == 'True':
            las.points = las.points[las.classification == CLASS_NUMBER]
            logger.info("Classes: ")
            logger.info(set(list(las.classification)))
        
        # Convert lidar data to numpy array
        point_data = np.stack((las.x, las.y, las.z)).transpose()
        # colors = np.stack((las.red, las.green, las.blue)).transpose()
    # Filter point cloud with min roof altitude (remove point below the roof) 
    pcd_filter = point_data[point_data[:,2] > alti_roof]

    # Conversion of numpy array to Open3D format and visualisation
    geom = o3d.geometry.PointCloud()
    geom.points = o3d.utility.Vector3dVector(pcd_filter)
    # geom.colors = o3d.utility.Vector3dVector(colors/65535)
    o3d.visualization.draw_geometries([geom])

    feature = PCD_NAME + '_filter.csv'        
    feature_path = os.path.join(output_dir, feature)
    df = pd.DataFrame(pcd_filter, columns =['X', 'Y', 'Z'] )
    df.to_csv(feature_path)
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