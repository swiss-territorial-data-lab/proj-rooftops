#!/bin/python
# -*- coding: utf-8 -*-

#  Proj rooftops
#
#      Clemence Herny 
#      Gwenaelle Salamin
#      Alessandro Cerioni 
#      Copyright (c) 2020 Republic and Canton of Geneva
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os, sys
import time
import argparse
import yaml
import numpy as np
import pandas as pd
import geopandas as gpd
import laspy
import open3d as o3d
from loguru import logger

# the following allows us to import modules from within this file's parent folder
sys.path.insert(0, '.')

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
    INPUT_DIR = cfg['input_folder']
    OUTPUT_DIR = cfg['output_folder']
    DATA_NAME = cfg['dataname']
    EGID = cfg['egid']
    DATA_TYPE = cfg['pcd_type']
    SHP_NAME = cfg['shpname']
    FILTER_CLASS = cfg['filters']['filter_class']
    CLASS_NUMBER = cfg['filters']['class_number']
    FILTER_ROOF = cfg['filters']['filter_roof']

    # Create an output directory in case it doesn't exist
    output_dir = os.path.join(OUTPUT_DIR + DATA_NAME)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    written_files = []

    # Prepare the point cloud 

    data = DATA_NAME + "." + DATA_TYPE
    data_path = os.path.join(output_dir, data)
    
    if DATA_TYPE == 'las':    
        # Open and read las file 
        input_dir = os.path.join(INPUT_DIR + DATA_NAME)
        las = laspy.read(input_dir + DATA_NAME + "." + DATA_TYPE)
        # las.header
        logger.info("3D Point cloud name: " + data)
        logger.info("Number of points: " + str(las.header.point_count))
        logger.info("Point Cloud available infos: ")
        logger.info(list(las.point_format.dimension_names))
        logger.info("Classes: ")
        logger.info(set(list(las.classification)))

        # intensity = las.intensity

        # Clip point cloud with shaepfile 
        logger.info('Read shapefile...')
        gdf_roofs = gpd.read_file(INPUT_DIR + SHP_NAME)
        dissolved = gdf_roofs.dissolve('EGID', as_index=False)
        dissolved.drop(['OBJECTID', 'ALTI_MAX', 'DATE_LEVE', 'SHAPE_AREA', 'SHAPE_LEN'], axis=1)
        alti_roof = dissolved.loc[dissolved['EGID'] == EGID, 'ALTI_MIN'].iloc[0] - 1.0             # -1 as a below buffer 

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

    feature = DATA_NAME + '_filter.csv'        
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