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

import time
import argparse
import yaml
import os, sys
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.ops import unary_union

from loguru import logger

# the following allows us to import modules from within this file's parent folder
sys.path.insert(0, '.')

logger.remove()
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}", level="INFO")

def union(poly1, poly2):

    new_poly = unary_union([poly1, poly2])

    return new_poly



def IOU(pol1_xy, pol2_xy):
    # Define each polygon
    polygon1_shape = pol1_xy
    polygon2_shape = pol2_xy

    # Calculate intersection and union, and tne IOU
    polygon_intersection = polygon1_shape.intersection(polygon2_shape).area
    polygon_union = polygon1_shape.area + polygon2_shape.area - polygon_intersection
    return polygon_intersection / polygon_union

if __name__ == "__main__":

    # Start chronometer
    tic = time.time()
    logger.info('Starting...')

    # Argument and parameter specification
    parser = argparse.ArgumentParser(description="The script prepares dataset to process the quarries detection project (STDL.proj-dqry)")
    parser.add_argument('config_file', type=str, help='Framework configuration file')
    args = parser.parse_args()

    logger.info(f"Using {args.config_file} as config file.")
 
    with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)[os.path.basename(__file__)]

    # Load input parameters
    INPUT_DIR = cfg['input_folder']
    OUTPUT_DIR = cfg['output_folder']
    DATA_NAME = cfg['dataname']
    SHP_NAME = cfg['shpname']
    SHP_GT = cfg['gt_shapefile']
    EGID = cfg['egid']

    # Create an output directory in case it doesn't exist
    output_dir = os.path.join(OUTPUT_DIR + DATA_NAME + '_test/')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    written_files = []

    # Open shapefiles
    gdf_gt = gpd.read_file(SHP_GT)
    logger.info(f"Read GT file: {len(gdf_gt)} shapes")

    input_dir = os.path.join(INPUT_DIR + DATA_NAME + "_test/")
    gdf_detec = gpd.read_file(input_dir + SHP_NAME)
    logger.info(f"Read detection file: {len(gdf_detec)} shapes")

    # print(gdf_gt)
    # print(gdf_detec)

    # Iterrate over dataframe to compute IOU metrics
    logger.info(f"Compute IOU metric")
    df = pd.DataFrame(index=[0])
    df_all = pd.DataFrame()
    for i, row1 in gdf_detec.iterrows():
        geom1 = gdf_detec['geometry'].iloc[i]
        for ii, row2 in gdf_gt.iterrows():
            geom2 = gdf_gt['geometry'].iloc[ii]
            iou = IOU(geom1, geom2) 

            df['ID_detection'] = i + 1
            df['area_detection'] = geom1.area
            df['ID_GT'] = ii + 1
            df['area_GT'] = geom2.area
            df['IOU'] = iou           
            df_all = pd.concat([df_all, df], ignore_index=True)

    # Keep only the better IOU score for each GT shape
    logger.info(f"")

    df = pd.DataFrame()
    df_IOU = pd.DataFrame()

    for i, row in df_all.iterrows():
        df = df_all.loc[df_all['ID_detection'] == (i - 1)].reset_index()
        list = df[df.IOU != df.IOU.max()].index
        df.drop(df.index[list], inplace=True)
        df_IOU = pd.concat([df_IOU, df], ignore_index=True).drop(columns=['index'])

    logger.info(f"Final IOU metric")
    print(df_IOU)

    iou_average = df_IOU['IOU'].mean()

    feature = DATA_NAME + '_IOU.csv'        
    feature_path = os.path.join(output_dir, feature)
    df_IOU.to_csv(feature_path)
    written_files.append(feature_path)  
    logger.info(f"...done. A file was written: {feature_path}")

    logger.info(f"Compute precision, recall, f1 and averaged IOU metrics:")
    nb_GT = len(gdf_gt)
    nb_detec = len(gdf_detec)
    TP = len(df_IOU)
    FP = nb_detec - TP
    FN = nb_GT - TP

    precision = TP / (TP + FN)
    recall = TP / (TP + FP)
    f1 = 2 * (recall * precision) / (recall + precision)

    logger.info(f"precision = {precision:.2f}")
    logger.info(f"recall = {recall:.2f}")
    logger.info(f"f1 = {f1:.2f}")
    logger.info(f"IOU average = {iou_average:.2f}")

    df = pd.DataFrame(columns =['precision', 'recall', 'f1', 'IOU'] )
    df['precision'] = precision
    df['recall'] = recall
    df['f1'] = f1
    df['IOU'] = iou_average

    feature = DATA_NAME + '_metrics.csv'        
    feature_path = os.path.join(output_dir, feature)
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