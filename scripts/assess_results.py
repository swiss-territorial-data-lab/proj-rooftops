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
from functions import *
# the following allows us to import modules from within this file's parent folder
sys.path.insert(0, '.')

logger.remove()
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}", level="INFO")



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
    output_dir = os.path.join(OUTPUT_DIR + DATA_NAME + '/')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    written_files = []

    # Open shapefiles
    gdf_gt = gpd.read_file(SHP_GT)
    logger.info(f"Read GT file: {len(gdf_gt)} shapes")

    input_dir = os.path.join(INPUT_DIR + DATA_NAME + '/' + SHP_NAME)
    gdf_detec = gpd.read_file(input_dir)
    logger.info(f"Read detection file: {len(gdf_detec)} shapes")

    the_preds_gdf = gdf_detec[gdf_detec['occupation'] == 1]
    the_labels_gdf = gdf_gt[gdf_gt['occupation'] == 1]

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
            df['occupation_detection'] = gdf_detec['occupation'].iloc[i]
            df['area_detection'] = geom1.area
            df['ID_GT'] = ii + 1
            df['occupation_GT'] = gdf_gt['occupation'].iloc[ii]
            df['area_GT'] = geom2.area
            df['IOU'] = iou           
            df_all = pd.concat([df_all, df], ignore_index=True)

    logger.info(f"Select the max IOU score for a given detection ID")
    df_IOU = pd.DataFrame()
    df_filter = df_all[df_all['occupation_detection'] == df_all['occupation_GT']].reset_index()
    df = df_filter.drop(df_filter[(df_filter['IOU'] == 0)].index)
    for i in np.unique(df['ID_detection']):
            df_ID = df[(df['ID_detection'] == i)]
            df_drop = df_ID[(df_ID['IOU'] == df_ID['IOU'].max())]
            df_IOU = pd.concat([df_IOU, df_drop], ignore_index=True).drop(columns=['index'])
    iou_average = df_IOU['IOU'].mean()
    df['IOU'] = iou_average

    logger.info(f"Compute TP, FP and FN")

    # Required a bit of adaptation work to use the assessment function developed for the object-detector   
    # tp_gdf = df_IOU[(df_IOU['occupation_detection'] == 1)]
    # fp_gdf = the_preds_gdf
    # fn_gdf = the_labels_gdf

    TP = len(df_IOU[(df_IOU['occupation_detection'] == 1)])
    FP = len(the_preds_gdf) - TP
    df_object = df_IOU[(df_IOU['occupation_GT'] == 1)]
    FN = len(the_labels_gdf) - len(np.unique(df_object['ID_GT']))
    logger.info(f"Compute precision, recall and f1-score")

    precision = TP / (TP + FP)
    recall    = TP / (TP + FN)
    f1 = 2*precision*recall/(precision+recall)

    # tp_gdf, fp_gdf, fn_gdf = get_fractional_sets(the_preds_gdf, the_labels_gdf)
    # precision, recall, f1 = get_metrics(tp_gdf, fp_gdf, fn_gdf)

    logger.info(f"Metrics results:")
    # logger.info(f"TP = {len(tp_gdf)}, FP = {len(fp_gdf)}, FN = {len(fn_gdf)}")
    logger.info(f"TP = {TP}, FP = {FP}, FN = {FN}")
    logger.info(f"precision = {precision:.2f}, recall = {recall:.2f}, f1 = {f1:.2f}")
    logger.info(f"IOU average = {iou_average:.2f}")

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