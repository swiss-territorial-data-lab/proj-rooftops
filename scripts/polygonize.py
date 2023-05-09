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
import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import shape,Polygon,MultiPolygon,mapping, Point
from shapely.ops import cascaded_union
from loguru import logger
from functions import vectorize_convex, vectorize_concave

# the following allows us to import modules from within this file's parent folder
sys.path.insert(0, '.')

logger.remove()
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}", level="INFO")


if __name__ == "__main__":

    # Start chronometer
    tic = time.time()
    logger.info('Starting...')

    # Argument and parameter specification
    parser = argparse.ArgumentParser(description="The script allows to transform 3D segmented point clouds to 2D polygons (STDL.proj-rooftops)")
    parser.add_argument('config_file', type=str, help='Framework configuration file')
    args = parser.parse_args()

    logger.info(f"Using {args.config_file} as config file.")
 
    with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)[os.path.basename(__file__)]

    # Load input parameters
    INPUT_DIR = cfg['input_folder']
    OUTPUT_DIR = cfg['output_folder']
    DATA_NAME = cfg['dataname']
    EPSG = cfg['epsg']
    AREA_THRESHOLD = cfg['area_threshold']
    VISU = cfg['visu']

    # Create an output directory in case it doesn't exist
    output_dir = os.path.join(OUTPUT_DIR + DATA_NAME + '/')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    written_files = []

    logger.info(f"Read point cloud file")
    input_dir = os.path.join(INPUT_DIR + DATA_NAME + '/' + DATA_NAME + "_all.csv")
    df = pd.read_csv(input_dir)

    # Load planes in df 
    df_planes = df[df['type']=='plane']
    planes = np.unique(df_planes['group'])
    logger.info(f"Number of planes: {np.max(planes) + 1}")

    # Vectorized planes polygons
    logger.info(f"Vectorize plane(s) polygon(s)")
    # df_poly_planes = vectorize_convex(df_planes, planes, 'plane', VISU)
    df_poly_planes = vectorize_concave(df_planes, planes, 'plane', VISU)

    # Remove element with area below threshold value and store it to add it to objects df
    small_planes = gpd.GeoDataFrame()
    for i in range(len(df_poly_planes)):
        area = df_poly_planes['area'].iloc[i] 
        if area <= AREA_THRESHOLD:
            small_planes = pd.concat([small_planes, df_poly_planes.iloc[[i]]], ignore_index=True)
            df_poly_planes = df_poly_planes.drop(i)

    logger.info(f"Create plane(s) geodataframe")
    gdf_planes = gpd.GeoDataFrame(df_poly_planes, crs='EPSG:{}'.format(EPSG), geometry='geometry')
    feature_path = DATA_NAME + "_planes.gpkg"
    gdf_planes.to_file(output_dir + feature_path)
    written_files.append(feature_path)  
    logger.info(f"...done. A file was written: {feature_path}")

    # Load remaining clusters in df 
    df_clusters = df[df['type']=='cluster']
    clusters = np.unique(df_clusters['group'])
    # clusters = clusters[clusters >= 0]                          # Remove outlier class (-1)
    logger.info(f"Number of clusters: {np.max(clusters) + 1}")

    # Vectorized clusters polygons
    logger.info(f"Vectorize object(s) polygon(s)")
    # df_poly_clusters = vectorize_convex(df_clusters, clusters, 'object', VISU)
    df_poly_clusters = vectorize_concave(df_clusters, clusters, 'object', VISU)

     # Potentially add new oject(s) previously present(s) in plane df  
    logger.info(f"Add {len(small_planes)} object(s) to the dataframe") 
    df_poly_clusters = pd.concat([df_poly_clusters, small_planes], ignore_index=True,axis=0)
    df_poly_clusters.loc[df_poly_clusters["class"] == "plane", "class"] = 'object' 

    logger.info(f"Create object(s) geodataframe")
    gdf_clusters = gpd.GeoDataFrame(df_poly_clusters, crs='EPSG:{}'.format(EPSG), geometry='geometry')
    feature_path = DATA_NAME + "_objects.gpkg"
    gdf_clusters.to_file(output_dir + feature_path)
    written_files.append(feature_path)  
    logger.info(f"...done. A file was written: {feature_path}")

    # Vectorized clusters polygons  
    logger.info(f"Compute difference (plane - objects) polygon")
    diff = gpd.GeoDataFrame()
    # ref = gdf_planes['geometry']
    diff = gdf_planes['geometry']
    # for i in range(len(df_poly_planes)):
    #     polygons = gdf_planes['geometry'].iloc[i] 
    #     boundary = gpd.GeoSeries(cascaded_union(polygons))
    #     boundary.plot(color = 'red')
    #     plt.show()
    for ii in range(len(df_poly_clusters)):
        polygons = gdf_clusters['geometry'].iloc[ii] 
        diff = diff.symmetric_difference(polygons)
        # boundary = gpd.GeoSeries(cascaded_union(polygons))
        # boundary.plot(color = 'blue')
        # plt.show()

    gdf_free = gpd.GeoDataFrame()
    gdf_free['geometry'] = diff
    gdf_free['occupation'] = 0
    gdf_free = pd.concat([gdf_free, gdf_planes['area']], axis=1)
    feature_path = DATA_NAME + "_free.gpkg"
    gdf_free.to_file(output_dir + feature_path)
    written_files.append(feature_path)  
    logger.info(f"...done. A file was written: {feature_path}")

    gdf_free.plot(color = 'green')
    plt.savefig(output_dir + DATA_NAME + "_freearea.png")
    # plt.show()

    gdf_objects = gdf_clusters.drop(['class'], axis=1) 
    gdf_objects['occupation'] = 1

    logger.info(f"Create a binary (free: 0 ; occupied: 1) rooftop occupation file")
    gdf_occupation = gpd.GeoDataFrame()
    gdf_occupation = pd.concat([gdf_free, gdf_objects], ignore_index=True)
    feature_path = DATA_NAME + "_occupation.gpkg"
    gdf_occupation.to_file(output_dir + feature_path)
    written_files.append(feature_path)  
    logger.info(f"...done. A file was written: {feature_path}")

    print()
    logger.info("The following files were written. Let's check them out!")
    for written_file in written_files:
        logger.info(written_file)
    print()

    # Stop chronometer  
    toc = time.time()
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()