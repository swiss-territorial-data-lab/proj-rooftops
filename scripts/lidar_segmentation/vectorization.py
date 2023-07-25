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
import matplotlib.pyplot as plt
from shapely.ops import unary_union

sys.path.insert(1, 'scripts')
import functions.fct_misc as fct_misc
import functions.fct_lidar_segmentation as fct_seg

logger = fct_misc.format_logger(logger)
logger.remove()
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}", level="INFO")


if __name__ == "__main__":

    # Start chronometer
    tic = time.time()
    logger.info('Starting...')

    # Argument and parameter specification
    parser = argparse.ArgumentParser(description='The script allows to transform 3D segmented point cloud to 2D polygon (STDL.proj-rooftops)')
    parser.add_argument('config_file', type=str, help='Framework configuration file')
    args = parser.parse_args()

    logger.info(f"Using {args.config_file} as config file.")
 
    with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)[os.path.basename(__file__)]

    # Load input parameters
    WORKING_DIR = cfg['working_dir']
    INPUT_DIR = cfg['input_dir']
    OUTPUT_DIR = cfg['output_dir']
    PCD_NAME = cfg['pcd_name']
    EGID = cfg['egid']
    EPSG = cfg['epsg']
    AREA_THRESHOLD = cfg['area_threshold']
    ALPHA = cfg['alpha_shape']
    VISU = cfg['visualisation']

    os.chdir(WORKING_DIR)

    file_name = PCD_NAME + '_EGID' + str(EGID)
    output_dir = os.path.join(OUTPUT_DIR, file_name)
    # Create an output directory in case it doesn't exist
    fct_misc.ensure_dir_exists(output_dir)

    written_files = []

    logger.info("Read point cloud data file")
    input_dir = os.path.join(INPUT_DIR, file_name, file_name + "_segmented.csv")
    pcd_df = pd.read_csv(input_dir)

    # Create a plane dataframe
    plane_df = pcd_df[pcd_df['type'] == 'plane']
    plane = np.unique(plane_df['group'])
    print("")
    logger.info(f"Number of plane(s): {np.max(plane) + 1}")

    # Plane vectorization
    logger.info(f"Vectorize plane")
    plane_vec_gdf = fct_seg.vectorize_concave(plane_df, plane, EPSG, ALPHA, 'plane', VISU)
    # plane_vec_gdf = fct_seg.vectorize_convex(plane_df, plane, EPSG, 'plane', VISU) 

    # Load clusters in a dataframe 
    cluster_df = pcd_df[pcd_df['type'] == 'cluster']
    cluster = np.unique(cluster_df['group'])
    cluster = cluster[cluster >= 0]                                         # Remove outlier class (-1): none classified points
    print("")
    logger.info(f"Number of cluster(s): {np.max(cluster) + 1}")

    # Cluster vectorisation
    logger.info(f"Vectorize object")
    cluster_vec_gdf = fct_seg.vectorize_concave(cluster_df, cluster, EPSG, ALPHA, 'object', VISU)
    # cluster_vec_gdf = fct_seg.vectorize_convex(cluster_df, cluster, EPSG, 'object', VISU)

    # Filtering: identify and remove plane element with an area below threshold value to be added to the object gdf
    # An object can be classified as a plane in 'plane_segmentation.py' script. We can add an area thd value for which the plane can be considered to belong to a rooftop object
    small_plane_gdf = plane_vec_gdf[plane_vec_gdf['area'] <= AREA_THRESHOLD]
    plane_vec_gdf = plane_vec_gdf.drop(small_plane_gdf.index)

    # If it exists, add cluster previously classified as plane to the object class 
    if small_plane_gdf.empty:
        pass
    else:
        print("")
        logger.info(f"Area filtering performed on plane objects (small object will be considered as object and not as plane)") 
        for i in range(len(small_plane_gdf)):       
            logger.info(f"Area = {(small_plane_gdf['area'].iloc[i]):.2f} m2 <= area threshold value : {AREA_THRESHOLD} m2") 
        logger.info(f"Add {len(small_plane_gdf)} object(s) from the plane gdf to the object gdf") 
        cluster_vec_gdf = pd.concat([cluster_vec_gdf, small_plane_gdf], ignore_index=True, axis=0)
        cluster_vec_gdf.loc[cluster_vec_gdf["class"] == "plane", "class"] = 'object' 

    # Create occupation layer
    print("")
    logger.info(f"Compute difference (plane - objects) polygon")
    
    # # Control: plot plane polygon, uncomment to see
    # for i in range(len(plane_vec_gdf)): 
        # boundary = gpd.GeoSeries(unary_union(plane_vec_gdf['geometry'].iloc[i]))
        # boundary.plot(color = 'red')
        # plt.show()

    # Free polygon = Plane polygon(s) - Object polygon(s)
    for ii in range(len(cluster_vec_gdf)):
        diff_geom = plane_vec_gdf['geometry'].symmetric_difference(cluster_vec_gdf['geometry'].iloc[ii])
        # # Control: plot object polygon, uncomment to see            
        # boundary = gpd.GeoSeries(unary_union(cluster_vec_gdf['geometry'].iloc[ii]))
        # boundary.plot(color = 'blue')
        # plt.show()

    # Build free area dataframe
    free_df = pd.DataFrame({'area': plane_vec_gdf['area'],'occupation': 'free','geometry': diff_geom})

    # Build occupied area dataframe
    objects_df = cluster_vec_gdf.drop(['class'], axis=1) 
    objects_df['occupation'] = 'object'

    # Build occupation geodataframe
    logger.info(f"Create a binary (free, object) occupation vector file")
    occupation_df = pd.concat([free_df, objects_df], ignore_index=True)
    occupation_gdf = gpd.GeoDataFrame(occupation_df, crs='EPSG:{}'.format(EPSG), geometry='geometry')

    feature_path = os.path.join(output_dir, PCD_NAME + "_occupation.gpkg")
    occupation_gdf.to_file(feature_path)
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