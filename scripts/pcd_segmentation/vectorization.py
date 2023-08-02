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
import functions.fct_com as fct_com
import functions.fct_pcdseg as fct_seg

logger = fct_com.format_logger(logger)


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
    # Create an output directory in case it doesn't exist
    output_dir = fct_com.ensure_dir_exists(os.path.join(OUTPUT_DIR, file_name))

    written_files = []

    logger.info("Read point cloud file")
    input_dir = os.path.join(INPUT_DIR, file_name, file_name + "_segmented.csv")
    pcd_df = pd.read_csv(input_dir)

    # Create a plane dataframe
    plane_df = pcd_df[pcd_df['type'] == 'plane']
    plane = np.unique(plane_df['group'])
    print("")
    logger.info(f"Number of plane(s): {np.max(plane) + 1}")

    # Plane vectorization
    logger.info(f"Vectorize the planes")
    plane_vec_gdf = fct_seg.vectorize_concave(plane_df, plane, EPSG, ALPHA)
    # plane_vec_gdf = fct_seg.vectorize_convex(plane_df, plane) 

    # Load clusters in a dataframe 
    cluster_df = pcd_df[pcd_df['type'] == 'cluster']
    cluster = np.unique(cluster_df['group'])
    cluster = cluster[cluster >= 0]                                         # Remove outlier class (-1): none classified points
    print("")
    logger.info(f"Number of cluster(s): {np.max(cluster) + 1}")

    # Cluster vectorisation
    logger.info(f"Vectorize object")
    cluster_vec_gdf = fct_seg.vectorize_concave(cluster_df, cluster, EPSG, ALPHA)
    # cluster_vec_gdf = fct_seg.vectorize_convex(cluster_df, cluster, EPSG)

    # Filtering: identify and remove plane element with an area below threshold value to be added to the object gdf
    # An object can be classified as a plane in 'plane_segmentation.py' script. 
    # We can add an threshold value on the area under which the plane can be considered to belong to a rooftop object
    logger.info(f"Area filtering for plane objects. Small planes will be considered as object and not as roof parts.") 
    small_plane_gdf = plane_vec_gdf[plane_vec_gdf['area'] <= AREA_THRESHOLD]
    plane_vec_gdf.drop(small_plane_gdf.index, inplace = True)

    # If it exists, add cluster previously classified as plane to the object class 
    if not small_plane_gdf.empty:
        print("")
        logger.info(f"Add {len(small_plane_gdf)} object{'s' if len(small_plane_gdf)>1 else ''} from the planes to the objects.") 
        cluster_vec_gdf = pd.concat([cluster_vec_gdf, small_plane_gdf], ignore_index=True, axis=0)
        cluster_vec_gdf.loc[cluster_vec_gdf["class"] == "plane", "class"] = 'object' 

    # Create occupation layer
    print("")
    logger.info(f"Compute difference between planes and objects")
    
    # Control: plot plane polygon, uncomment to see
    boundary = gpd.GeoSeries(plane_vec_gdf.unary_union)
    boundary.plot(color = 'red')
    plt.savefig('processed/test_outputs/segmented_planes.jpg', bbox_inches='tight')

    # Free polygon = Plane polygon(s) - Object polygon(s)
    diff_geom=[]
    i=0
    for geom in plane_vec_gdf.geometry.to_numpy():
        diff_geom.append(geom.difference(cluster_vec_gdf.geometry.unary_union))
        # Control: plot object polygon, uncomment to see          
        boundary = gpd.GeoSeries(diff_geom)
        boundary.plot(color = 'blue')
        plt.savefig(f'processed/test_outputs/segmented_free_space_{i}.jpg', bbox_inches='tight')
        i+=1

    # Build free area dataframe
    free_df = gpd.GeoDataFrame({'occupation': 0, 'geometry': diff_geom}, index=range(len(plane_vec_gdf)))
    free_df['area']=free_df.area

    # Build occupied area dataframe
    objects_df = cluster_vec_gdf.drop(['class'], axis=1) 
    objects_df['occupation'] = 1

    # Build occupation geodataframe
    logger.info(f"Create a binary (free, object) occupation vector file")
    occupation_df = pd.concat([free_df, objects_df], ignore_index=True)
    occupation_gdf = gpd.GeoDataFrame(occupation_df, crs='EPSG:{}'.format(EPSG), geometry='geometry')
    occupation_gdf['id']=occupation_gdf.index

    feature_path = os.path.join(output_dir, file_name + "_occupation.gpkg")
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