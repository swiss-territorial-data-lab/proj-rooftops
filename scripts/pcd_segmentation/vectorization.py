#!/bin/python
# -*- coding: utf-8 -*-

#  Proj rooftops
#
#      Clemence Herny 
#      Gwenaelle Salamin
#      Alessandro Cerioni 


import argparse
import os
import sys
import warnings
from loguru import logger
from time import time
from tqdm import tqdm
from yaml import load, FullLoader

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.ops import unary_union

sys.path.insert(1, 'scripts')
import functions.fct_pcdseg as fct_seg
from functions.fct_metrics import intersection_over_union
from functions.fct_misc import ensure_dir_exists, format_logger

warnings.filterwarnings("ignore", message="CRS not set for some of the concatenation inputs")
warnings.filterwarnings("ignore", message="root:Singular matrix")

logger = format_logger(logger)

# Define functions ----------------------

def handle_overlapping_cluster(clusters_gdf):
    """Delete clusters inside another cluster.

    Args:
        clusters_gdf (GeoDataFrame): clusters of objects as determined by the vecotrization procedure

    Returns:
        GeoDataFrame, int: Cleaned clusters and the number of fusion between two clusters that occured.
    """
        
    reviewed_cluster = clusters_gdf.copy()
    dropped_index = []

    for cluster in clusters_gdf.itertuples():

        # Stop if you are at the last line of the table
        if cluster.Index+1 == clusters_gdf.shape[0]:
            nbr_dropped_objects = len(dropped_index)
            if nbr_dropped_objects > 0:
                logger.info(f'{nbr_dropped_objects} objects were dropped, because they were within another object.')
            break

        for second_cluster in clusters_gdf.loc[clusters_gdf.index > cluster.Index].itertuples():
            first_geometry = cluster.geometry
            second_geometry = second_cluster.geometry
            
            if cluster.Index in dropped_index:
                nbr_dropped_objects = len(dropped_index)
                if cluster.Index+1 == clusters_gdf.shape[0] & nbr_dropped_objects > 0:
                     logger.info(f'{nbr_dropped_objects} objects were dropped, because they were within another object.')
                break

            if first_geometry.intersects(second_geometry) & (second_cluster.Index not in dropped_index):
                    
                    if second_geometry.within(first_geometry):
                        reviewed_cluster.drop(index=(second_cluster.Index), inplace=True)
                        dropped_index.append(second_cluster.Index)

                    elif first_geometry.within(second_geometry):
                        reviewed_cluster.drop(index=(cluster.Index), inplace=True)
                        dropped_index.append(cluster.Index)

    reviewed_cluster.reset_index(drop=True, inplace=True)

    return reviewed_cluster

def main(WORKING_DIR, INPUT_DIR, OUTPUT_DIR, EGIDS, EPSG = 2056, min_plane_area = 18, max_cluster_area = 42, alpha_shape = None, visu = False):
    """Transform the segmented point cloud into polygons and sort them into free space and cluster

    Args:
        WORKING_DIR (path): working directory
        INPUT_DIR (path): input directory
        OUTPUT_DIR (path): output directory
        EGIDS (list): EGIDs of interest
        EPSG (int, optional): reference number of the CRS. Defaults to 2056.
        min_plane_area (float, optional): minimum area for a plane. Defaults to 5.
        max_cluster_area (float, optional): maximum area for an object. Defaults to 25.
        alpha_shape (float, optional): alpha value for the shape algorithm, None means that alpha is optimized. Defaults to None.
        visu (bool, optional): make the vizualisation. Defaults to False.

    Returns:
        _type_: _description_
    """

    logger.info(f"Planes smaller than {min_plane_area} m2 will be considered as object and not as roof sections.") 
    logger.info(f"Objects larger than {max_cluster_area} m2 will be considered as roof sections and not as objects.") 

    os.chdir(WORKING_DIR)

    # Create an output directory in case it doesn't exist
    _ = ensure_dir_exists(OUTPUT_DIR)
    feature_path = os.path.join(OUTPUT_DIR, "all_EGID_occupation.gpkg")

    written_layers = []

    # Get the EGIDS of interest
    egids=pd.read_csv(EGIDS)

    all_occupation_gdf=gpd.GeoDataFrame(columns=['occupation', 'EGID', 'area', 'geometry'], crs='EPSG:{}'.format(EPSG))
    for egid in tqdm(egids.EGID.to_numpy()):
        file_name = 'EGID_' + str(egid)

        input_dir = os.path.join(INPUT_DIR, file_name + "_segmented.csv")
        try:
            pcd_df = pd.read_csv(input_dir)
        except FileNotFoundError:
            logger.error(f"No segmentation file for the EGID {egid}.")

        # Create a plane dataframe
        plane_df = pcd_df[pcd_df['type'] == 'plane']
        plane = np.unique(plane_df['group'])

        # Plane vectorization
        if plane_df.empty:
            logger.error('No planes to vectorize')
        plane_vec_gdf = fct_seg.vectorize_concave(plane_df, plane, EPSG, alpha_shape, visu)
        # plane_vec_gdf = fct_seg.vectorize_convex(plane_df, plane) 

        # Load clusters in a dataframe 
        cluster_df = pcd_df[pcd_df['type'] == 'cluster']
        cluster = np.unique(cluster_df['group'])
        cluster = cluster[cluster >= 0]         # Remove outlier class (-1): none classified points

        # Cluster vectorisation
        if cluster_df.empty:
            logger.error('No clusters to vectorize')
        cluster_vec_gdf = fct_seg.vectorize_concave(cluster_df, cluster, EPSG, alpha_shape, visu)
        # cluster_vec_gdf = fct_seg.vectorize_convex(cluster_df, cluster, EPSG)

        # Filtering: identify and isolate plane that are too small
        if not plane_vec_gdf.empty:
            small_plane_gdf = plane_vec_gdf[plane_vec_gdf['area'] <= min_plane_area]
            plane_vec_gdf.drop(small_plane_gdf.index, inplace = True)

            # If it exists, add cluster previously classified as plane to the object class 
            if not small_plane_gdf.empty:
                print("")
                logger.info(f"Add {len(small_plane_gdf)} plane{'s' if len(small_plane_gdf)>1 else ''} to the objects.") 
                cluster_vec_gdf = pd.concat([cluster_vec_gdf, small_plane_gdf], ignore_index=True, axis=0)
                cluster_vec_gdf.loc[cluster_vec_gdf["class"] == "plane", "class"] = 'object' 
            del small_plane_gdf

        # Filtering: identify and isolate objects that are too big
        if not cluster_vec_gdf.empty:
            large_objects_gdf = cluster_vec_gdf[cluster_vec_gdf['area'] > max_cluster_area]
            cluster_vec_gdf.drop(large_objects_gdf.index, inplace = True)        

            # If it exists, add cluster previously classified as object to the plane class 
            if not large_objects_gdf.empty:
                print("")
                logger.info(f"Add {len(large_objects_gdf)} object{'s' if len(large_objects_gdf)>1 else ''} to the roof sections.") 
                plane_vec_gdf = pd.concat([plane_vec_gdf, large_objects_gdf], ignore_index=True, axis=0)
                plane_vec_gdf.loc[plane_vec_gdf["class"] == "plane", "class"] = 'object' 
            del large_objects_gdf

        # Create occupation layer
        if False:
            # Control: plot plane polygon
            boundary = gpd.GeoSeries(plane_vec_gdf.unary_union)
            boundary.plot(color = 'red')
            plt.savefig('processed/test_outputs/segmented_planes.jpg', bbox_inches='tight')

        # cluster_vec_gdf = handle_overlapping_cluster(cluster_vec_gdf)

        if not cluster_vec_gdf.empty:
            # Drop cluster smaller than 1.5 pixels
            cluster_vec_gdf=cluster_vec_gdf[cluster_vec_gdf.area > 0.015]

            # Free polygon = Plane polygon(s) - Object polygon(s)
            diff_geom=[]
            i=0
            if not plane_vec_gdf.empty:
                for geom in plane_vec_gdf.geometry.to_numpy():
                    diff_geom.append(geom.difference(cluster_vec_gdf.geometry.unary_union))

                if False:
                    # Control: plot object polygon       
                    boundary = gpd.GeoSeries(diff_geom)
                    boundary.plot(color = 'blue')
                    plt.savefig(f'processed/test_outputs/segmented_free_space_{i}.jpg', bbox_inches='tight')
                    i+=1

            # Build free area dataframe
            free_gdf = gpd.GeoDataFrame({'occupation': 0, 'geometry': diff_geom}, index=range(len(plane_vec_gdf)))

            # Build occupied area dataframe
            objects_gdf = cluster_vec_gdf.drop(['class'], axis=1) 
            objects_gdf['occupation'] = 1
        
        else:
            free_gdf=plane_vec_gdf.copy()
            free_gdf['occupation'] = 0

            objects_gdf=cluster_vec_gdf.copy()

        if not plane_vec_gdf.empty:
            free_gdf['area']=free_gdf.area

        # Build occupation geodataframe
        occupation_df = pd.concat([free_gdf, objects_gdf], ignore_index=True)
        if not occupation_df.empty:
            occupation_gdf = gpd.GeoDataFrame(occupation_df, crs='EPSG:{}'.format(EPSG), geometry='geometry')
            # occupation_gdf.to_file(feature_path, layer=file_name, index=False)
            # written_layers.append(file_name)  

            occupation_gdf['EGID']=egid
            all_occupation_gdf=pd.concat([all_occupation_gdf, occupation_gdf[all_occupation_gdf.columns]], ignore_index=True)


    all_occupation_gdf['pred_id']=all_occupation_gdf.index
    all_occupation_gdf.to_file(feature_path, layer='occupation_for_all_EGIDs', index=False)
    written_layers.append('occupation_for_all_EGIDs')

    print()
    logger.success(f"The following layers were written in the file '{feature_path}'. Let's check them out!")
    for layer in written_layers:
        logger.info(layer)
    print()

    return all_occupation_gdf


# ------------------------------------------

if __name__ == "__main__":
    # Start chronometer
    tic = time()
    logger.info('Starting...')

    # Argument and parameter specification
    parser = argparse.ArgumentParser(description='The script allows to transform 3D segmented point cloud to 2D polygon (STDL.proj-rooftops)')
    parser.add_argument('config_file', type=str, help='Framework configuration file')
    args = parser.parse_args()

    logger.info(f"Using {args.config_file} as config file.")

    with open(args.config_file) as fp:
        cfg = load(fp, Loader=FullLoader)[os.path.basename(__file__)]

    # Load input parameters
    WORKING_DIR = cfg['working_dir']
    INPUT_DIR = cfg['input_dir']
    OUTPUT_DIR = cfg['output_dir']

    EGIDS = cfg['egids']
    EPSG = cfg['epsg']
    AREA_MIN_PLANE = cfg['area_threshold']['min']
    AREA_MAX_OBJECT = cfg['area_threshold']['max']
    ALPHA = cfg['alpha_shape']
    VISU = cfg['visualisation']

    all_occupation_gdf=main(WORKING_DIR, INPUT_DIR, OUTPUT_DIR, EGIDS, EPSG, AREA_MIN_PLANE, AREA_MAX_OBJECT, ALPHA, VISU)

    # Stop chronometer  
    toc = time()
    logger.success(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()