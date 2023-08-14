#!/bin/python
# -*- coding: utf-8 -*-

#  Proj rooftops
#
#      Clemence Herny 
#      Gwenaelle Salamin
#      Alessandro Cerioni 


import os, sys
import argparse
from loguru import logger
from time import time
from tqdm import tqdm
from yaml import load, FullLoader

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.ops import unary_union

sys.path.insert(1, 'scripts')
import functions.fct_pcdseg as fct_seg
from functions.fct_misc import ensure_dir_exists, format_logger

logger = format_logger(logger)

# Define functions ----------------------

def main(WORKING_DIR, INPUT_DIR, OUTPUT_DIR, EGIDS, EPSG = 2056, AREA_MIN_PLANE = 5, AREA_MAX_OBJECT = 25, ALPHA = None, VISU = False):
    """Transform the segmented point cloud into polygons and sort them into free space and cluster

    Args:
        WORKING_DIR (path): working directory
        INPUT_DIR (path): input directory
        OUTPUT_DIR (path): output directory
        EGIDS (list): EGIDs of interest
        EPSG (int, optional): reference number of the CRS. Defaults to 2056.
        AREA_MIN_PLANE (float, optional): minimum area for a plane. Defaults to 5.
        AREA_MAX_OBJECT (float, optional): maximum area for an object. Defaults to 25.
        ALPHA (float, optional): alpha value for the shape algorithm, None means that alpha is optimized. Defaults to None.
        VISU (bool, optional): make the vizualisation. Defaults to False.

    Returns:
        _type_: _description_
    """

    logger.info(f"Planes smaller than {AREA_MIN_PLANE} m2 will be considered as object and not as roof sections.") 
    logger.info(f"Objects larger than {AREA_MAX_OBJECT} m2 will be considered as roof sections and not as objects.") 

    os.chdir(WORKING_DIR)

    # Create an output directory in case it doesn't exist
    _ = ensure_dir_exists(OUTPUT_DIR)
    feature_path = os.path.join(OUTPUT_DIR, "all_EGID_occupation.gpkg")

    written_layers = []

    # Get the EGIDS of interest
    with open(EGIDS, 'r') as src:
        egids=src.read()
    egid_list=egids.split("\n")

    all_occupation_gdf=gpd.GeoDataFrame(columns=['occupation', 'EGID', 'area', 'geometry'], crs='EPSG:{}'.format(EPSG))
    for egid in tqdm(egid_list):
        file_name = 'EGID_' + str(egid)

        input_dir = os.path.join(INPUT_DIR, file_name, file_name + "_segmented.csv")
        pcd_df = pd.read_csv(input_dir)

        # Create a plane dataframe
        plane_df = pcd_df[pcd_df['type'] == 'plane']
        plane = np.unique(plane_df['group'])

        # Plane vectorization
        plane_vec_gdf = fct_seg.vectorize_concave(plane_df, plane, EPSG, ALPHA, VISU)
        # plane_vec_gdf = fct_seg.vectorize_convex(plane_df, plane) 

        # Load clusters in a dataframe 
        cluster_df = pcd_df[pcd_df['type'] == 'cluster']
        cluster = np.unique(cluster_df['group'])
        cluster = cluster[cluster >= 0]                                         # Remove outlier class (-1): none classified points

        # Cluster vectorisation
        cluster_vec_gdf = fct_seg.vectorize_concave(cluster_df, cluster, EPSG, ALPHA, VISU)
        # cluster_vec_gdf = fct_seg.vectorize_convex(cluster_df, cluster, EPSG)

        # Filtering: identify and isolate plane that are too small
        small_plane_gdf = plane_vec_gdf[plane_vec_gdf['area'] <= AREA_MIN_PLANE]
        plane_vec_gdf.drop(small_plane_gdf.index, inplace = True)

        # If it exists, add cluster previously classified as plane to the object class 
        if not small_plane_gdf.empty:
            print("")
            logger.info(f"Add {len(small_plane_gdf)} object{'s' if len(small_plane_gdf)>1 else ''} from the planes to the objects.") 
            cluster_vec_gdf = pd.concat([cluster_vec_gdf, small_plane_gdf], ignore_index=True, axis=0)
            cluster_vec_gdf.loc[cluster_vec_gdf["class"] == "plane", "class"] = 'object' 
        del small_plane_gdf

        # Filtering: identify and isolate plane that are too big
        large_objects_gdf = cluster_vec_gdf[cluster_vec_gdf['area'] > AREA_MAX_OBJECT]
        cluster_vec_gdf.drop(large_objects_gdf.index, inplace = True)        

        # If it exists, add cluster previously classified as plane to the object class 
        if not large_objects_gdf.empty:
            print("")
            logger.info(f"Add {len(large_objects_gdf)} plane{'s' if len(large_objects_gdf)>1 else ''} from the objects to the roof sections.") 
            plane_vec_gdf = pd.concat([plane_vec_gdf, large_objects_gdf], ignore_index=True, axis=0)
            plane_vec_gdf.loc[plane_vec_gdf["class"] == "plane", "class"] = 'object' 
        del large_objects_gdf

        # Create occupation layer
        if False:
            # Control: plot plane polygon, uncomment to see
            boundary = gpd.GeoSeries(plane_vec_gdf.unary_union)
            boundary.plot(color = 'red')
            plt.savefig('processed/test_outputs/segmented_planes.jpg', bbox_inches='tight')

        # Remove clusters within another cluster
        if True:
            reviewed_cluster=cluster_vec_gdf.copy()
            dropped_index=[]
            for cluster in cluster_vec_gdf.itertuples():

                if cluster.Index+1 == cluster_vec_gdf.shape[0]:
                    if cluster_vec_gdf.shape != reviewed_cluster.shape:
                        logger.info(f'{cluster_vec_gdf.shape[0]-reviewed_cluster.shape[0]} objects were dropped because they were within another objects')
                        cluster_vec_gdf=reviewed_cluster.copy()
                    break

                elif cluster.index in dropped_index:
                    continue

                for second_cluster in cluster_vec_gdf.loc[cluster_vec_gdf.index > cluster.Index].itertuples():
                    if cluster.geometry.intersects(second_cluster.geometry):
                            if second_cluster.geometry.within(cluster.geometry):
                                reviewed_cluster.drop(index=(second_cluster.Index), inplace=True)
                                dropped_index.append(second_cluster.index)

                            elif cluster.geometry.within(second_cluster.geometry):
                                reviewed_cluster.drop(index=(cluster.Index), inplace=True)
                                dropped_index.append(second_cluster.index)

        # Drop cluster smaller than 1.5 pixels
        cluster_vec_gdf=cluster_vec_gdf[cluster_vec_gdf.area > 0.015]

        # Free polygon = Plane polygon(s) - Object polygon(s)
        diff_geom=[]
        i=0
        for geom in plane_vec_gdf.geometry.to_numpy():
            diff_geom.append(geom.difference(cluster_vec_gdf.geometry.unary_union))

            if False:
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
        occupation_df = pd.concat([free_df, objects_df], ignore_index=True)
        occupation_gdf = gpd.GeoDataFrame(occupation_df, crs='EPSG:{}'.format(EPSG), geometry='geometry')

        occupation_gdf.to_file(feature_path, layer=file_name, index=False)
        written_layers.append(file_name)  

        occupation_gdf['EGID']=egid
        all_occupation_gdf=pd.concat([all_occupation_gdf, occupation_gdf[all_occupation_gdf.columns]], ignore_index=True)

    all_occupation_gdf['pred_id']=all_occupation_gdf.index
    all_occupation_gdf.to_file(feature_path, layer='occupation_for_all_EGIDs', index=False)
    written_layers.append('occupation_for_all_EGIDs')

    print()
    logger.info(f"The following layers were written in the file '{feature_path}'. Let's check them out!")
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
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()