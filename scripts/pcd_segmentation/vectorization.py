#!/bin/python
# -*- coding: utf-8 -*-

#  proj-rooftops


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
from shapely.errors import GEOSException
from shapely.validation import make_valid

sys.path.insert(1, 'scripts')
import functions.fct_pcdseg as pcdseg
from functions.fct_misc import ensure_dir_exists, format_logger

warnings.filterwarnings("ignore", message="CRS not set for some of the concatenation inputs")
warnings.filterwarnings("ignore", message="root:Singular matrix")

logger = format_logger(logger)


def handle_multipolygon(gdf, limit_number=10, limit_area=0.01):
    """Transform multipolygons into polygons

    Args:
        gdf (GeoDataFrame): geodataframe to control
        limit_number (int, optional): Limit number of parts for pruning. Defaults to 10.
        limit_area (float, optional): Limit area in square meter for pruning. Defaults to 0.01.

    Returns:
        GeoDataFrame: the same dataframe but with only polygons and an adjusted id.
    """

    exploded_gdf = gdf.explode(index_parts=False)
    exploded_gdf['area'] = exploded_gdf.area
    if exploded_gdf.shape[0] == gdf.shape[0]:
        return exploded_gdf

    number_parts_df = exploded_gdf['multipoly_id'].value_counts()
    for multipoly_id, number_parts in number_parts_df.items():
        if number_parts > 25:
            exploded_gdf = exploded_gdf[~((exploded_gdf.multipoly_id==multipoly_id) & (exploded_gdf.area<=0.5))].copy()
        elif number_parts > limit_number:
            exploded_gdf = exploded_gdf[~((exploded_gdf.multipoly_id==multipoly_id) & (exploded_gdf.area<=limit_area))].copy()

    exploded_gdf.reset_index(drop=True, inplace=True)

    return exploded_gdf


def main(WORKING_DIR, INPUT_DIR, OUTPUT_DIR, EGIDS, SHP_EGID_ROOFS, epsg = 2056, min_plane_area = 18, max_cluster_area = 42, alpha_shape = None, visu = False):
    """Transform the segmented point cloud into polygons and sort them into free space and cluster

    Args:
        WORKING_DIR (path): working directory
        INPUT_DIR (path): input directory
        OUTPUT_DIR (path): output directory
        EGIDS (list): EGIDs of interest
        epsg (int, optional): reference number of the CRS. Defaults to 2056.
        min_plane_area (float, optional): minimum area for a plane. Defaults to 5.
        max_cluster_area (float, optional): maximum area for an object. Defaults to 25.
        alpha_shape (float, optional): alpha value for the shape algorithm, None means that alpha is optimized. Defaults to None.
        visu (bool, optional): make the vizualisation. Defaults to False.

    Returns:
        tuple: 
            - GeoDataFrame: free and occupied surfaces
            - dict: dictonnary with the path as key and the layers in the path as list.
    """

    logger.info(f"Planes smaller than {min_plane_area} m2 will be considered as object and not as roof sections.") 
    logger.info(f"Objects larger than {max_cluster_area} m2 will be considered as roof sections and not as objects.") 

    os.chdir(WORKING_DIR)

    # Create an output directory in case it doesn't exist
    _ = ensure_dir_exists(OUTPUT_DIR)
    feature_path = os.path.join(OUTPUT_DIR, "all_EGID_occupation.gpkg")

    written_layers = {feature_path: []}

    # Get the EGIDS of interest
    egids=pd.read_csv(EGIDS)

    # Get the rooftops shapes
    rooftops = gpd.read_file(SHP_EGID_ROOFS)

    all_occupation_gdf=gpd.GeoDataFrame(columns=['occupation', 'EGID', 'area', 'geometry'], crs='EPSG:{}'.format(epsg))
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
        plane_multipoly_gdf = pcdseg.vectorize_concave(plane_df, plane, epsg, alpha_shape, visu)
        # plane_multipoly_gdf = pcdseg.vectorize_convex(plane_df, plane) 

        # Load clusters in a dataframe 
        cluster_df = pcd_df[pcd_df['type'] == 'cluster']
        cluster = np.unique(cluster_df['group'])
        cluster = cluster[cluster >= 0]         # Remove outlier class (-1): none classified points

        # Cluster vectorisation
        if cluster_df.empty:
            logger.error('No clusters to vectorize')
        cluster_multipoly_gdf = pcdseg.vectorize_concave(cluster_df, cluster, epsg, alpha_shape, visu)
        # cluster_multipoly_gdf = pcdseg.vectorize_convex(cluster_df, cluster, EPSG)


        # Deal with multipolygon
        plane_multipoly_gdf['multipoly_id'] = plane_multipoly_gdf.index
        plane_vec_gdf = handle_multipolygon(plane_multipoly_gdf) if not plane_multipoly_gdf.empty else plane_multipoly_gdf

        cluster_multipoly_gdf['multipoly_id'] = cluster_multipoly_gdf.index
        cluster_vec_gdf = handle_multipolygon(cluster_multipoly_gdf) if not cluster_multipoly_gdf.empty else cluster_multipoly_gdf


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
        if not cluster_vec_gdf.empty:
            # Drop cluster smaller than 1.5 pixels
            cluster_vec_gdf = cluster_vec_gdf[cluster_vec_gdf.area > 0.01]
            cluster_vec_gdf.set_geometry('geometry', inplace=True)

            # Free polygon = Plane polygon(s) - Object polygon(s)
            diff_geom = []
            i = 0
            if not plane_vec_gdf.empty:
                plane_vec_gdf.set_geometry('geometry', inplace=True)
                for geom in plane_vec_gdf.geometry.to_numpy():
                    diff_geom.append(geom.difference(cluster_vec_gdf.geometry.unary_union))

            # Build free area dataframe
            free_gdf = gpd.GeoDataFrame({'occupation': 0, 'geometry': diff_geom}, index=range(len(plane_vec_gdf)))

            # Build occupied area dataframe
            objects_gdf = cluster_vec_gdf.drop(['class'], axis=1) 
            objects_gdf['occupation'] = 1
        
        else:
            free_gdf = plane_vec_gdf.copy()
            free_gdf['occupation'] = 0

            objects_gdf = cluster_vec_gdf.copy()

        if not plane_vec_gdf.empty:
            free_gdf['area']=free_gdf.area

        # Build occupation geodataframe
        occupation_df = pd.concat([free_gdf, objects_gdf], ignore_index=True)
        if not occupation_df.empty:
            occupation_gdf = gpd.GeoDataFrame(occupation_df, crs='EPSG:{}'.format(epsg), geometry='geometry')

            try:
                clipped_occupation_gdf = occupation_gdf.clip(rooftops.loc[rooftops.EGID==egid, 'geometry'].buffer(-0.01), keep_geom_type=True)
            except GEOSException:
                occupation_gdf.loc[:, 'geometry'] = [make_valid(geom) for geom in occupation_gdf.geometry]
                clipped_occupation_gdf = occupation_gdf.clip(rooftops.loc[rooftops.EGID==egid, 'geometry'].buffer(-0.01), keep_geom_type=True)

            clipped_occupation_gdf.loc[:,'area'] = clipped_occupation_gdf.area
            clipped_occupation_gdf['EGID'] = egid
            all_occupation_gdf = pd.concat([all_occupation_gdf, clipped_occupation_gdf[all_occupation_gdf.columns]], ignore_index=True)


    all_occupation_gdf['det_id'] = all_occupation_gdf.index
    all_occupation_gdf.to_file(feature_path, layer='occupation_for_all_EGIDs', index=False)
    written_layers[feature_path].append('occupation_for_all_EGIDs')


    return all_occupation_gdf, written_layers


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
    SHP_EGID_ROOFS = cfg['roofs']
    EPSG = cfg['epsg']
    AREA_MIN_PLANE = cfg['area_threshold']['min']
    AREA_MAX_OBJECT = cfg['area_threshold']['max']
    ALPHA = cfg['alpha_shape']
    VISU = cfg['visualisation']

    all_occupation_gdf, written_files = main(WORKING_DIR, INPUT_DIR, OUTPUT_DIR, EGIDS, SHP_EGID_ROOFS, EPSG, AREA_MIN_PLANE, AREA_MAX_OBJECT, ALPHA, VISU)

    print()
    dict_only_key=list(written_files.keys())[0]
    logger.success(f"The following layers were written in the file '{dict_only_key}'. Let's check them out!")
    for layer in written_files[dict_only_key]:
        logger.info(f'    - {layer}')
    print()

    # Stop chronometer  
    toc = time()
    logger.success(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()