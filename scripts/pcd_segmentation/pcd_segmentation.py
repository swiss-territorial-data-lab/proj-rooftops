#!/bin/python
# -*- coding: utf-8 -*-

#  proj-rooftops
#
#      Clemence Herny 
#      Gwenaelle Salamin
#      Alessandro Cerioni 

import argparse
import os
import sys
from loguru import logger
from time import time
from tqdm import tqdm
from yaml import load, FullLoader

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas as pd

sys.path.insert(1, 'scripts')
import functions.fct_misc as fct_misc

logger = fct_misc.format_logger(logger)


# Define functions ----------------------

def main (WORKING_DIR, INPUT_DIR, OUTPUT_DIR, 
          EGIDS, 
          distance_threshold, ransac, iterations, eps_planes, min_points_planes, eps_clusters, min_points_clusters,
          number_planes=None, visu=False):
    """Perform the segmentation of the point cloud in planes and clusters in order to find the roof planes.

    Args:
        WORKING_DIR (path): working directory
        INPUT_DIR (path): input directory
        OUTPUT_DIR (path): output direcotry
        EGIDS (list): EGIDs of interest
        distance_threshold (float): distance to consider for the noise in the ransac algorithm
        ransac (int): number of points to consider for the ransac algorithm
        iterations (int): number of iteration for the ransac algorithm
        eps_planes (float): distance to neighbours in a plane
        min_points_planes (int): minimum number of points in a plane
        eps_clusters (float): distance to neighbours in a cluster
        min_points_clusters (int): minimum number of points in a cluster
        number_planes (int): approximate number of planes to find. If 'None' take the number of planes from the vector file. Default to None
        visu (boolean): visualize the results. Default to false

    Returns:
        list: list of the written files
    """
    
    os.chdir(WORKING_DIR)

    # Create an output directory in case it doesn't exist
    _ = fct_misc.ensure_dir_exists(OUTPUT_DIR)

    written_files = []

    # Get the EGIDS of interest
    egids=pd.read_csv(EGIDS)

    if not number_planes:
        logger.info('The number of planes for each EGID is deduced from the original layer.')
    number_planes_ini=number_planes

    for egid_info in tqdm(egids.itertuples()):

        file_name = 'EGID_' + str(egid_info.EGID)
        # Read pcd file and get points array
        csv_input_path = os.path.join(INPUT_DIR, file_name + ".csv")
        pcd_df = pd.read_csv(csv_input_path)
        pcd_df = pcd_df.drop(['Unnamed: 0'], axis=1) 
        array_pts = pcd_df.to_numpy()

        # Conversion of numpy array to Open3D format + visualisation
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(array_pts)
        # if visu:
        #     o3d.visualization.draw_geometries([pcd])

        # Point cloud plane segmentation  

        # Parameters for the plane equation
        segment_models = {}
        # Points in the plane
        segments = {}

        remaining_pts = pcd
        planes_df = pd.DataFrame(columns=['X', 'Y', 'Z', 'group', 'type'])

        if not number_planes_ini:
            number_planes = int(egid_info.nbr_elemen)
            if number_planes > 100:
                number_planes = 100
            
            logger.info(f'Working on EGID {egid_info.EGID} with {number_planes} planes...')

        for i in range(number_planes):
            # Exploration of the best plane candidate + point clustering
            segment_models[i], inliers = remaining_pts.segment_plane(
                distance_threshold = distance_threshold, ransac_n = ransac, num_iterations = iterations
            )
            segments[i] = remaining_pts.select_by_index(inliers)
            labels = np.array(segments[i].cluster_dbscan(eps = eps_planes, min_points = min_points_planes, print_progress = True))
            candidates = pd.DataFrame({
                'value': [j for j in np.unique(labels) if j!=-1],
                'value_count':[len(np.where(labels == j)[0]) for j in np.unique(labels) if j!=-1]
            })
            if len(candidates)==0:
                break
            elif len(candidates)==1:
                best_candidate=candidates.iloc[0,0]
            else:
                best_candidate = int(candidates.loc[candidates.value_count==candidates.value_count.max(), 'value'].iloc[0])
            logger.info(f"   - The best candidate is: {best_candidate}")
            
            # Select the remaining points in the pcd to find a new plane
            remaining_pts = remaining_pts.select_by_index(inliers, invert = True) + \
                segments[i].select_by_index(list(np.where(labels != best_candidate)[0]))
            segments[i] = segments[i].select_by_index(list(np.where(labels == best_candidate)[0]))  
            
            colors = plt.get_cmap("tab20")(i)
            segments[i].paint_uniform_color(list(colors[:3]))
            logger.info(f"   - plane {i} {segments[i]}")
            logger.info(f"   - pass {i} / {number_planes} done.")

            # Add segmented planes to a dataframe
            plane = np.asarray(segments[i].points)
            plane_df = pd.DataFrame({'X': plane[:, 0],'Y': plane[:, 1],'Z': plane[:, 2], 'group': i, 'type': 'plane'})
            planes_df = pd.concat([planes_df, plane_df], ignore_index = True)

            if len(remaining_pts.points)<=ransac:
                break

        number_planes = i

        # Cluster remaining points (not belonging to a plane) of the pcd after plane segmentation
        labels = np.array(remaining_pts.cluster_dbscan(eps = eps_clusters, min_points = min_points_clusters))
        if labels.size == 0:
            clusters_df = pd.DataFrame({'X': [], 'Y': [], 'Z': [], 'group': [], 'type': []})
        else: 
            max_label = labels.max()

            colors = plt.get_cmap("tab10")(labels / (max_label if max_label > 0 else 1))
            colors[labels < 0] = 0
            remaining_pts.colors = o3d.utility.Vector3dVector(colors[:, :3])

            # Add segmented clusters to a dataframe
            clusters = np.asarray(remaining_pts.points)
            clusters_df = pd.DataFrame({'X': clusters[:, 0], 'Y': clusters[:, 1], 'Z': clusters[:, 2], 'group': labels, 'type': 'cluster'})

        # Merge planes and clusters into a single dataframe 
        pcd_seg_df = pd.DataFrame(pd.concat([planes_df, clusters_df], ignore_index = True))

        feature_path = os.path.join(OUTPUT_DIR, file_name + '_segmented.csv')
        pcd_seg_df.to_csv(feature_path)
        written_files.append(feature_path)  
        logger.info(f"...done. A file was written: {feature_path}")

        # Segmented pcd vizualisation
        if visu:
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius = 0.5, max_nn = 16), fast_normal_computation = True)
            pcd.paint_uniform_color([0.6, 0.6, 0.6])
            o3d.visualization.draw_geometries([segments[i] for i in range(number_planes)] + [remaining_pts])
            print()

    return written_files


# ------------------------------------------

if __name__ == "__main__":
    # Start chronometer
    tic = time()
    logger.info('Starting...')

    # Argument and parameter specification
    parser = argparse.ArgumentParser(description='The script allows to segment points in 3D point cloud data (STDL.proj-rooftops)')
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

    SEGMENTATION=cfg['segmentation']
    # NB_PLANES = SEGMENTATION['planes']['number_planes']
    DISTANCE_THRESHOLD = SEGMENTATION['planes']['distance_threshold']
    RANSAC = SEGMENTATION['planes']['ransac']
    ITER = SEGMENTATION['planes']['iterations']
    EPS_PLANE = SEGMENTATION['planes']['eps']
    MIN_POINTS_PLANE = SEGMENTATION['planes']['min_points']
    EPS_CLUSTER = SEGMENTATION['clusters']['eps']
    MIN_POINTS_CLUSTER = SEGMENTATION['clusters']['min_points']

    VISU = cfg['visualisation']

    written_files = main(WORKING_DIR, INPUT_DIR, OUTPUT_DIR, EGIDS, 
        DISTANCE_THRESHOLD, RANSAC, ITER, EPS_PLANE, MIN_POINTS_PLANE, EPS_CLUSTER, MIN_POINTS_CLUSTER,
        #  NB_PLANES, 
        visu=VISU
         )
    
    print()
    logger.success("The following files were written. Let's check them out!")
    for written_file in written_files:
        logger.info(written_file)
    print()

    
    # Stop chronometer  
    toc = time()
    logger.success(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()