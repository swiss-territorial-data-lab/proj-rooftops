#!/bin/python
# -*- coding: utf-8 -*-

#  proj-rooftops
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
import matplotlib.pyplot as plt
import open3d as o3d

sys.path.insert(1, 'scripts')
import functions.fct_com as fct_com

logger = fct_com.format_logger(logger)
logger.remove()
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}", level="INFO")


if __name__ == "__main__":

    # Start chronometer
    tic = time.time()
    logger.info('Starting...')

    # Argument and parameter specification
    parser = argparse.ArgumentParser(description='The script allows to segment points in 3D point cloud data (STDL.proj-rooftops)')
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
    NB_PLANE = cfg['segmentation']['plane']['number_plane']
    DISTANCE_THRESHOLD = cfg['segmentation']['plane']['distance_threshold']
    RANSAC = cfg['segmentation']['plane']['ransac']
    ITE = cfg['segmentation']['plane']['iteration']
    EPS_PLANE = cfg['segmentation']['plane']['eps']
    MIN_POINTS_PLANE = cfg['segmentation']['plane']['min_points']
    EPS_CLUSTER = cfg['segmentation']['cluster']['eps']
    MIN_POINTS_CLUSTER = cfg['segmentation']['cluster']['min_points']
    VISU = cfg['visualisation']

    os.chdir(WORKING_DIR)

    file_name = PCD_NAME + '_EGID' + str(EGID)
    output_dir = os.path.join(OUTPUT_DIR, file_name)
    # Create an output directory in case it doesn't exist
    fct_com.ensure_dir_exists(output_dir)

    written_files = []

    # Read pcd file and get points array
    logger.info("Read point cloud data file")
    input_dir = os.path.join(INPUT_DIR, file_name, file_name + ".csv")
    pcd_df = pd.read_csv(input_dir)
    pcd_df = pcd_df.drop(['Unnamed: 0'], axis=1) 
    pts = pcd_df.to_numpy()
    logger.info(f"   - {len(pts)} points")

    # Conversion of numpy array to Open3D format + visualisation
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if VISU == 'True':
        o3d.visualization.draw_geometries([pcd])

    # Point cloud plane segmentation  
    segment_models = {}
    segments = {}
    
    remaining_pts = pcd
    planes_df = pd.DataFrame(columns=['X', 'Y', 'Z', 'group', 'type'])

    logger.info(f"Segment and cluster main planes in the pcd")
    for i in range(NB_PLANE):
        # Exploration of the best plane candidate + point clustering
        segment_models[i], inliers = remaining_pts.segment_plane(distance_threshold = DISTANCE_THRESHOLD, ransac_n = RANSAC, num_iterations = ITE)
        segments[i] = remaining_pts.select_by_index(inliers)
        labels = np.array(segments[i].cluster_dbscan(eps = EPS_PLANE, min_points = MIN_POINTS_PLANE, print_progress = True))
        candidates = [len(np.where(labels == j)[0]) for j in np.unique(labels)]
        best_candidate = int(np.unique(labels)[np.where(candidates == np.max(candidates))[0]])
        logger.info(f"   - The best candidate is: {best_candidate}")
        
        # Select the remaining points in the pcd to find a new plane
        remaining_pts = remaining_pts.select_by_index(inliers, invert = True) + segments[i].select_by_index(list(np.where(labels != best_candidate)[0]))
        segments[i] = segments[i].select_by_index(list(np.where(labels == best_candidate)[0]))  
        
        colors = plt.get_cmap("tab20")(i)
        segments[i].paint_uniform_color(list(colors[:3]))
        logger.info(f"   - plane {i} {segments[i]}")
        logger.info(f"   - pass {i} / {NB_PLANE} done.")

        # # Allow to get the coloured planes pcd !!! Not essential but need to be merged in one single file with cluster pcd !!!
        # feature_path = os.path.join(output_dir, file_name + '_plane'+ str(i) + '.ply')
        # o3d.io.write_point_cloud(feature_path, segments[i])
        # written_files.append(feature_path)  
        # logger.info(f"...done. A file was written: {feature_path}")

        # Add segmented planes to a dataframe
        plane = np.asarray(segments[i].points)
        plane_df = pd.DataFrame({'X': plane[:, 0],'Y': plane[:, 1],'Z': plane[:, 2], 'group': i, 'type': 'plane'})
        planes_df = pd.concat([planes_df, plane_df], ignore_index = True)

    # Cluster remaining points (not belonging to a plane) of the pcd after plane segmentation 
    logger.info(f"Remaining points clustering")
    labels = np.array(remaining_pts.cluster_dbscan(eps = EPS_CLUSTER, min_points = MIN_POINTS_CLUSTER))
    max_label = labels.max()
    logger.info(f"   - Point cloud has {max_label + 1} clusters")
    
    colors = plt.get_cmap("tab10")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    remaining_pts.colors = o3d.utility.Vector3dVector(colors[:, :3])
    
    # # Allow to get the coloured cluster pcd !!! Not essential but need to be merged in one single file with planes pcd !!! 
    # feature_path = os.path.join(output_dir, file_name + '_remaining_pts.ply')
    # o3d.io.write_point_cloud(feature_path, remaining_pts)
    # written_files.append(feature_path)  
    # logger.info(f"...done. A file was written: {feature_path}")

    # Add segmented clusters to a dataframe
    clusters = np.asarray(remaining_pts.points)
    clusters_df = pd.DataFrame({'X': clusters[:, 0], 'Y': clusters[:, 1], 'Z': clusters[:, 2], 'group': labels, 'type': 'cluster'})

    # Merge planes and clusters into a single dataframe 
    pcd_seg_df = pd.DataFrame(pd.concat([planes_df, clusters_df], ignore_index = True))

    feature_path = os.path.join(output_dir, file_name + '_segmented.csv')
    pcd_seg_df.to_csv(feature_path)
    written_files.append(feature_path)  
    logger.info(f"...done. A file was written: {feature_path}")

    # Segmented pcd vizualisation
    if VISU == 'True':
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius = 0.5, max_nn = 16), fast_normal_computation = True)
        pcd.paint_uniform_color([0.6, 0.6, 0.6])
        o3d.visualization.draw_geometries([segments[i] for i in range(NB_PLANE)] + [remaining_pts])

    # Save the parameter values used for pcd segmentation
    parameters_df = pd.DataFrame({'number_plane': [NB_PLANE], 
                                'distance_threshold': [DISTANCE_THRESHOLD],
                                'ransac': [RANSAC], 
                                'iteration': [ITE], 
                                'eps_plane': [EPS_PLANE],
                                'min_points_plane': [MIN_POINTS_PLANE],                                
                                'eps_cluster': [EPS_CLUSTER],
                                'min_points_cluster': [MIN_POINTS_CLUSTER]   
                                })
    feature_path = os.path.join(output_dir, file_name + '_parameters.csv')
    parameters_df.to_csv(feature_path)
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