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
import matplotlib.pyplot as plt
import open3d as o3d

from loguru import logger

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
    VISU = cfg['visu']

    # Create an output directory in case it doesn't exist
    output_dir = os.path.join(OUTPUT_DIR + DATA_NAME + '_test/')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    written_files = []

    # Open point cloud file
    input_dir = os.path.join(INPUT_DIR + DATA_NAME + '_test/')
    df = pd.read_csv(input_dir + DATA_NAME + '_filter.csv')
    df = df.drop(['Unnamed: 0'], axis=1) 
    point_data = df.to_numpy()
    logger.info(f"Read point cloud file: {len(point_data)} points")

    # Conversion of numpy array to Open3D format and visualisation
    geom = o3d.geometry.PointCloud()
    geom.points = o3d.utility.Vector3dVector(point_data)
    # o3d.visualization.draw_geometries([geom])

    # Point cloud plane segmentation  
    segment_models={}
    segments={}
    max_plane_idx=2
    d_threshold=1.0
    rest=geom
    df1 = pd.DataFrame()

    logger.info(f"Segment and cluster main planes in point cloud")
    for i in range(max_plane_idx):
        print(i)
        colors = plt.get_cmap("tab20")(i)
        segment_models[i], inliers = rest.segment_plane(distance_threshold=0.1,ransac_n=3,num_iterations=1000)
        segments[i]=rest.select_by_index(inliers)
        labels = np.array(segments[i].cluster_dbscan(eps=d_threshold, min_points=50, print_progress=True))
        candidates=[len(np.where(labels==j)[0]) for j in np.unique(labels)]
        best_candidate=int(np.unique(labels)[np.where(candidates== np.max(candidates))[0]])

        logger.info(f"the best candidate is: {best_candidate}")
        rest = rest.select_by_index(inliers, invert=True) + segments[i].select_by_index(list(np.where(labels!=best_candidate)[0]))
        segments[i]=segments[i].select_by_index(list(np.where(labels== best_candidate)[0]))
        segments[i].paint_uniform_color(list(colors[:3]))
        logger.info(f"plane {i} {segments[i]}")
        logger.info(f"pass {i} / {max_plane_idx} done.")

        feature_path = DATA_NAME + '_plane'+ str(i) + '.ply'
        o3d.io.write_point_cloud(output_dir + feature_path, segments[i])
        written_files.append(feature_path)  
        logger.info(f"...done. A file was written: {feature_path}")

        segs = np.asarray(segments[i].points)
        df = pd.DataFrame({'X':segs[:,0],'Y':segs[:,1],'Z':segs[:,2]})
        df['group'] = i
        df['type'] = 'plane' 
        df1=pd.concat([df1, df], ignore_index=True)  
    feature_path = DATA_NAME + '_planes.csv'
    df1.to_csv(OUTPUT_DIR + feature_path)
    written_files.append(feature_path)  
    logger.info(f"...done. A file was written: {feature_path}")

    # Cluster remaining points in point cloud  
    logger.info(f"Cluster remaining points")
    labels = np.array(rest.cluster_dbscan(eps=0.5, min_points=10))
    max_label = labels.max()
    logger.info(f"Point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab10")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    rest.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.io.write_point_cloud(output_dir + DATA_NAME + '_rest.ply', rest)

    points = np.asarray(rest.points)
    df2 = pd.DataFrame({'X':points[:,0],'Y':points[:,1],'Z':points[:,2]})
    df2['group'] = labels
    df2['type'] = 'cluster'
    feature_path = DATA_NAME + '_rest.csv'
    df2.to_csv(output_dir + feature_path)
    written_files.append(feature_path)  
    logger.info(f"...done. A file was written: {feature_path}")

    # Merge all planes and clusters  
    df3 = pd.DataFrame()
    df3=pd.concat([df1,df2],ignore_index=True)
    feature_path = DATA_NAME + '_all.csv'
    df3.to_csv(output_dir + feature_path)
    written_files.append(feature_path)  
    logger.info(f"...done. A file was written: {feature_path}")

    # Segmented point cloud vizualisation
    if VISU == 'True':
        geom.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=16), fast_normal_computation=True)
        geom.paint_uniform_color([0.6,0.6,0.6])
        o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_idx)]+[rest])

    print()
    logger.info("The following files were written. Let's check them out!")
    for written_file in written_files:
        logger.info(written_file)
    print()

    # Stop chronometer  
    toc = time.time()
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()