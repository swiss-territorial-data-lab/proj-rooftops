#!/bin/python
# -*- coding: utf-8 -*-

#  proj-rooftops


import os
import sys
from argparse import ArgumentParser
from loguru import logger
from time import time
from tqdm import tqdm
from yaml import load, FullLoader

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import mapping

import visvalingamwyatt as vw

sys.path.insert(1, 'scripts')
import functions.fct_misc as misc

logger = misc.format_logger(logger)


# Start chronometer
tic = time()
logger.info('Starting...')

# Argument and parameter specification
parser = ArgumentParser(description="The script prepares the point cloud dataset to be processed (STDL.proj-rooftops)")
parser.add_argument('config_file', type=str, help='Framework configuration file')
args = parser.parse_args()

logger.info(f"Using {args.config_file} as config file.")

with open(args.config_file) as fp:
    cfg = load(fp, Loader=FullLoader)[os.path.basename(__file__)]

# Load input parameters
WORKING_DIR = cfg['working_dir']
OUTPUT_DIR = cfg['output_dir']
ROOFS = cfg['roofs']

MERGE_FILES = cfg['merge_files']
BUFFER = cfg['buffer'] if 'buffer' in cfg.keys() else None
VW = cfg['vw'] if 'vw' in cfg.keys() else None
MERGE_DETECTIONS = cfg['merge_detections']


os.chdir(WORKING_DIR)
_ = misc.ensure_dir_exists(OUTPUT_DIR)

written_files = []

logger.info('Read file for roofs')

if ('EGID' in ROOFS) | ('egid' in ROOFS):
    roofs_gdf = gpd.read_file(ROOFS)
else:
    # Get the rooftops shapes
    _, ROOFS_NAME = os.path.split(ROOFS)
    attribute = 'EGID'
    original_file_path = ROOFS
    desired_file_path = os.path.join(OUTPUT_DIR, ROOFS_NAME[:-4] + "_" + attribute + ".shp")

    roofs_gdf = misc.dissolve_by_attribute(desired_file_path, original_file_path, name=ROOFS_NAME[:-4], attribute=attribute)

roofs_gdf['EGID'] = roofs_gdf['EGID'].astype(int)
logger.info(f'    - {roofs_gdf.shape[0]} roofs')

if MERGE_FILES:
    logger.info('Merge the different files for detections.')
    detections_dict = {}
    for name, filepath in cfg['files'].items():
        detections_dict[name] = gpd.read_file(filepath)

    if not ('all_detections' in detections_dict.keys()):
        logger.critical('None of the given files is designated as "all_detections".')
        sys.exit(1)

    detections_gdf = detections_dict['all_detections'].copy()
    del detections_dict['all_detections']

    for improved_dets in detections_dict.values():
        unchanged_detections_gdf = detections_gdf[~detections_gdf.EGID.isin(improved_dets.EGID.unique())].copy()
        detections_gdf = pd.concat([unchanged_detections_gdf, improved_dets], ignore_index=True)

    filepath = os.path.join(OUTPUT_DIR, 'detections.gpkg')
    detections_gdf.to_file(filepath)
    written_files.append(filepath)

else:
    detections_gdf = gpd.read_file(cfg['detections'])


if BUFFER:
    BUFFER_SIZE = cfg['buffer_size']

    logger.info('Simplify the geometries with a buffer')
    buffered_dets_gdf = detections_gdf.copy()
    buffered_dets_gdf['geometry'] = buffered_dets_gdf.buffer(BUFFER_SIZE)
    buffered_dets_gdf['geometry'] = buffered_dets_gdf.buffer(-BUFFER_SIZE)

    detections_gdf = buffered_dets_gdf.copy()


if VW:
    VW_THRESHOLD = cfg['vw_threshold']

    logger.info('Simplify the geometries with the Visvalingam-Wyatt algorithm')
    simplified_dets_gdf = gpd.GeoDataFrame()
    failed_transform = 0

    if 'MultiPolygon' in detections_gdf.geometry.geom_type.values:
        detections_gdf = detections_gdf.explode(ignore_index=True)
    mapped_objects = mapping(detections_gdf)
    for feature in tqdm(mapped_objects['features'], "Simplifying features"):
        coords = feature['geometry']['coordinates'][0]
        coords_post_vw = vw.Simplifier(coords).simplify(threshold=VW_THRESHOLD)
        if len(coords_post_vw) >= 3:
            feature['geometry']['coordinates'] = (tuple([tuple(arr) for arr in coords_post_vw]),)
            continue
        else:
            coords_post_vw = vw.Simplifier(coords).simplify(threshold=VW_THRESHOLD/2)
            if len(coords_post_vw) >= 3:
                feature['geometry']['coordinates'] = (tuple([tuple(arr) for arr in coords_post_vw]),)
                continue
            
        failed_transform += 1

    simplified_dets_gdf = gpd.GeoDataFrame.from_features(mapped_objects, crs='EPSG:2056')
    logger.info(f'{failed_transform} polygons on {simplified_dets_gdf.shape[0]} failed to be simplified.')

    simplified_dets_gdf = misc.check_validity(simplified_dets_gdf, correct=True)

    detections_gdf = simplified_dets_gdf.copy()

if MERGE_DETECTIONS:
    logger.info('Merge surfaces by type over EGID')
    dissolved_obstacles_gdf = detections_gdf[['EGID', 'occupation', 'geometry']].dissolve(['EGID', 'occupation']).explode(index_parts=False).reset_index()
    dissolved_obstacles_gdf['area'] = dissolved_obstacles_gdf.area
    dissolved_obstacles_gdf['det_id'] = dissolved_obstacles_gdf.index

    detections_gdf = dissolved_obstacles_gdf.copy()

if BUFFER or VW or MERGE_DETECTIONS:
    filepath = os.path.join(OUTPUT_DIR, f'{"merged" if MERGE_DETECTIONS else "simplified"}_detections{"_buffer" if BUFFER else ""}{"_VW" if VW else ""}.gpkg')
    detections_gdf.to_file(filepath)
    written_files.append(filepath)


logger.info('Deduce the free surface from the roof extend')
detections_gdf['occupation'] = detections_gdf['occupation'].astype(int)
occupied_surface_gdf = detections_gdf[detections_gdf.occupation==1].copy()
available_surface_gdf = gpd.overlay(roofs_gdf[['EGID', 'geometry']], occupied_surface_gdf[['det_id','geometry']], how='difference', keep_geom_type=True)
available_surface_gdf['area'] = available_surface_gdf.area

surface_partition_gdf = pd.concat([occupied_surface_gdf, available_surface_gdf], ignore_index=True)
surface_partition_gdf['surface_id'] = surface_partition_gdf.index
surface_partition_gdf.loc[surface_partition_gdf.occupation.isna(), 'occupation'] = 0

filepath = os.path.join(OUTPUT_DIR, 'roof_partition.gpkg')
surface_partition_gdf.to_file(filepath)
written_files.append(filepath)

print()
logger.success("The following files were written. Let's check them out!")
for written_file in written_files:
    logger.success(written_file)


# Stop chronometer
toc = time()
logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")
