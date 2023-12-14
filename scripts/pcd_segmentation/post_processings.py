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
from shapely.geometry import mapping, Polygon
from shapely.validation import make_valid

from rdp import rdp
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

MERGE_FILES = cfg['merge_files']
RDP = cfg['rdp'] if 'rdp' in cfg.keys() else None
BUFFER = cfg['buffer'] if 'buffer' in cfg.keys() else None
VW = cfg['vw'] if 'vw' in cfg.keys() else None


os.chdir(WORKING_DIR)
_ = misc.ensure_dir_exists(OUTPUT_DIR)

written_files = []

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


if RDP:

    logger.info('Simplify the geometries with RDP')
    simplified_dets_gdf = gpd.GeoDataFrame()
    failed_transform = 0

    if 'MultiPolygon' in detections_gdf.geometry.geom_type.values:
        detections_gdf = detections_gdf.explode(ignore_index=True)
    mapped_objects = mapping(detections_gdf)
    for feature in tqdm(mapped_objects['features'], "Simplifying features"):
        coords = feature['geometry']['coordinates']
        coords_post_rdp = [rdp(x, epsilon=0.25) for x in coords]
        feature['geometry']['coordinates'] = tuple(coords_post_rdp)
        # try:
        #     if len(coords_post_rdp[0]) > 3:
        #         if (not Polygon(coords_post_rdp[0]).is_valid) & (make_valid(Polygon(coords_post_rdp[0])).geom_type=='Polygon'):
        #             coords_post_rdp = mapping(make_valid(Polygon(coords_post_rdp[0])))['coordinates']
        #         elif (not Polygon(coords_post_rdp[0]).is_valid):
        #             coords_post_rdp = coords
        #             failed_transform +=1

        #         feature['geometry']['coordinates'] = tuple(coords_post_rdp)
        #         tmp_gdf = gpd.GeoDataFrame.from_features(mapped_objects, crs='EPSG:2056')
        #     else:
        #         failed_transform += 1
        # except ValueError:
        #     failed_transform +=1

    simplified_dets_gdf = gpd.GeoDataFrame.from_features(mapped_objects, crs='EPSG:2056')
    logger.info(f'{failed_transform} polygons on {simplified_dets_gdf.shape[0]} failed to be simplified.')

    filepath = os.path.join(OUTPUT_DIR, 'simplified_detections_RDP.gpkg')
    simplified_dets_gdf.to_file(filepath)
    written_files.append(filepath)


if BUFFER:
    BUFFER_SIZE = cfg['buffer_size']

    logger.info('Simplify the geometries with a buffer')
    buffered_dets_gdf = detections_gdf.copy()
    buffered_dets_gdf['geometry'] = buffered_dets_gdf.buffer(BUFFER_SIZE)
    buffered_dets_gdf['geometry'] = buffered_dets_gdf.buffer(-BUFFER_SIZE)

    filepath = os.path.join(OUTPUT_DIR, 'simplified_detections_buffer.gpkg')
    buffered_dets_gdf.to_file(filepath)
    written_files.append(filepath)

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

    simplified_dets_gdf = misc.test_valid_geom(simplified_dets_gdf, correct=True, gdf_obj_name='simplified detections')

    filepath = os.path.join(OUTPUT_DIR, 'simplified_detections_VW.gpkg')
    simplified_dets_gdf.to_file(filepath)
    written_files.append(filepath)


print()
logger.success("The following files were written. Let's check them out!")
for written_file in written_files:
    logger.success(written_file)


# Stop chronometer
toc = time()
logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")
