#!/bin/python
# -*- coding: utf-8 -*-
# 
#  proj-rooftops

import argparse
import os
import sys
from glob import glob
from loguru import logger
from tqdm import tqdm
from yaml import load, FullLoader

import pandas as pd
import geopandas as gpd
import rasterio as rio
import rasterio.features
from numpy import NaN
from shapely.geometry import shape
from rasterstats import zonal_stats

sys.path.insert(1, 'scripts')
import functions.fct_misc as misc

logger = misc.format_logger(logger)


# Define parameters ---------------

# Argument and parameter specification
parser = argparse.ArgumentParser(description="The script calculate the zonal stats of intensity and roughness over roof planes.")
parser.add_argument('config_file', type=str, help='Framework configuration file')
args = parser.parse_args()

logger.info(f"Using {args.config_file} as config file.")
with open(args.config_file) as fp:
    cfg = load(fp, Loader=FullLoader)[os.path.basename(__file__)]

DEBUG = cfg['debug_mode']
CHECK_TILES = cfg['check_tiles']

WORKING_DIR = cfg['working_dir']
OUTPUT_DIR = cfg['output_dir']
INPUT_DIR_IMAGES = cfg['input_dir_rasters']

LIDAR_TILES = cfg['lidar_tiles']
ROOFS = cfg['roofs']

TILE_ID = cfg['tile_id']

# Parameters
PROJECTED_AREA = 2
Z = 2

# Main -------------------------

os.chdir(WORKING_DIR)
_ = misc.ensure_dir_exists(OUTPUT_DIR)

logger.info('Read the files and getting the tile paths...')
im_list_intensity = glob(os.path.join(INPUT_DIR_IMAGES, 'intensity', '*.tif'))
im_list_roughness = glob(os.path.join(INPUT_DIR_IMAGES, 'roughness', '*.tif'))
lidar_tiles = gpd.read_file(LIDAR_TILES)
lidar_tiles = lidar_tiles[[TILE_ID, 'geometry']].copy()
roofs = gpd.read_file(ROOFS)
roofs = roofs[['OBJECTID', 'ALTI_MIN', 'geometry']].copy()

logger.info('Filter roofs below threshold area...')
condition = roofs.area < PROJECTED_AREA
small_roofs = roofs[condition].reset_index(drop=True)
small_roofs['status'] = 'occupied'
small_roofs['reason'] = f'Projected area < {PROJECTED_AREA} m2'
small_roofs['nodata_overlap'] = NaN

logger.info(f'{small_roofs.shape[0]} roof planes are classified as occupied' + 
            f' because their projected surface is smaller than {PROJECTED_AREA} m2.')
logger.info(f'A total projected area of {small_roofs.geometry.area.sum().round(1)} m2 was eliminated.')

large_roofs = roofs[~condition].copy()

if DEBUG:
    large_roofs = large_roofs.sample(frac=0.1, ignore_index=True, random_state=1)

logger.info('Clip labels...')
lidar_tiles.rename(columns={TILE_ID: 'id'}, inplace=True)
clipped_roofs = misc.clip_labels(large_roofs, lidar_tiles)
clipped_roofs['tilepath_intensity'] = [
    misc.get_tilepath_from_id(tile_id, im_list_intensity) 
    if any(tile_id in tilepath for tilepath in im_list_intensity) else None 
    for tile_id in clipped_roofs['tile_id'].to_numpy().tolist()
]
clipped_roofs['tilepath_roughness'] = [
    misc.get_tilepath_from_id(tile_id, im_list_roughness) 
    if any(tile_id in tilepath for tilepath in im_list_roughness) else None 
    for tile_id in clipped_roofs['tile_id'].to_numpy().tolist()
]
existing_clipped_roofs = clipped_roofs[~(clipped_roofs['geometry'].is_empty | clipped_roofs['geometry'].isna() | clipped_roofs['tilepath_intensity'].isnull())].copy()

nbr_existing_clipped_roofs = existing_clipped_roofs.shape[0]

del large_roofs, clipped_roofs

# cf https://gis.stackexchange.com/questions/295362/polygonize-raster-file-according-to-band-values
nodata_polygons = []
for tile_id in tqdm(lidar_tiles['id'].values, desc='Transform nodata area to polygons...'):

    if any(tile_id in tilepath for tilepath in im_list_intensity):

        tilepath = misc.get_tilepath_from_id(tile_id, im_list_intensity)

        with rasterio.open(tilepath, crs='EPSG:2056') as src:
            intensity = src.read(1)

            shapes = list(rasterio.features.shapes(intensity, transform=src.transform))
            nodata_polygons.extend([shape(geom) for geom, value in shapes if value == src.nodata])

nodata_df = gpd.GeoDataFrame({'id': [i for i in range(len(nodata_polygons))], 'geometry': nodata_polygons}, crs='EPSG:2056')

logger.info('Get the overlap between nodata values and the roof planes...')

existing_clipped_roofs['clipped_area'] = existing_clipped_roofs.geometry.area
nodata_overlap = gpd.overlay(nodata_df, existing_clipped_roofs, keep_geom_type=True)
nodata_overlap['joined_area'] = nodata_overlap.geometry.area

nodata_overlap_grouped = nodata_overlap[['OBJECTID', 'tile_id', 'joined_area']].groupby(by=['OBJECTID', 'tile_id']).sum().reset_index()
nodata_overlap_full = gpd.GeoDataFrame(nodata_overlap_grouped.merge(existing_clipped_roofs, how='right', on=['OBJECTID', 'tile_id']), crs='EPSG:2056')
nodata_overlap_full['nodata_overlap'] = nodata_overlap_full['joined_area'] / nodata_overlap_full['clipped_area']
nodata_overlap_full.loc[nodata_overlap_full.nodata_overlap.isna(), 'nodata_overlap'] = 0
nodata_overlap_full = nodata_overlap_full.round(3)

logger.info('Exclude roofs not classified as building in the LiDAR point cloud...')
condition = nodata_overlap_full['nodata_overlap'] > 0.75

no_roofs = nodata_overlap_full[condition].reset_index(drop=True)
no_roofs['status'] = 'undefined'
no_roofs['reason'] = 'More than 75% of the roof area is not classified as building (in LiDAR point clooud). Check weather it is a building or not.'

building_roofs = nodata_overlap_full[(~condition) | (nodata_overlap_full['nodata_overlap'].isna())].copy()

nbr_no_roofs = no_roofs.shape[0]
logger.info(f'{nbr_no_roofs} roofs are classified as undefined, because they do not overlap with the building class at more than 75%.')

if (nbr_no_roofs + building_roofs.shape[0] != nbr_existing_clipped_roofs):
    logger.error(f'There is a difference of {nbr_no_roofs + building_roofs.shape[0]  - nbr_existing_clipped_roofs} roofs after filtering nodata values.')

del existing_clipped_roofs
del nodata_polygons, nodata_df
del nodata_overlap, nodata_overlap_grouped, nodata_overlap_full

# Compute stats for each polygon
zs_per_roof = gpd.GeoDataFrame()
for tile_id in tqdm(lidar_tiles['id'].unique(), desc='Getting zonal stats from tiles...'):

    if any(tile_id in tilepath for tilepath in im_list_intensity) and any(tile_id in tilepath for tilepath in im_list_roughness):

        roofs_on_tile = building_roofs[building_roofs['tile_id']==tile_id].reset_index(drop=True)

        # Intensity statistics
        tilepath = misc.get_tilepath_from_id(tile_id, im_list_intensity)
        
        with rio.open(tilepath, crs='EPSG:2056') as src:
            intensity = src.read(1)
            meta = src.meta
        
        meta.update({'crs': rio.crs.CRS.from_epsg(2056)})

        zs_df_intensty = pd.DataFrame(zonal_stats(roofs_on_tile, intensity, affine=meta['transform'],
                                        stats=['min', 'max', 'mean', 'median', 'std', 'count'], nodata=meta['nodata']))
        zs_per_roof_on_tile = pd.concat([roofs_on_tile, zs_df_intensty], axis=1)
        zs_per_roof_on_tile.rename(columns={'min': 'min_i', 'max': 'max_i', 'mean': 'mean_i', 'median': 'median_i',
                                            'std': 'std_i', 'count': 'count_i'}, inplace=True)
    
        # Roughness statistics
        tilepath = misc.get_tilepath_from_id(tile_id, im_list_roughness)
        
        with rio.open(tilepath, crs='EPSG:2056') as src:
            roughness = src.read(1)
            meta = src.meta
        
        meta.update({'crs': rio.crs.CRS.from_epsg(2056)})

        zs_df_roughness = pd.DataFrame(zonal_stats(roofs_on_tile, roughness, affine=meta['transform'],
                                        stats=['min', 'max', 'mean', 'median', 'std', 'count'], nodata=meta['nodata']))
        zs_per_roof_on_tile = pd.concat([zs_per_roof_on_tile, zs_df_roughness], axis=1)
        zs_per_roof_on_tile.rename(columns={'min': 'min_r', 'max': 'max_r', 'mean': 'mean_r', 'median': 'median_r',
                                            'std': 'std_r', 'count': 'count_r'}, inplace=True)

        zs_per_roof = pd.concat([zs_per_roof, zs_per_roof_on_tile], ignore_index=True)
                              
    elif CHECK_TILES:
      logger.error(f'No raster found for the id {tile_id}.')

# Compute the margin of error
zs_per_roof['MOE_i'] = Z*zs_per_roof['std_i'] / (zs_per_roof['count_i']**(1/2))

if (nbr_no_roofs + zs_per_roof.shape[0] != nbr_existing_clipped_roofs):
    logger.error(f'There is a difference of {nbr_no_roofs + zs_per_roof.shape[0]  - nbr_existing_clipped_roofs} roofs after filtering nodata values.')


roof_stats = pd.concat([zs_per_roof, small_roofs, no_roofs], ignore_index=True)
roof_stats['clipped_area'] = roof_stats.geometry.area

# roof_stats = roof_stats[~((roof_stats.count_i==0) & (roof_stats.clipped_area < 0.2))].copy()
condition= (roof_stats.count_i==0) | (roof_stats.count_r==0)
roof_stats.loc[condition, 'status'] = 'undefined'
roof_stats.loc[condition, 'reason'] = 'not enough values to determine zonal statistics'
if roof_stats[condition].shape[0] != 0:
    logger.warning(f'There are still {roof_stats[condition].shape[0]} roofs set as undefind because of missing zonal statistics.')

roof_stats_cleaned_df = roof_stats.sort_values('clipped_area', ascending=False).drop_duplicates('OBJECTID', ignore_index=True)
logger.info(f'{roof_stats.shape[0]-roof_stats_cleaned_df.shape[0]} geometries were deleted'+
            f' because they were affected by the clip to the tile.')

# Reattach to original geometries
roof_stats_cleaned_df = pd.DataFrame(roof_stats_cleaned_df.drop(columns=['geometry', 'clipped_area', 'tilepath_intensity', 'tilepath_roughness']))
roof_stats_cleaned_gdf = gpd.GeoDataFrame(roof_stats_cleaned_df.merge(roofs[['OBJECTID', 'geometry']], on='OBJECTID', how='left'), crs='EPSG:2056')
roof_stats_cleaned_gdf = roof_stats_cleaned_gdf.round(3)
roof_stats_cleaned_gdf.drop(
    columns=['tile_id', 'joined_area'],
    inplace=True
)

logger.info('Save file...')
filepath = os.path.join(OUTPUT_DIR, 'roofs.gpkg')
layername = 'roof_stats'
roof_stats_cleaned_gdf.to_file(filepath, layer=layername)

logger.success(f'The files were written in the geopackage "{filepath}" in the layer {layername}.')