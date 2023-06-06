import os, sys
from glob import glob
from loguru import logger
from tqdm import tqdm
from yaml import load, FullLoader

import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.features
from numpy import NaN
from shapely.geometry import shape
from rasterstats import zonal_stats

sys.path.insert(1, 'scripts')
import functions.fct_misc as fct

logger=fct.format_logger(logger)

logger.info(f"Using config.yaml as config file.")
with open('config/config.yaml') as fp:
    cfg = load(fp, Loader=FullLoader)['filter_surfaces_by_attributes.py']

# Define functions ---------------

def cause_occupation(df, message='Undefined cause'):
    '''
    Set the status to “occupied” and write the reason behind this.

    - df: dataframe of the occupied surface
    - message: message to write
    return: df with the column 'status' and 'reason'
    '''

    df['status']='occupied'
    df['reason']=message
    df.reset_index(drop=True, inplace=True)

    return df

def get_tilepath_from_id(tile_id, im_list):
    '''
    Get the tilepath containing the tile id from a list and check that there is no more than one.

    - tile_id: string with (part of) the tile name
    - im_list: list of the image/tile pathes
    return: matching tilepath
    '''

    matching_path=[tilepath for tilepath in im_list if tile_id in tilepath]
    if len(matching_path)>2:
        logger.critical(f'There are multiple tiles corresponding to the id {tile_id}.')
        sys.exit(1)
    else:
        tilepath=matching_path[0]

    return tilepath

# Define constants ----------------

DEBUG=False

WORKING_DIR=cfg['working_dir']
INPUT_DIR_IMAGES=cfg['input_dir_images']

LIDAR_TILES=cfg['lidar_tiles']
ROOFS=cfg['roofs']

# Parameter for eliminations
PROJECTED_AREA=2
NODATA_OVERLAP=0.25
LIM_STD=5500
LIM_MOE=400
Z=2
LIM_ROUGHNESS=7.5

STAT_LIMITS={'MOE_i': LIM_MOE, 'std_i': LIM_STD, 'median_r': LIM_ROUGHNESS}

os.chdir(WORKING_DIR)

logger.info('Getting the files...')
im_list_intensity=glob(os.path.join(INPUT_DIR_IMAGES, 'intensity', '*.tif'))
im_list_roughness=glob(os.path.join(INPUT_DIR_IMAGES, 'roughness', '*.tif'))
lidar_tiles=gpd.read_file(LIDAR_TILES)
roofs=gpd.read_file(ROOFS)

logger.info('Filtering for roofs clearly too small...')
small_roofs=roofs[roofs['SHAPE_AREA']<PROJECTED_AREA].reset_index(drop=True)
small_roofs=cause_occupation(small_roofs, f'projected area < {PROJECTED_AREA} m2')
small_roofs['nodata_overlap']=NaN

logger.info(f'{small_roofs.shape[0]} roof planes are classified as occupied' + 
            f' because their projected surface is smaller than {PROJECTED_AREA} m2.')
logger.info(f'{small_roofs.geometry.area.sum().round(1)} projected m2 were eliminated.')

large_roofs=roofs[~(roofs['SHAPE_AREA']<PROJECTED_AREA)].copy()

if DEBUG:
    large_roofs=large_roofs.sample(frac=0.1, ignore_index=True, random_state=1)

logger.info('Clipping labels...')
lidar_tiles.rename(columns={'fme_basena': 'id'}, inplace=True)
clipped_roofs=fct.clip_labels(large_roofs, lidar_tiles)
existing_clipped_roofs=clipped_roofs[~(clipped_roofs['geometry'].is_empty | clipped_roofs['geometry'].isna())].copy()

del large_roofs, clipped_roofs

# Get roofs planes with a high percentage of nodata values
# cf https://gis.stackexchange.com/questions/295362/polygonize-raster-file-according-to-band-values
logger.info('Getting zones with nodata value...')

nodata_polygons=[]
for tile_id in tqdm(lidar_tiles['id'].values, desc='Getting nodata area on tiles...'):

    if any(tile_id in tilepath for tilepath in im_list_intensity):

        tilepath=get_tilepath_from_id(tile_id, im_list_intensity)

        with rasterio.open(tilepath, crs='EPSG:2056') as src:
            intensity=src.read(1)

            shapes = list(rasterio.features.shapes(intensity, transform=src.transform))
            nodata_polygons.extend([shape(geom) for geom, value in shapes if value == src.nodata])

nodata_df=gpd.GeoDataFrame({'id': [i for i in range(len(nodata_polygons))], 'geometry': nodata_polygons}, crs='EPSG:2056')

logger.info('Getting the overlap between nodata values and the roof plans...')

existing_clipped_roofs['clipped_area']=existing_clipped_roofs.geometry.area
nodata_overlap=gpd.overlay(nodata_df, existing_clipped_roofs)
nodata_overlap['joined_area']=nodata_overlap.geometry.area

nodata_overlap_grouped=nodata_overlap[['OBJECTID', 'tile_id', 'joined_area']].groupby(by=['OBJECTID', 'tile_id']).sum().reset_index()
nodata_overlap_full=gpd.GeoDataFrame(nodata_overlap_grouped.merge(existing_clipped_roofs, how='right', on=['OBJECTID', 'tile_id']), crs='EPSG:2056')
nodata_overlap_full['nodata_overlap']=nodata_overlap_full['joined_area']/nodata_overlap_full['clipped_area']
nodata_overlap_full=nodata_overlap_full.round(3)

logger.info('Excluding roofs with too much area not classifed as building...')

no_roofs=nodata_overlap_full[nodata_overlap_full['nodata_overlap']>0.75].copy()
no_roofs['status']='undefined'
no_roofs['reason']='More than 75% of the roof area is not classified as building. Check wether it is a building or not.'
no_roofs.reset_index(drop=True, inplace=True)

building_roofs=nodata_overlap_full[(nodata_overlap_full['nodata_overlap']<=0.75) | (nodata_overlap_full['nodata_overlap'].isna())].copy()

other_classes_roofs=building_roofs[building_roofs['nodata_overlap']>NODATA_OVERLAP].copy()
other_classes_roofs=cause_occupation(other_classes_roofs, f'More than {NODATA_OVERLAP*100}% of the area is not classified as building.')

clipped_roofs_cleaned=building_roofs[(building_roofs['nodata_overlap']<=NODATA_OVERLAP) | (nodata_overlap_full['nodata_overlap'].isna())].reset_index(drop=True)

logger.info(f'{no_roofs.shape[0]} roofs are classified as undefined, because they do not overlap with the building class at more than 75%.')
logger.info(f'{other_classes_roofs.shape[0]} roofs are classified as occupied, '+
            f'because more than {NODATA_OVERLAP*100}% of their surface is not classified as building.')

del existing_clipped_roofs
del nodata_polygons, nodata_df
del nodata_overlap, nodata_overlap_grouped, nodata_overlap_full
del building_roofs

# Make stats per polygon
zs_per_roof=gpd.GeoDataFrame()
for tile_id in tqdm(lidar_tiles['id'].values, desc='Getting zonal stats from tiles...'):

    if any(tile_id in tilepath for tilepath in im_list_intensity) and any(tile_id in tilepath for tilepath in im_list_intensity):

        roofs_on_tile=clipped_roofs_cleaned[clipped_roofs_cleaned['tile_id']==tile_id].reset_index(drop=True)

        # Intensity statistics
        tilepath=get_tilepath_from_id(tile_id, im_list_intensity)
        
        with rasterio.open(tilepath, crs='EPSG:2056') as src:
            intensity=src.read(1)
            meta=src.meta
        
        meta.update({'crs': rasterio.crs.CRS.from_epsg(2056)})

        zs_df_intensty=pd.DataFrame(zonal_stats(roofs_on_tile, intensity, affine=meta['transform'],
                                        stats=['min', 'max', 'mean', 'median', 'std', 'count'], nodata=meta['nodata']))
        zs_per_roof_on_tile=pd.concat([roofs_on_tile, zs_df_intensty], axis=1)
        zs_per_roof_on_tile.rename(columns={'min': 'min_i', 'max': 'max_i', 'mean': 'mean_i', 'median': 'median_i',
                                            'std': 'std_i', 'count': 'count_i'}, inplace=True)
    
        # Roughness statistics
        tilepath=get_tilepath_from_id(tile_id, im_list_roughness)
        
        with rasterio.open(tilepath, crs='EPSG:2056') as src:
            roughness=src.read(1)
            meta=src.meta
        
        meta.update({'crs': rasterio.crs.CRS.from_epsg(2056)})

        zs_df_roughness=pd.DataFrame(zonal_stats(roofs_on_tile, roughness, affine=meta['transform'],
                                        stats=['min', 'max', 'mean', 'median', 'std', 'count'], nodata=meta['nodata']))
        zs_per_roof_on_tile=pd.concat([zs_per_roof_on_tile, zs_df_roughness], axis=1)
        zs_per_roof_on_tile.rename(columns={'min': 'min_r', 'max': 'max_r', 'mean': 'mean_r', 'median': 'median_r',
                                            'std': 'std_r', 'count': 'count_r'}, inplace=True)

        zs_per_roof=pd.concat([zs_per_roof, zs_per_roof_on_tile], ignore_index=True)
                              
    # else:
    #     logger.error(f'No raster found for the id {tile_id}.')

logger.info('Checking for already occupied roofs...')

# Check the marge or error
zs_per_roof['MOE_i'] = Z*zs_per_roof['std_i']/(zs_per_roof['count_i']**(1/2))

index_over_lim=[]
for attribute in STAT_LIMITS.keys():
    index_over_lim.extend(zs_per_roof.index[zs_per_roof[attribute] > STAT_LIMITS[attribute]].tolist())
seen = set()
dupes_index = [roof_index for roof_index in index_over_lim if roof_index in seen or seen.add(roof_index)]  

roofs_high_variability=zs_per_roof[zs_per_roof.index.isin(dupes_index)]
temp_roofs=zs_per_roof[~zs_per_roof.index.isin(dupes_index)]

roofs_high_std=temp_roofs[temp_roofs['std_i']>LIM_STD]
roofs_high_moe=temp_roofs[temp_roofs['MOE_i']>LIM_MOE]
rough_roofs=temp_roofs[temp_roofs['median_r']>LIM_ROUGHNESS]
zs_per_clear_roofs=temp_roofs[~((temp_roofs['MOE_i']>LIM_MOE) | (temp_roofs['std_i']>LIM_STD) | (temp_roofs['median_r']>LIM_ROUGHNESS))]

roofs_high_variability=cause_occupation(
    roofs_high_variability,
    f'Several parameters are over the set limits.'
)

roofs_high_moe=cause_occupation(roofs_high_moe, f'The marge of error of the mean for the intensity is over {LIM_MOE}.')

roofs_high_std=cause_occupation(roofs_high_std, f'The standard deviation for the intensity is over {LIM_STD}.')

rough_roofs=cause_occupation(rough_roofs, f'The median of the roughness is over {LIM_ROUGHNESS}.')

roofs_high_variability=pd.concat([roofs_high_variability, roofs_high_moe, roofs_high_std, rough_roofs], ignore_index=True)

zs_per_clear_roofs['status']='not occupied'

# If roofs appear several times, keep the largest surface
roofs_occupation=pd.concat([zs_per_clear_roofs, roofs_high_variability, small_roofs, other_classes_roofs, no_roofs], ignore_index=True)
roofs_occupation['clipped_area']=roofs_occupation.geometry.area
roofs_occupation_cleaned=roofs_occupation.sort_values('clipped_area', ascending=False).drop_duplicates('OBJECTID', ignore_index=True)

logger.info(f'{roofs_occupation.shape[0]-roofs_occupation_cleaned.shape[0]} geometries were dropped'+
            f' because they were duplicates due to the label clipping.')

# Reattach to original geometries
roofs_occupation_cleaned_df=pd.DataFrame(roofs_occupation_cleaned.drop(columns=['geometry', 'clipped_area']))
roofs_occupation_cleaned_gdf=gpd.GeoDataFrame(roofs_occupation_cleaned_df.merge(roofs[['OBJECTID', 'geometry']], on='OBJECTID', how='left'), crs='EPSG:2056')
roofs_occupation_cleaned_gdf=roofs_occupation_cleaned_gdf.round(3)
roofs_occupation_cleaned_gdf.drop(columns=['max_i', 'min_i', 'mean_i', 'max_r', 'min_r', 'mean_r', 'count_r',])

logger.info('Saving files...')
filepath=os.path.join('processed', 'tests', 'roofs.gpkg')

layername='roof_occupation'
roofs_occupation_cleaned_gdf.to_file(filepath, layer=layername)

logger.success(f'The files were written in the geopackage "{filepath}" in the layer {layername}.')