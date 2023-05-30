import os, sys
from glob import glob
from loguru import logger
from tqdm import tqdm
from yaml import load, FullLoader

import pandas as pd
import geopandas as gpd
import rasterio
from rasterstats import zonal_stats

sys.path.insert(1, 'scripts')
import functions.fct_misc as fct

logger=fct.format_logger(logger)

logger.info(f"Using config.yaml as config file.")
with open('config/config.yaml') as fp:
    cfg = load(fp, Loader=FullLoader)['deviation_determination.py']

# Define functions ---------------

def cause_occupation(df, message='Undefined cause'):

    df['status']='occupied'
    df['reason']=message
    df.reset_index(drop=True, inplace=True)

    return df

# Define constants ----------------

WORKING_DIR=cfg['working_dir']
INPUT_DIR_IMAGES=cfg['input_dir_images']

LIDAR_TILES=cfg['lidar_tiles']
ROOFS=cfg['roofs']

# Parameter for eliminations
PROJECTED_AREA=2
LIM_STD=6000
LIM_MOE=750
Z=2

os.chdir(WORKING_DIR)

logger.info('Getting the files...')
im_list=glob(os.path.join(INPUT_DIR_IMAGES, '*.tif'))
lidar_tiles=gpd.read_file(LIDAR_TILES)
roofs=gpd.read_file(ROOFS)

logger.info('Filtering for roofs clearly too small...')
small_roofs=roofs[roofs['SHAPE_AREA']<PROJECTED_AREA].reset_index(drop=True)
small_roofs=cause_occupation(small_roofs, f'projected area < {PROJECTED_AREA} m2')

logger.info(f'{small_roofs.shape[0]+1} roof planes are classified as occupied' + 
            f' because their projected surface for those is smaller than {PROJECTED_AREA} m2.')
logger.info(f'{small_roofs.geometry.area.sum().round(1)} projected m2 were eliminated.')


large_roofs=roofs[~(roofs['SHAPE_AREA']<PROJECTED_AREA)].copy()

# Get roofs planes with a high percentage of nodata values
# cf https://gis.stackexchange.com/questions/295362/polygonize-raster-file-according-to-band-values

# -> Create polygons of nodata
# -> Calculate the area of intersection with the roof plane
# -> Exclude if the intersection is more than XX%


# Make stats per polygon
logger.info('Clipping labels...')
lidar_tiles.rename(columns={'fme_basena': 'id'}, inplace=True)
clipped_roofs=fct.clip_labels(large_roofs, lidar_tiles)
clipped_roofs_cleaned=clipped_roofs[~(clipped_roofs['geometry'].is_empty | clipped_roofs['geometry'].isna())].copy()

clipped_roofs_cleaned.to_file(os.path.join('processed', 'tests', 'test_clipped_roofs.shp'))

zs_per_clear_roofs=gpd.GeoDataFrame()
roofs_high_variability=gpd.GeoDataFrame()
roofs_high_std=gpd.GeoDataFrame()
roofs_high_moe=gpd.GeoDataFrame()
for tile_id in tqdm(lidar_tiles['id'].values, desc='Getting zonal stats from tiles...'):

    if any(tile_id in tilepath for tilepath in im_list):

        roofs_on_tile=clipped_roofs_cleaned[clipped_roofs_cleaned['tile_id']==tile_id].reset_index(drop=True)

        matching_path=[tilepath for tilepath in im_list if tile_id in tilepath]
        if len(matching_path)>1:
            logger.critical(f'There are multiple tiles corresponding to the id {tile_id}.')
            sys.exit(1)
        else:
            tilepath=matching_path[0]
        
        with rasterio.open(tilepath, crs='EPSG:2056') as src:
            intensity=src.read(1)
            meta=src.meta
        
        meta.update({'crs': rasterio.crs.CRS.from_epsg(2056)})

        zs_df=pd.DataFrame(zonal_stats(roofs_on_tile, intensity, affine=meta['transform'],
                                        stats=['min', 'max', 'mean', 'median', 'std', 'count'], nodata=meta['nodata']))
        zs_per_roof=pd.concat([roofs_on_tile, zs_df], axis=1)

        # Check the marge or error
        zs_per_roof['MOE'] = Z*zs_per_roof['std']/(zs_per_roof['count']**(1/2))

        roofs_high_variability=pd.concat([roofs_high_variability, zs_per_roof[(zs_per_roof['std']>LIM_STD) & (zs_per_roof['MOE']>LIM_MOE)]])
        temp_roofs=zs_per_roof[~((zs_per_roof['MOE']>LIM_MOE) & (zs_per_roof['std']>LIM_STD))]

        roofs_high_std=pd.concat([roofs_high_std, temp_roofs[temp_roofs['std']>LIM_STD]], ignore_index=True)
        roofs_high_moe=pd.concat([roofs_high_moe, temp_roofs[temp_roofs['MOE']>LIM_MOE]], ignore_index=True)
        zs_per_clear_roofs=pd.concat(
            [zs_per_clear_roofs, temp_roofs[~((temp_roofs['MOE']>LIM_MOE) | (temp_roofs['std']>LIM_STD))]], 
            ignore_index=True
        )

    # else:
    #     logger.error(f'No raster found for the id {tile_id}.')

roofs_high_variability=cause_occupation(
    roofs_high_variability,
    f'The marge of error of the mean and the standard deviation are both over the limit, respectively {LIM_MOE} and {LIM_STD}.'
)

roofs_high_moe=cause_occupation(roofs_high_moe, f'The marge of error of the mean is over {LIM_MOE}.')

roofs_high_std=cause_occupation(roofs_high_std, f'The standard deviation is over {LIM_STD}.')

roofs_high_variability=pd.concat([roofs_high_variability, roofs_high_moe, roofs_high_std], ignore_index=True)


# If roofs appear several times, keep the largest surface
roofs_occupation=pd.concat([zs_per_clear_roofs, roofs_high_variability, small_roofs], ignore_index=True)
roofs_occupation['area']=roofs_occupation.geometry.area
roofs_occupation_cleaned=roofs_occupation.sort_values('area', ascending=False).drop_duplicates('OBJECTID', ignore_index=True)

logger.info(f'{roofs_occupation.shape[0]-roofs_occupation_cleaned.shape[0]} geometries were dropped because they were duplicates due to the label clipping.')

# Reattach to original geometries
roofs_occupation_cleaned_df=pd.DataFrame(roofs_occupation_cleaned.drop(columns=['geometry', 'area']))
roofs_occupation_cleaned_gdf=gpd.GeoDataFrame(roofs_occupation_cleaned_df.merge(roofs[['OBJECTID', 'geometry']], on='OBJECTID', how='left'))

logger.info('Saving files...')
roofs_occupation_cleaned_gdf.to_file(os.path.join('processed', 'tests', 'roofs.gpkg'), layer='roof_occupation')

small_roofs.to_file(os.path.join('processed', 'tests', 'roofs.gpkg'), layer='small_roofs', driver="GPKG")
roofs_high_variability.to_file(os.path.join('processed', 'tests', 'roofs.gpkg'), layer='variable_roofs')

roofs_high_std.to_file(os.path.join('processed', 'tests', 'roofs.gpkg'), layer='variable_roofs_std')
roofs_high_moe.to_file(os.path.join('processed', 'tests', 'roofs.gpkg'), layer='variable_roofs_moe')