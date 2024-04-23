import argparse
import os
import sys
from glob import glob
from loguru import logger
from tqdm import tqdm
from yaml import load, FullLoader

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from osgeo import gdal
from osgeo import gdal_array
from rasterio.features import shapes
from rasterio.mask import mask
from rdp import rdp
from scipy.ndimage import binary_dilation
from shapely.geometry import shape, mapping
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

sys.path.insert(1, 'scripts')
import functions.fct_misc as misc

logger = misc.format_logger(logger)

# Argument and parameter specification
parser = argparse.ArgumentParser(description="The script detects the objects by filtering the intensity raster.")
parser.add_argument('config_file', type=str, help='Framework configuration file')
args = parser.parse_args()

logger.info(f"Using {args.config_file} as config file.")
with open(args.config_file) as fp:
    cfg = load(fp, Loader=FullLoader)[os.path.basename(__file__)]


# Constant definitions -------------------
DEBUG = cfg['debug_mode']

WORKING_DIR = cfg['working_dir']
INPUT_DIR_IMAGES = cfg['input_dir_images']

LIDAR_TILES = cfg['lidar_tiles']

ROOF_OCCUPATION = cfg['roofs']
LAYER = cfg['roofs_layer']

KERNEL = cfg['kernel']

PROCEDURE = cfg['parameters']['procedure']
RDP = cfg['parameters']['rdp']

os.chdir(WORKING_DIR)
OUTPUT_DIR = misc.ensure_dir_exists('processed/roofs')

logger.info('Reading files...')

roofs = gpd.read_file(ROOF_OCCUPATION, layer=LAYER)
tiles = gpd.read_file(LIDAR_TILES)

im_list = glob(os.path.join(INPUT_DIR_IMAGES, '*.tif'))

# Data processing  -----------------------

free_roofs = roofs[roofs['status']=='potentially free'].reset_index(drop=True)
free_roofs = free_roofs[['OBJECTID', 'EGID', 'ALTI_MAX', 'ALTI_MIN', 'geometry']]

nbr_roofs = free_roofs.shape[0]
if nbr_roofs != 0:
    logger.info(f'{nbr_roofs} roofs are estimed to be potentially free.')
else:
    logger.critical(f'No roofs are estimated to be potentially free.')
    sys.exit(1)

logger.info('Getting the median intensity by EGID...')

tiles_per_roof = gpd.sjoin(free_roofs, tiles, how='left')
tiles_per_roof.reset_index(inplace=True)
tiles_per_roof.rename(columns={'fme_basena': 'tile_id'}, inplace=True)

tiles_per_roof['tilepath'] = [misc.get_tilepath_from_id(tile_id, im_list) for tile_id in tiles_per_roof['tile_id'].values]
tiles_per_roof = tiles_per_roof[~tiles_per_roof['tilepath'].isna()]

objects = gpd.GeoDataFrame()
for roof_id in tqdm(tiles_per_roof['OBJECTID'].unique().tolist()):

    needed_tiles = tiles_per_roof[tiles_per_roof['OBJECTID']==roof_id].reset_index(drop=True)

    # Read masked tile(s)
    intensity_arrays = []
    if needed_tiles.shape[0] > 1:
        intensity_values = np.array([])
        for roof_and_tile in needed_tiles.itertuples():
                with rasterio.open(roof_and_tile.tilepath, crs='EPSG:2056') as src:
                    im, mask_transform = mask(src, [roof_and_tile.geometry], crop=True, filled=True, nodata=src.nodata)
                    im_profile = src.profile

                intensity_arrays.append([im, mask_transform, im_profile])
                intensity_values = np.concatenate([intensity_values, im.flatten()])
    else:
        with rasterio.open(needed_tiles.tilepath[0], crs='EPSG:2056') as src:
            intensity_values, mask_transform = mask(src, needed_tiles.geometry, crop=True, filled=True, nodata=src.nodata)
            im_profile = src.profile

            intensity_arrays.append([intensity_values, mask_transform, im_profile])

    nan_intensity_values = intensity_values.copy()
    nan_intensity_values[nan_intensity_values==im_profile['nodata']] = np.nan

    nan_normalized_intensity = np.divide(
        nan_intensity_values - np.nanmin(nan_intensity_values),
        np.nanmax(nan_intensity_values) - np.nanmin(nan_intensity_values)
    )
    normalized_intensity = nan_normalized_intensity[~np.isnan(nan_normalized_intensity)].flatten()

    if KERNEL:
        # Do kernel density
        fig, ax = plt.subplots()

        for method in ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']:
            grid = GridSearchCV(KernelDensity(kernel=method),
                            {'bandwidth': np.linspace(0.1, 5, 100)},
                            cv=20) # 20-fold cross-validation
            grid.fit(normalized_intensity[:, None])
            kde = grid.best_estimator_

            x_grid = np.linspace(np.nanmin(normalized_intensity), np.nanmax(normalized_intensity), 200)
            pdf = np.exp(kde.score_samples(x_grid[:, None]))

            ax.plot(x_grid, pdf, linewidth=3, alpha=0.5, label=f'{method}, {round(list(grid.best_params_.values())[0], 2)}')

        ax.hist(normalized_intensity.flatten(), 30, fc='gray', histtype='stepfilled', alpha=0.3)
        ax.legend(loc='upper left')
        fig.savefig(os.path.join(OUTPUT_DIR, str(roof_id) + '_density.jpeg'))

    # Determine the median value
    mean_intensity = np.nanmean(nan_intensity_values)
    std_intensity = np.nanstd(nan_intensity_values)

    for intensity_info in intensity_arrays:
        intensity_arr = intensity_info[0]
        mask_transform = intensity_info[1]
        im_profile = intensity_info[2]

        im_profile.update(transform=mask_transform, crs='EPSG:2056', width=intensity_arr.shape[2], height=intensity_arr.shape[1])

        # Filter roof according to the median and standard deviation
        factor = 1
        noisy_object_loc = np.where(intensity_arr > mean_intensity + factor*std_intensity, 1, 
                                    np.where(intensity_arr < mean_intensity - factor*std_intensity, 1, 0))

        # Eliminate noise
        if PROCEDURE == 'ds':
            dilated_object_loc = binary_dilation(noisy_object_loc, iterations=1).astype('int32')
            gdal_noisy_object_loc = gdal_array.OpenArray(dilated_object_loc)
            band = gdal_noisy_object_loc.GetRasterBand(1)
            gdal.SieveFilter(srcBand=band, maskBand=None, dstBand=band, threshold=15, connectedness=4)
            filtered_object_loc = band.ReadAsArray()
        elif PROCEDURE == 'sd':
            gdal_noisy_object_loc = gdal_array.OpenArray(noisy_object_loc)
            band = gdal_noisy_object_loc.GetRasterBand(1)
            gdal.SieveFilter(srcBand=band, maskBand=None, dstBand=band, threshold=2, connectedness=8)
            gdal_object_loc = band.ReadAsArray()
            filtered_object_loc = binary_dilation(gdal_object_loc, iterations=1).astype('int32')
        else:
            logger.error('No applicable procedure passed as parameter.')
            sys.exit(1)

        # Polygonize objects
        object_mask = filtered_object_loc == 1
 
        geoms = ((shape(s), v) for s, v in shapes(filtered_object_loc, object_mask, transform=im_profile['transform']))
        object_gdf = gpd.GeoDataFrame(geoms, columns=['geometry', 'class'])
        object_gdf.set_crs(crs=im_profile['crs'], inplace=True)

        roof = free_roofs.loc[free_roofs['OBJECTID']==roof_id, ['geometry', 'EGID']]
        objects_in_roof = object_gdf.overlay(roof)
        if 'MultiPolygon' in objects_in_roof.geometry.geom_type.values:
            objects_in_roof = objects_in_roof.explode(ignore_index=True)

        if RDP:
            mapped_objects = mapping(objects_in_roof)
            for feature in mapped_objects['features']:
                coords = feature['geometry']['coordinates']
                coords_post_rdp = []
                for coor in coords:
                    tmp = rdp(np.asarray(coor), epsilon=0.25)
                    coords_post_rdp.append(tuple(map(tuple, tmp.round(3))))
                feature['geometry']['coordinates'] = tuple(coords_post_rdp)

            try:
                objects = pd.concat([objects, gpd.GeoDataFrame.from_features(mapped_objects, crs='EPSG:2056')], ignore_index=True)
            except ValueError:
                objects = pd.concat([objects, objects_in_roof], ignore_index=True)
        else:
            objects = pd.concat([objects, objects_in_roof], ignore_index=True)


# Save filtered zones as objects
logger.info('Saving geodataframe to geopackage...')
filename = os.path.join(OUTPUT_DIR, 'roofs.gpkg')

if PROCEDURE == 'sd':
    layername = f'objects_on_roofs_{factor}_sd_rdp_{RDP}'
    # Avoid polygons thin as lines
    tmp = objects['geometry'].buffer(0.2)
    tmp2 = tmp.buffer(-0.2)
    objects['geometry'] = tmp2
elif PROCEDURE == 'ds':
    layername = f'objects_on_roofs_{factor}_ds_rdp_{RDP}'

objects.to_file(filename, layer=layername)
objects.to_file(filename, layer='objects_on_roofs')

logger.success(f'The results were saved in {filename} in the layer {layername}.')