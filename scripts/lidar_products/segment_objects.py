import os, sys
from glob import glob
from loguru import logger
from tqdm import tqdm
from yaml import load, FullLoader

import geopandas as gpd
import numpy as np
import rasterio

from rasterio.mask import mask

from skimage.color import gray2rgb
from skimage.graph import rag_mean_color, cut_normalized
from skimage.segmentation import felzenszwalb, quickshift, slic, watershed

sys.path.insert(1, 'scripts')
import functions.fct_misc as fct

logger=fct.format_logger(logger)

logger.info(f"Using config.yaml as config file.")
with open('config/config.yaml') as fp:
    cfg = load(fp, Loader=FullLoader)['segment_objects.py']

DEBUG=True
SAVE_NORMALIZED_INTESITY=False

WORKING_DIR=cfg['working_dir']
INPUT_DIR_IMAGES=cfg['input_dir_images']

LIDAR_TILES=cfg['lidar_tiles']

ROOF_OCCUPATION=cfg['roofs']
LAYER=cfg['roofs_layer']

METHOD=cfg['method']

os.chdir(WORKING_DIR)
OUTPUT_PATH='processed/roofs/merged_colors/segm_test.gpkg'
_ = fct.ensure_dir_exists(os.path.dirname(OUTPUT_PATH))

logger.info('Reading files...')

roofs=gpd.read_file(ROOF_OCCUPATION, layer=LAYER)
tiles=gpd.read_file(LIDAR_TILES)

im_list=glob(os.path.join(INPUT_DIR_IMAGES, '*.tif'))

# Data treatment  -----------------------

free_roofs=roofs[roofs['status']=='not occupied'].reset_index(drop=True)
free_roofs=free_roofs[['OBJECTID', 'EGID', 'geometry']]

if DEBUG:
    free_roofs=free_roofs.loc[:20]

logger.info(f'{free_roofs.shape[0]} roofs are estimed unoccupied.')

tiles_per_roof=gpd.sjoin(free_roofs, tiles, how='inner')
tiles_per_roof.reset_index(inplace=True, drop=True)
tiles_per_roof.rename(columns={'fme_basena': 'tile_id'}, inplace=True)

tiles_per_roof['tilepath']=[fct.get_tilepath_from_id(tile_id, im_list) for tile_id in tiles_per_roof['tile_id'].values]
tiles_per_roof = tiles_per_roof[~tiles_per_roof['tilepath'].isna()]

for roof_id in tqdm(tiles_per_roof['OBJECTID'].unique().tolist(), desc='Segmenting roof plans...'):

    needed_tiles=tiles_per_roof[tiles_per_roof['OBJECTID']==roof_id].reset_index(drop=True)

    for roof_and_tile in needed_tiles.itertuples():

        with rasterio.open(roof_and_tile.tilepath, crs='EPSG:2056') as src:
            im, mask_transform = mask(src, [roof_and_tile.geometry], crop=True, filled=False, nodata=src.nodata)
            im_profile=src.profile

        im_mask=~im.mask[0]
        intensity=im.data[0]

        nan_intensity=intensity.copy()
        nan_intensity[nan_intensity==im_profile['nodata']]=np.nan
        nan_intensity[im_mask==False]=np.nan

        nan_normalized_intensity = np.divide(
            nan_intensity - np.nanmin(nan_intensity),
            np.nanmax(nan_intensity)-np.nanmin(nan_intensity)
        )*255

        normalized_intensity=np.nan_to_num(nan_normalized_intensity, copy=True, nan=999)

        if SAVE_NORMALIZED_INTESITY:
            im_profile.update(transform=mask_transform, crs='EPSG:2056', width=normalized_intensity.shape[1], height=normalized_intensity.shape[0], nodata=999)
            with rasterio.open(os.path.join('processed/roofs/normalized_masked_intensity_images', str(roof_and_tile.OBJECTID) + '_normalized_intensity.tif'), 'w', **im_profile) as dst:
                dst.write(normalized_intensity,1)

        if METHOD=='felzenszwalb':
            scale=100
            sigma=0.4
            min_size=15
            segm_mask=felzenszwalb(normalized_intensity, scale=scale, sigma=sigma, min_size=min_size, channel_axis=None)

            LAYER=f'felzenszwalb_{scale}_{sigma}_{min_size}'

        elif METHOD=='quickshift':
            ratio=1
            kernel_size=5
            max_dist=10
            segm_mask=quickshift(normalized_intensity, ratio=ratio, kernel_size=kernel_size, max_dist=max_dist, convert2lab=False)

            LAYER=f'quickshift_{ratio}_{kernel_size}_{max_dist}'

        elif METHOD=='slic':
            n_segments=20
            compactness=0.01
            max_num_iter=10
            sigma=0
            enforce_connectivity=False
            min_size_factor=0.5
            slic_zero=False
            start_label=1
            segmented_slic_image=slic(normalized_intensity, n_segments=n_segments, compactness=compactness, max_num_iter=max_num_iter,
                           sigma=sigma, enforce_connectivity=enforce_connectivity, min_size_factor=min_size_factor, slic_zero=slic_zero,
                           start_label=start_label, mask=im_mask, channel_axis=None)
            
            # cf. https://stackoverflow.com/questions/58812104/skimage-rag-merging-for-gray-images
            sigma_rag=50
            image_three_bands=gray2rgb(normalized_intensity)
            graph_slic=rag_mean_color(image_three_bands, segmented_slic_image, mode='similarity', sigma=sigma_rag)
            segm_mask=cut_normalized(segmented_slic_image, graph_slic)
            
            LAYER=f'slic_{n_segments}_{compactness}_{max_num_iter}_{sigma}_{enforce_connectivity}_{min_size_factor}' +\
                        f'_{slic_zero}_{start_label}_{sigma_rag}'

        elif METHOD=='watershed':
            markers=200
            compactness=0.001
            watershed_line=False
            segm_mask=watershed(normalized_intensity, mask=im_mask, markers=markers, compactness=compactness, watershed_line=watershed_line)

            LAYER=f'watershed_{markers}_{compactness}_{watershed_line}'

        else:
            logger.error('No method corresponds to the chose one.')
            sys.exit(1)

        im_profile.update(transform=mask_transform, crs='EPSG:2056', width=segm_mask.shape[1], height=segm_mask.shape[0], nodata=0)
        with rasterio.open(os.path.join(os.path.dirname(OUTPUT_PATH), str(roof_and_tile.OBJECTID) + '_' + LAYER + '.tif'), 'w', **im_profile) as dst:
            dst.write(segm_mask,1)

logger.success(f'The file was written in {OUTPUT_PATH}.')