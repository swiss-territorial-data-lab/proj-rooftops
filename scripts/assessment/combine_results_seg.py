import argparse
import os
import sys
import time

from loguru import logger
from yaml import load, FullLoader

import geopandas as gpd
import pandas as pd

sys.path.insert(1, 'scripts')
import functions.fct_misc as misc

logger = misc.format_logger(logger)


def read_gpd(DETECTIONS):

    if isinstance(DETECTIONS, str):
        detections_gdf = gpd.read_file(DETECTIONS)
    elif isinstance(DETECTIONS, gpd.GeoDataFrame):
        detections_gdf = DETECTIONS.copy()
    else:
        logger.critical(f'Unrecognized variable type for the detections: {type(DETECTIONS)}.')
        sys.exit(1)

    if 'occupation' in detections_gdf.columns:
        detections_gdf = detections_gdf[detections_gdf['occupation'].astype(int) == 1].copy()

    if 'det_id' in detections_gdf.columns:
        detections_gdf['detection_id'] = detections_gdf.det_id.astype(int)
    else:
        detections_gdf['detection_id'] = detections_gdf.index

    return detections_gdf


def main(WORKING_DIR, OUTPUT_DIR, DETECTIONS_PCD, DETECTIONS_IMG):
    """Assess the results by calculating the precision, recall and f1-score.

    Args:
        WORKING_DIR (path): working directory
        OUTPUT_DIR (path): output directory
        DETECTIONS_PCD (path): detection shapefile for LiDAR segementation
        DETECTIONS_IMG (path): detection shapefile for image segementation

    Returns:
    """

    os.chdir(WORKING_DIR)

    # Create an output directory in case it doesn't exist
    output_dir = misc.ensure_dir_exists(os.path.join(OUTPUT_DIR))

    written_files = {}

    logger.info("Get input data")

    # Read detection shapefiles from LiDAR segmentation (PCD) and image segmentation (IMG) 
    DETECTIONS_PCD = os.path.join(DETECTIONS_PCD)
    _ = misc.ensure_file_exists(DETECTIONS_PCD)
    pcd_gdf = gpd.read_file(DETECTIONS_PCD)
    pcd_gdf = pcd_gdf.rename(columns={'det_id': 'detection_id'})
    pcd_gdf = pcd_gdf[pcd_gdf['occupation'] == 1] 
    DETECTIONS_IMG = os.path.join(DETECTIONS_IMG)
    _ = misc.ensure_file_exists(DETECTIONS_IMG)
    img_gdf = read_gpd(DETECTIONS_IMG)

    logger.info(f"- {len(pcd_gdf)} detection shapes")
    logger.info(f"- {len(img_gdf)} detection shapes")

    # Concatenate img segmentation polygons with pcd segmentation polygons  
    final_gdf = pd.concat([img_gdf, pcd_gdf]).reset_index(drop=True)
    feature_path = os.path.join(output_dir, "roof_segmentation_concatenate.gpkg")
    final_gdf.to_file(feature_path, driver="GPKG") 

    # Filter img segmentation polygons with pcd segmentation polygons 
    left_join = gpd.sjoin(img_gdf, pcd_gdf, how='inner', predicate='intersects', lsuffix='img', rsuffix='pcd')
    left_join = left_join.rename(columns={"EGID_img": "EGID", "area_img": "area"})
    left_join = left_join.drop('EGID_pcd', axis=1)
    feature_path = os.path.join(output_dir, "roof_segmentation_sjoin.gpkg")
    left_join.to_file(feature_path, driver="GPKG") 

    return written_files


if __name__ == "__main__":

    # Start chronometer
    tic = time.time()
    logger.info("Combine results")
    logger.info('Starting...')

    # Argument and parameter specification
    parser = argparse.ArgumentParser(description="The script allows to combine LiDAR and image segmentation results (STDL.proj-rooftops)")
    parser.add_argument('config_file', type=str, help='Framework configuration file')
    args = parser.parse_args()

    logger.info(f"Using {args.config_file} as config file.")

    with open(args.config_file) as fp:
        cfg = load(fp, Loader=FullLoader)[os.path.basename(__file__)]

    # Load input parameters
    WORKING_DIR = cfg['working_dir']
    OUTPUT_DIR = cfg['output_dir']

    DETECTIONS_PCD = cfg['pcd_seg']
    # LAYER_PCD = cfg['layer_pcd']
    DETECTIONS_IMG = cfg['img_seg']

    written_files = main(WORKING_DIR, OUTPUT_DIR, DETECTIONS_PCD, DETECTIONS_IMG)

    logger.success("The following files were written. Let's check them out!")
    for path in written_files.keys():
        logger.success(f'  file: {path}{"" if written_files[path] == "" else f", layer: {written_files[path]}"}')

    # Stop chronometer  
    toc = time.time()
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()