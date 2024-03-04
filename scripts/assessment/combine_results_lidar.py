import os
import sys
from argparse import ArgumentParser
from loguru import logger
from time import time
from yaml import load, FullLoader

import geopandas as gpd
import pandas as pd

sys.path.insert(1, 'scripts')
import functions.fct_misc as misc
import assess_results
import calculate_free_area

logger = misc.format_logger(logger)

# Definitions of functions -------------------------------------------

def assess_free_surfaces(labels_gdf, detections_gdf, classified_roofs_gdf, office):
    written_files = []

    logger.info(f'Working with the data for the {office} office.')
    output_dir = misc.ensure_dir_exists(os.path.join(OUTPUT_DIR, office.upper()))
    office = office.lower()

    logger.info('Filter for labels and detections on potentially free surfaces')
    geoms_pot_free = classified_roofs_gdf.loc[classified_roofs_gdf[f'pred_status_{office}'] == 'occupied', 'geometry']
    geom_parts = geoms_pot_free.buffer(0.1, join_style=2).unary_union.buffer(-0.1, join_style=2)
    pot_free_gdf = gpd.GeoDataFrame(
        {'id': [i for i in range(len(geom_parts.geoms))], 'geometry': [geom for geom in geom_parts.geoms]}, 
        crs='EPSG:2056'
    )

    filtered_labels = gpd.overlay(labels_gdf, pot_free_gdf, how='difference', keep_geom_type=True)
    filepath = os.path.join(output_dir, 'labels_on_free_roofs.gpkg')
    filtered_labels.to_file(filepath)
    written_files.append(filepath)
    
    filtered_detections = gpd.overlay(detections_gdf, pot_free_gdf, how='difference', keep_geom_type=True)
    filepath = os.path.join(output_dir, 'occupation.gpkg')
    filtered_detections.to_file(filepath)
    written_files.append(filepath)

    logger.info('Assess the result')
    _, _, written_files_assessment = assess_results.main(WORKING_DIR, output_dir, filtered_labels, filtered_detections, EGIDS, ROOFS,
                                                method=METHOD, threshold=THRESHOLD,
                                                object_parameters=OBJECT_PARAMETERS, ranges=RANGES,
                                                additional_metrics=ADDITIONAL_METRICS, visualisation=VISU)
    written_files.extend(written_files_assessment)

    logger.info('Calculate the free area')
    written_files_area = calculate_free_area.main(WORKING_DIR, output_dir, filtered_labels, filtered_detections, ROOFS, EGIDS, BINS,
                         METHOD, visualisation=VISU)
    written_files.extend(written_files_area)
    
    return written_files


# Main ---------------------------------------------------------------

# Start chronometer
tic = time()
logger.info('Starting...')

# Argument and parameter specification
parser = ArgumentParser(description="The script allows to evaluate the workflow results (STDL.proj-rooftops)")
parser.add_argument('config_file', type=str, help='Framework configuration file')
args = parser.parse_args()

logger.info(f"Using {args.config_file} as config file.")

with open(args.config_file) as fp:
    cfg = load(fp, Loader=FullLoader)[os.path.basename(__file__)]

# Load input parameters
WORKING_DIR = cfg['working_dir']
OUTPUT_DIR = cfg['output_dir']

DETECTIONS=cfg['detections']
LABELS = cfg['ground_truth']
EGIDS = cfg['egids']
ROOFS = cfg['roofs']
CLASSIFIED_ROOFS = cfg['classified_roofs']
CLASSIFIED_ROOFS_LAYER = cfg['classified_roofs_layer']

METHOD = cfg['method']
THRESHOLD = cfg['threshold']

ADDITIONAL_METRICS = cfg['additional_metrics'] if 'additional_metrics' in cfg.keys() else False
OBJECT_PARAMETERS = cfg['object_attributes']['parameters']
AREA_RANGES = cfg['object_attributes']['area_ranges']
DISTANCE_RANGES = cfg['object_attributes']['distance_ranges']
BINS = cfg['bins']
VISU = cfg['visualisation'] if 'visualisation' in cfg.keys() else False

RANGES = [AREA_RANGES] + [DISTANCE_RANGES]

os.chdir(WORKING_DIR)
_ = misc.ensure_dir_exists(OUTPUT_DIR)

logger.info('Get input data')

_, _, labels_gdf, detections_gdf = misc.get_inputs_for_assessment(EGIDS, ROOFS, OUTPUT_DIR, LABELS, DETECTIONS)
classified_roofs_gdf = gpd.read_file(CLASSIFIED_ROOFS, layer=CLASSIFIED_ROOFS_LAYER)

written_files = assess_free_surfaces(labels_gdf, detections_gdf, classified_roofs_gdf, 'OCAN')

written_files.extend(assess_free_surfaces(labels_gdf, detections_gdf, classified_roofs_gdf, 'OCEN'))

logger.success("The following files were written. Let's check them out!")
for path in written_files:
    logger.success(f'  - {path}')
if VISU and ADDITIONAL_METRICS:
    logger.success('Some figures were also written.')

# Stop chronometer  
toc = time()
logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

sys.stderr.flush()
