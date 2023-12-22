import os
import sys
from argparse import ArgumentParser
from loguru import logger
from time import time
from yaml import load, FullLoader

import geopandas as gpd
import pandas as pd

sys.path.insert(1, 'scripts')
import functions.fct_figures as figures
import functions.fct_misc as misc
import functions.fct_metrics as metrics

logger = misc.format_logger(logger)

# Functions --------------------------

def main(WORKING_DIR, OUTPUT_DIR, LABELS, DETECTIONS, ROOFS, EGIDS, BINS, METHOD, visualisation=False):
    """Estimate the difference in free and occupied areas between labels and detections.

    Args:
        WORKING_DIR (path): working directory
        OUTPUT_DIR (path): output directory
        LABELS (path): file of the ground truth
        DETECTIONS (path): file of the detections
        ROOFS (path): file of the roof border and main elements
        EGIDS (list): EGIDs of interest
        method (string): method used for the assessment of the results, either one-to-one, one-to-many or many-to-many.
        visualisation (bool): wheter or not to do and save the plots. Defaults to False.

    Returns:
        tuple:
            - DataFrame: metrics computed for different attribute.
            - GeoDataFrame: missing labels (lost during the process)
    """

    os.chdir(WORKING_DIR)

    # Create an output directory in case it doesn't exist
    output_dir = misc.ensure_dir_exists(os.path.join(OUTPUT_DIR, METHOD))

    written_files = {}

    logger.info('Get input data')

    egids, roofs_gdf, labels_gdf, detections_gdf = misc.get_inputs_for_assessment(EGIDS, ROOFS, OUTPUT_DIR, LABELS, DETECTIONS)
    roofs_gdf['area'] = roofs_gdf.area

    # Get attribute keys
    roof_attributes = egids.keys().tolist()
    roof_attributes.remove('EGID')
    if 'nbr_elem' in roof_attributes:
        roof_attributes.remove('nbr_elem')

    logger.info('Get the free and occupied surface by EGID')
    
    egid_surfaces_df = egids.drop_duplicates(subset=['EGID']).reset_index(drop=True)
    egid_surfaces_df['total_area'] = [
        roofs_gdf.loc[roofs_gdf.EGID == egid, 'area'].iloc[0]
        if egid in roofs_gdf.EGID.unique() else 0
        for egid in egid_surfaces_df.EGID.unique()
    ]

    logger.info('    - for detections')
    detections_free_gdf = metrics.get_free_area(detections_gdf, roofs_gdf)
    egid_surfaces_df, surfaces_df, attribute_surface_df = metrics.area_estimations(detections_free_gdf, egid_surfaces_df, 'free', 'detections', BINS, roof_attributes)
    egid_surfaces_df, surfaces_df, attribute_surface_df = metrics.area_estimations(
        detections_free_gdf, egid_surfaces_df, 'occupied', 'detections', BINS, roof_attributes,
        surfaces_df=surfaces_df, attribute_surface_df=attribute_surface_df
    )

    if not labels_gdf.empty:
        logger.info('    - for labels')
        labels_free_gdf = metrics.get_free_area(labels_gdf, roofs_gdf)
        egid_surfaces_df, surfaces_df, attribute_surface_df = metrics.area_estimations(
            labels_free_gdf, egid_surfaces_df, 'free', 'labels', BINS, roof_attributes,
            surfaces_df=surfaces_df, attribute_surface_df=attribute_surface_df
        )
        egid_surfaces_df, surfaces_df, attribute_surface_df = metrics.area_estimations(
            labels_free_gdf, egid_surfaces_df, 'occupied', 'labels', BINS, roof_attributes,
            surfaces_df=surfaces_df, attribute_surface_df=attribute_surface_df
        )

        logger.info('Get the difference between labels and detections...')
        egid_surfaces_df, surfaces_df, attribute_surface_df = metrics.area_comparisons(egid_surfaces_df, surfaces_df, attribute_surface_df, 'free')
        egid_surfaces_df, surfaces_df, attribute_surface_df = metrics.area_comparisons(egid_surfaces_df, surfaces_df, attribute_surface_df, 'occupied')

        # Error of the occupied area relative to the total area
        for df in [egid_surfaces_df, surfaces_df, attribute_surface_df]:
            df['rel_occup_error'] = abs(df.occup_area_dets - df.occup_area_labels) / df.total_area

        # Assess surface bins, 0: different bin, 1: same bin -> if ok on one side (free/occupied), it should be ok on the other.
        egid_surfaces_df[f'assess_classif_bins'] = [
            1 if bin_area_det == bin_area_label else 0
            for bin_area_det, bin_area_label in zip(egid_surfaces_df[f'bin_free_area_dets (%)'], egid_surfaces_df[f'bin_free_area_labels (%)'])
        ]

        # Determine the global number of EGID occupied areas in the correct bin
        surfaces_df['right_bin'] = len(egid_surfaces_df[egid_surfaces_df['assess_classif_bins']==1])
        surfaces_df['wrong_bin'] = len(egid_surfaces_df[egid_surfaces_df['assess_classif_bins']==0])

        # Determine the global accuracy of detected areas
        surfaces_df['global_bin_accuracy'] = surfaces_df['right_bin'] / len(egid_surfaces_df['EGID'])

        # Determine the accuracy of detected surfaces by area bins
        for area_bin in egid_surfaces_df['bin_free_area_labels (%)'].sort_values().unique():
            surfaces_df['accuracy bin ' + area_bin] = len(
                egid_surfaces_df.loc[(egid_surfaces_df['bin_free_area_labels (%)']==area_bin) & (egid_surfaces_df['assess_classif_bins']==1)]
            ) \
                / len(egid_surfaces_df[egid_surfaces_df['bin_free_area_labels (%)']==area_bin])
            
        print()
        logger.info(f"Occupied surface relative error for all EGIDs = {(surfaces_df.loc[0,'occup_rel_diff'] ):.2f}")
        logger.info(f"Free surface relative error for all EGIDs = {(surfaces_df.loc[0, 'free_rel_diff']):.2f}")
        print()

        if visualisation:
            # Plots
            xlabel_dict = {'EGID': '', 'roof_type': '', 'roof_inclination': ''}
            bin_labels = [f"accuracy bin {BINS[i]}-{BINS[i+1]}" for i in range(len(BINS)-1)]

            _ = figures.plot_surface_bin(output_dir, surfaces_df, bins=bin_labels)
            for attr in attribute_surface_df.attribute.unique():
                if attr in xlabel_dict.keys():
                    filepath = figures.plot_surface(output_dir, attribute_surface_df, attribute=attr, xlabel=xlabel_dict[attr])
                    written_files[filepath] = ''


    logger.info('Save the calculated values...')

    # Save the values by EGID 
    feature_path = os.path.join(output_dir, 'EGID_surfaces.csv')
    egid_surfaces_df.drop(columns=[
        'nbr_elem', 
        'bin_free_area_dets (%)', 'bin_occup_area_dets (%)', 'bin_free_area_labels (%)', 'bin_occup_area_labels (%)', 'assess_classif_bins', 
        'ratio_occup_area_dets', 'ratio_occup_area_labels'
    ], errors='ignore').to_csv(feature_path, sep=',', index=False, float_format='%.3f')
    written_files[feature_path] = ''

    # Save the global values
    feature_path = os.path.join(output_dir, 'global_surfaces.csv')
    surfaces_df.to_csv(feature_path, sep=',', index=False, float_format='%.3f')
    written_files[feature_path] = '' 

    # Save the values by attribute
    feature_path = os.path.join(output_dir, 'surfaces_by_attributes.csv')
    attribute_surface_df.to_csv(feature_path, sep=',', index=False, float_format='%.3f')
    written_files[feature_path] = ''

    return written_files
    

if __name__ == "__main__":
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
    LABELS = cfg['ground_truth'] if 'ground_truth' in cfg.keys() else None
    ROOFS = cfg['roofs']
    EGIDS = cfg['egids']

    BINS = cfg['bins']
    METHOD = cfg['method']
    VISUALISATION = cfg['visualisation']

    written_files = main(WORKING_DIR, OUTPUT_DIR, LABELS, DETECTIONS, ROOFS, EGIDS, BINS,
                         METHOD, visualisation=VISUALISATION)

    logger.success("The following files were written. Let's check them out!")
    for path in written_files.keys():
        logger.success(f'  file: {path}{"" if written_files[path] == "" else f", layer: {written_files[path]}"}')

    # Stop chronometer  
    toc = time()
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()