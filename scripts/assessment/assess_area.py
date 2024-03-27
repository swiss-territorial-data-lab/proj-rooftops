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

def main(WORKING_DIR, OUTPUT_DIR, LABELS, DETECTIONS, ROOFS, EGIDS, METHOD, visualization=False):
    """Estimate the difference in free and occupied areas between labels and detections.

    Args:
        WORKING_DIR (path): working directory
        OUTPUT_DIR (path): output directory
        LABELS (path): file of the ground truth
        DETECTIONS (path): file of the detections
        ROOFS (path): file of the roof border and main elements
        EGIDS (list): EGIDs of interest
        METHOD (string): method used for the assessment of the results, either one-to-one, one-to-many or many-to-many.
        visualization (bool): wheter or not to do and save the plots. Defaults to False.

    Returns:
        tuple:
            - DataFrame: metrics computed for different attribute.
            - GeoDataFrame: missing labels (lost during the process)
    """

    os.chdir(WORKING_DIR)

    # Create an output directory in case it doesn't exist
    output_dir = misc.ensure_dir_exists(os.path.join(OUTPUT_DIR, 'assessment_' + METHOD))

    written_files = {}

    logger.info('Get input data')

    egids, roofs_gdf, labels_gdf, detections_gdf = misc.get_inputs_for_assessment(EGIDS, ROOFS, LABELS, DETECTIONS)
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
    egid_surfaces_df, surfaces_df, attribute_surfaces_df = metrics.area_estimation(detections_free_gdf, egid_surfaces_df, 'free', 'detections', roof_attributes)
    egid_surfaces_df, surfaces_df, attribute_surfaces_df = metrics.area_estimation(
        detections_free_gdf, egid_surfaces_df, 'occupied', 'detections', roof_attributes,
        surfaces_df=surfaces_df, attribute_surfaces_df=attribute_surfaces_df
    )

    if not labels_gdf.empty:
        logger.info('    - for labels')
        labels_free_gdf = metrics.get_free_area(labels_gdf, roofs_gdf)
        egid_surfaces_df, surfaces_df, attribute_surfaces_df = metrics.area_estimation(
            labels_free_gdf, egid_surfaces_df, 'free', 'labels', roof_attributes,
            surfaces_df=surfaces_df, attribute_surfaces_df=attribute_surfaces_df
        )
        egid_surfaces_df, surfaces_df, attribute_surfaces_df = metrics.area_estimation(
            labels_free_gdf, egid_surfaces_df, 'occupied', 'labels', roof_attributes,
            surfaces_df=surfaces_df, attribute_surfaces_df=attribute_surfaces_df
        )

        logger.info('Get the difference between labels and detections...')
        egid_surfaces_df, surfaces_df, attribute_surfaces_df = metrics.area_comparisons(egid_surfaces_df, surfaces_df, attribute_surfaces_df, 'free')
        egid_surfaces_df, surfaces_df, attribute_surfaces_df = metrics.area_comparisons(egid_surfaces_df, surfaces_df, attribute_surfaces_df, 'occupied')

        print()
        logger.info(f"Occupied surface relative error for all EGIDs = {(surfaces_df.loc[0,'occup_rel_error'] ):.2f}")
        logger.info(f"Free surface relative error for all EGIDs = {(surfaces_df.loc[0, 'free_rel_error']):.2f}")
        print()

        for df in [egid_surfaces_df, surfaces_df, attribute_surfaces_df]:
            df['occup_error_norm'] = abs(df.occup_area_dets - df.occup_area_labels) / df.total_area

        attribute_surfaces_df = pd.concat([surfaces_df, attribute_surfaces_df])
        if visualization:
            # Plots
            xlabel_dict = {'all': '', 'building_type': 'Building type', 'roof_type': 'Roof type'}

            for attr in attribute_surfaces_df.attribute.unique():
                if attr in xlabel_dict.keys():
                    filepath = figures.plot_area(output_dir, attribute_surfaces_df, attribute=attr, xlabel=xlabel_dict[attr])
                    written_files[filepath] = ''


    logger.info('Save the calculated values...')

    # Save the values by EGID 
    feature_path = os.path.join(output_dir, 'surfaces_by_EGID.csv')
    egid_surfaces_df.drop(columns=[
        'nbr_elem', 'ratio_occup_area_dets', 'ratio_occup_area_labels'
    ], errors='ignore').to_csv(feature_path, sep=',', index=False, float_format='%.3f')
    written_files[feature_path] = ''

    # Save the values by attribute
    feature_path = os.path.join(output_dir, 'surfaces_by_attributes.csv')
    attribute_surfaces_df.to_csv(feature_path, sep=',', index=False, float_format='%.3f')
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
    OUTPUT_DIR = cfg['output_dir'] in the configuration file

    DETECTIONS = cfg['detections']
    LABELS = cfg['ground_truth'] if 'ground_truth' in cfg.keys() else None
    ROOFS = cfg['roofs']
    EGIDS = cfg['egids']

    METHOD = cfg['method']
    visualization = cfg['visualization']

    written_files = main(WORKING_DIR, OUTPUT_DIR, LABELS, DETECTIONS, ROOFS, EGIDS,
                         METHOD, visualization=visualization)

    logger.success("The following files were written. Let's check them out!")
    for path in written_files.keys():
        logger.success(f'  file: {path}{"" if written_files[path] == "" else f", layer: {written_files[path]}"}')

    # Stop chronometer  
    toc = time()
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()