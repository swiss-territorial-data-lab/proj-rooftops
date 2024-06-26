import os
import sys
from argparse import ArgumentParser
from glob import glob
from loguru import logger
from yaml import load, FullLoader

import pandas as pd
from geopandas import read_file


sys.path.insert(1, 'scripts')
import functions.fct_misc as misc

logger = misc.format_logger(logger)

# Argument and parameter specification
parser = ArgumentParser()
parser.add_argument('config_file', type=str, help='Framework configuration file')
args = parser.parse_args()

logger.info(f"Using {args.config_file} as config file.")

with open(args.config_file) as fp:
    cfg = load(fp, Loader=FullLoader)[os.path.basename(__file__)]

# Define constants --------------
WORKING_DIR = cfg['working_directory']
INPUT_DIR = cfg['input_folder']
OUTPUT_DIR = cfg['output_folder']
OUTPUT_FILE = cfg['output_file']
EXPERT = cfg['expert']
APPEND = cfg['append']

if not OUTPUT_FILE.endswith('.xlsx'):
    OUTPUT_FILE = OUTPUT_FILE + '.xlsx'

os.chdir(WORKING_DIR)
OUTPUT_FILEPATH = os.path.join(misc.ensure_dir_exists(OUTPUT_DIR), os.path.basename(OUTPUT_FILE))

input_files = glob(os.path.join(INPUT_DIR, '*.gpkg'))
if len(input_files) == 0:
    logger.critical('No file to read.')
    sys.exit(1)

logger.info(F'Reading the input files from directory...')
try:
    occupation = read_file([file for file in input_files if 'roofs_occupation' in file][0])
    if 'agree' in occupation.columns:
        occupation.rename(columns={'agree': 'agreement'}, inplace=True)
except IndexError:
    occupation = pd.DataFrame(columns=['agreement', 'status'])
try:
    attributes_filtered = read_file([file for file in input_files if 'attributes_filtered' in file][0])
    attributes_filtered.loc[attributes_filtered['suitabilit'].isnull(), 'suitabilit'] = 'suitable'
except IndexError:
    attributes_filtered = pd.DataFrame(columns=['suitabilit'])

classes_occupation = occupation['status'].unique().tolist()
classes_filter = attributes_filtered['suitabilit'].unique().tolist()
satisfactions = dict.fromkeys(classes_occupation + classes_filter, [])

if not occupation.empty:
    logger.info('Calculating the satisfaction with the estimation of the occupation...')

    # Global satisfaction
    global_satisfaction_occupation = round(occupation['agreement'].sum()/occupation.shape[0], 3)
    logger.info(f'The expert agrees {global_satisfaction_occupation*100}% of the time with the results of the roof occupation.')

    # Per-class satisfaction
    for attribute_class in classes_occupation:
        roofs_subset = occupation[occupation['status'] == attribute_class]
        satisfactions[attribute_class] = round(roofs_subset['agreement'].sum()/roofs_subset.shape[0], 3)

if not attributes_filtered.empty:
    logger.info(f'Calculating the satisfaction with the filters on the attributes...')

    global_satisfaction_attributes = round(attributes_filtered['agreement'].sum()/attributes_filtered.shape[0], 3)
    logger.info(f'The expert agrees {global_satisfaction_attributes*100}% of the time with the results of the attributes-filtered roofs.')

    for attribute_class in classes_filter:
        roofs_subset = attributes_filtered[attributes_filtered['suitabilit'] == attribute_class]
        satisfactions[attribute_class] = round(roofs_subset['agreement'].sum()/roofs_subset.shape[0], 3)

satisfaction_df = pd.DataFrame(satisfactions, index=[EXPERT])

if not satisfaction_df.empty:
    if os.path.exists(OUTPUT_FILEPATH) and APPEND:
            with pd.ExcelWriter(OUTPUT_FILEPATH, mode="a", engine="openpyxl", if_sheet_exists="overlay") as writer:
                satisfaction_df.to_excel(writer, sheet_name="Sheet1", header=None, startrow=writer.sheets["Sheet1"].max_row, index=True)
    else:
        satisfaction_df.to_excel(OUTPUT_FILEPATH)

    logger.success(f'The file was written in {OUTPUT_FILEPATH}.')

else:
    logger.error('No satisfaction rate was properly caluclated.')