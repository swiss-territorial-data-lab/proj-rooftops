import os
import sys
import argparse
from glob import glob
from loguru import logger
from yaml import load, FullLoader

import pandas as pd
from geopandas import read_file


sys.path.insert(1, 'scripts')
import functions.fct_misc as misc

logger = misc.format_logger(logger)

# Argument and parameter specification
parser = argparse.ArgumentParser()
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

input_files = glob(os.path.join(INPUT_DIR, '*.shp'))

logger.info(F'Reading the input files from directory...')
occupation = read_file([file for file in input_files if 'roofs_occupation' in file][0])
attributes_filtered = read_file([file for file in input_files if 'attributes_filtered' in file][0])
attributes_filtered.loc[attributes_filtered['suitabilit'].isnull(), 'suitabilit'] = 'suitable'

logger.info('Calculating the global satisfaction...')

global_satisfaction_occupation = round(occupation['agreement'].sum()/occupation.shape[0], 3)
logger.info(f'The expert agrees {global_satisfaction_occupation*100}% of the time with the results of the roof occupation.')

global_satisfaction_attributes = round(attributes_filtered['agreement'].sum()/attributes_filtered.shape[0], 3)
logger.info(f'The expert agrees {global_satisfaction_attributes*100}% of the time with the results of the attributes-filtered roofs.')

logger.info(f'Calculating the per-class satisfaction...')
classes_occupation = occupation['status'].unique().tolist()
classes_filter = attributes_filtered['suitabilit'].unique().tolist()

satisfactions = dict.fromkeys(classes_occupation + classes_filter, [])
for attribute_class in classes_occupation:
    roofs_subset = occupation[occupation['status'] == attribute_class]
    satisfactions[attribute_class] = round(roofs_subset['agreement'].sum()/roofs_subset.shape[0], 3)

for attribute_class in classes_filter:
    roofs_subset = attributes_filtered[attributes_filtered['suitabilit'] == attribute_class]
    satisfactions[attribute_class] = round(roofs_subset['agreement'].sum()/roofs_subset.shape[0], 3)

satisfaction_df = pd.DataFrame(satisfactions, index=[EXPERT])

if os.path.exists(OUTPUT_FILEPATH):
    if APPEND == True:
        with pd.ExcelWriter(OUTPUT_FILEPATH, mode="a", engine="openpyxl", if_sheet_exists="overlay") as writer:
            satisfaction_df.to_excel(writer, sheet_name="Sheet1", header=None, startrow=writer.sheets["Sheet1"].max_row, index=True)
    else:
        satisfaction_df.to_excel(OUTPUT_FILEPATH)
else:
    satisfaction_df.to_excel(OUTPUT_FILEPATH)

logger.success(f'The file was written in {OUTPUT_FILEPATH}')