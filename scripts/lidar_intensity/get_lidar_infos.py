import os, sys
from loguru import logger
from glob import glob
from yaml import load, FullLoader

from whitebox import WhiteboxTools
wbt = WhiteboxTools()

sys.path.insert(1, 'scripts')
import functions.fct_misc as fct

logger=fct.format_logger(logger)

logger.info(f"Using config.yaml as config file.")
with open('config/config.yaml') as fp:
    cfg = load(fp, Loader=FullLoader)['get_lidar_infos.py']

# Define constants ----------------

WORKING_DIR=cfg['working_dir']
INPUT_DIR=cfg['input_dir']

OUTPUT_DIR_HTML=fct.ensure_dir_exists(os.path.join(WORKING_DIR,'processed/lidar_info'))

logger.info('Getting the list of files...')
lidar_files=glob(os.path.join(WORKING_DIR, INPUT_DIR))

logger.info('Treating files...')
for file in lidar_files:
    
    if '\\' in file:
        filename=file.split('\\')[-1].rstrip('.las')
        
    else:
        filename=file.split('/')[-1].rstirp('.las')

    output_path_html=os.path.join(OUTPUT_DIR_HTML, filename + '.html')

    wbt.lidar_info(
        file, 
        output_path_html, 
        density=True, 
        vlr=True, 
        geokeys=True,
    )

logger.success(f'The files were saved in the folder "{OUTPUT_DIR_HTML}".')