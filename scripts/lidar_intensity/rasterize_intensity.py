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
    cfg = load(fp, Loader=FullLoader)['rasterize_intensity.py']

# Define constants ----------------

WORKING_DIR=cfg['working_dir']
INPUT_DIR=cfg['input_dir']

OVERWRITE=cfg['overwrite'] if 'overwrite' in cfg.keys() else False

PARAMETERS=cfg['parameters']
METHOD=PARAMETERS['method'].lower()
RES=PARAMETERS['res']
RADIUS=PARAMETERS['radius']
RETURNS=PARAMETERS['returns']

OUTPUT_DIR_TIF=fct.ensure_dir_exists(os.path.join(WORKING_DIR,'processed/rasterized_lidar'))

logger.info('Getting the list of files...')
lidar_files=glob(os.path.join(WORKING_DIR, INPUT_DIR, '*.las'))

logger.info('Treating files...')
for file in lidar_files:
    
    if '\\' in file:
        filename=file.split('\\')[-1].rstrip('.las')
        
    else:
        filename=file.split('/')[-1].rstirp('.las')

    output_path_tif=os.path.join(OUTPUT_DIR_TIF, 
                                 filename + f'_{METHOD}_{str(RES).replace(".", "pt")}_{str(RADIUS).replace(".", "pt")}_{RETURNS}.tif')
    
    if (not os.path.isfile(output_path_tif)) | OVERWRITE:
        if METHOD=='idw':
            wbt.lidar_idw_interpolation(
                i=file, 
                output=output_path_tif, 
                parameter="intensity", 
                returns=RETURNS,
                exclude_cls='1,2,3,5,7,9,13,15,16,19',
                radius=RADIUS,
                resolution=RES,
            )
        elif METHOD=='nnb':
            wbt.lidar_nearest_neighbour_gridding(
                i=file, 
                output=output_path_tif, 
                parameter="intensity", 
                returns=RETURNS,
                exclude_cls='1,2,3,5,7,9,13,15,16,19',
                radius=RADIUS,
                resolution=RES,
            )
        else:
            logger.error('This method of interpolation is not supported. Please, pass "idw" or "nnb" as parameter.')

logger.success(f'The files were saved in the folder "{OUTPUT_DIR_TIF}".')