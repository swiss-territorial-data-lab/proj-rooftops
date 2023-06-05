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
    cfg = load(fp, Loader=FullLoader)['rasterize_roughness.py']

# Define constants ----------------

WORKING_DIR=cfg['working_dir']
INPUT_DIR=cfg['input_dir']

OVERWRITE=cfg['overwrite'] if 'overwrite' in cfg.keys() else False

MAKE_DEM=cfg['make_dem']
PARAMETERS_DEM=cfg['parameters_dem']
RES=PARAMETERS_DEM['resolution']
RADIUS=PARAMETERS_DEM['radius']
if MAKE_DEM:
    MIN_Z=PARAMETERS_DEM['min_z']
    MAX_Z=PARAMETERS_DEM['max_z']
    MAX_EDGE=PARAMETERS_DEM['max_edge']

MAKE_RGH=cfg['make_rgh']
if MAKE_RGH:
    PARAMETERS_RGH=cfg['parameters_rgh']
    MIN_SCALE=PARAMETERS_RGH['min_scale']
    MAX_SCALE=PARAMETERS_RGH['max_scale']
    STEP=PARAMETERS_RGH['step']

OUTPUT_DIR_DEM=fct.ensure_dir_exists(os.path.join(WORKING_DIR, 'processed/lidar/rasterized_lidar/DEM'))
OUTPUT_DIR_RGH=fct.ensure_dir_exists(os.path.join(WORKING_DIR,'processed/lidar/rasterized_lidar/roughness'))

logger.info('Getting the list of files...')
lidar_files=glob(os.path.join(WORKING_DIR, INPUT_DIR, '*.las'))

logger.info('Treating files...')
for file in lidar_files:
    
    if '\\' in file:
        filename=file.split('\\')[-1].rstrip('.las')
        
    else:
        filename=file.split('/')[-1].rstirp('.las')

    output_path_dem=os.path.join(
                OUTPUT_DIR_DEM,
                filename + f'_{str(RES).replace(".", "pt")}_{str(RADIUS).replace(".", "pt")}.tif'
            )

    if MAKE_DEM:

        if (not os.path.isfile(output_path_dem)) | OVERWRITE:
            wbt.lidar_digital_surface_model(
                i=file, 
                output=output_path_dem, 
                resolution=RES, 
                radius=RADIUS, 
                minz=MIN_Z, 
                maxz=MAX_Z, 
                max_triangle_edge_length=MAX_EDGE,
            )

    if MAKE_RGH:

        output_path_mag=os.path.join(OUTPUT_DIR_RGH, 
                                    filename + f'_{MIN_SCALE}_{MAX_SCALE}_{STEP}.tif')
        
        if (not os.path.isfile(output_path_mag)) | OVERWRITE:
            output_path_scale=os.path.join(OUTPUT_DIR_RGH, 
                                    'scale_' + filename + f'_{MIN_SCALE}_{MAX_SCALE}_{STEP}.tif')
            wbt.multiscale_roughness(
                output_path_dem, 
                output_path_mag, 
                output_path_scale, 
                max_scale=MAX_SCALE, 
                min_scale=MIN_SCALE, 
                step=STEP
            )

if MAKE_DEM:
    logger.success(f'The files of the DEM were saved in the folder "{OUTPUT_DIR_DEM}".')
if MAKE_RGH:
    logger.success(f'The files of the roughness were saved in the folder "{OUTPUT_DIR_RGH}".')