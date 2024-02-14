import argparse
import os
import sys
import time
import torch

from glob import glob
from loguru import logger
from osgeo import gdal
from PIL import Image
from samgeo import SamGeo
from tqdm import tqdm
from yaml import load, FullLoader

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "caching_allocator"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

sys.path.insert(1, 'scripts')
import functions.fct_misc as misc

logger = misc.format_logger(logger)


def main(WORKING_DIR, IMAGE_DIR, OUTPUT_DIR, SHP_EXT, CROP, 
         DL_CKP, CKP_DIR, CKP, METHOD, THD_SIZE, TILE_SIZE, RESAMPLE,
         FOREGROUND, UNIQUE, MASK_MULTI, 
         VISU, CUSTOM_SAM, sam_dic={}):

    os.chdir(WORKING_DIR)

    # Create output directories in case they don't exist
    misc.ensure_dir_exists(OUTPUT_DIR)
    segmented_images_dir = misc.ensure_dir_exists(os.path.join(OUTPUT_DIR, 'segmented_images'))
    if METHOD == "resample":
        resampling_dir = misc.ensure_dir_exists(os.path.join(OUTPUT_DIR, 'resample'))

    written_files = []

    logger.info(f"Read the image file names")
    tiles = glob(os.path.join(IMAGE_DIR, '*.tif'))

    if CROP:
        logger.info(f"Images will be cropped with size {SIZE} and written to {IMAGE_DIR}.")

    # Select and download the SAM pretrained model
    if DL_CKP == True:
        dl_dir = misc.ensure_dir_exists(CKP_DIR)
        ckp_dir = os.path.join(os.path.expanduser('~'), dl_dir)
    elif DL_CKP == False:
        ckp_dir = CKP_DIR
    checkpoint = os.path.join(ckp_dir, CKP)
    logger.info(f"Select pretrained model: {CKP}")

    # Provide customized SAM parameters or use the default (samgeo) ones
    if CUSTOM_SAM == True:
        logger.info("Using custom SAM parameters")
        sam_kwargs = sam_dic
    else:
        logger.info("Using default SAM parameters")
        sam_kwargs = None
 
    # Define SAM properties
    sam = SamGeo(
        checkpoint=checkpoint,
        model_type='vit_h',
        device=device,
        sam_kwargs=sam_kwargs,
        )

    logger.info(f"Process images:") 
    logger.info(f"    - Object detection and mask saving")    
    logger.info(f"    - Convert mask to vector")  

    for tile in tqdm(tiles, desc='Applying SAM to tiles', total=len(tiles)):

        # Subdivide the input image in smaller tiles if its number of pixel exceed the threshold value
        directory, file = os.path.split(tile)
        img = Image.open(tile)
        width, height = img.size
        size = width * height
        tilepath = tile
        if size >= THD_SIZE:
            if METHOD == "batch":
                logger.info(f"Image too large to be processed -> subdivided in tiles of {TILE_SIZE} px")
                BATCH = True
            elif METHOD == "resample":
                logger.info(f"Image too large to be processed -> pixel resampling to {RESAMPLE} m per pixel")
                tilepath = os.path.join(resampling_dir, file)
                gdal.Warp(tilepath, tile, xRes=RESAMPLE, yRes=RESAMPLE, resampleAlg='cubic')
                BATCH = False
        else:
            BATCH = False

        # Crop the input image with pixel values
        if CROP:
            cropped_tilepath = misc.crop(tilepath, SIZE, IMAGE_DIR)
            written_files.append(cropped_tilepath)  
            tilepath = cropped_tilepath

        # Produce and save masks
        file_path = os.path.join(segmented_images_dir, file.split('.')[0] + '_segment.tif')       

        mask_path = file_path
        sam.generate(tilepath, mask_path, batch=BATCH, sample_size=(TILE_SIZE, TILE_SIZE), foreground=FOREGROUND, unique=UNIQUE, erosion_kernel=(3,3), mask_multiplier=MASK_MULTI)

        if os.path.exists(file_path):
            written_files.append(file_path)  

            # Convert segmentation masks to vector layer 
            file_path = os.path.join(segmented_images_dir, file.split('.')[0] + '_segment')  
        
            if SHP_EXT == 'gpkg': 
                sam.tiff_to_gpkg(mask_path, file_path, simplify_tolerance=None)
            elif SHP_EXT == 'shp':       
                sam.tiff_to_vector(mask_path, file_path)
            written_files.append(file_path)  

            if VISU:
                file_path = os.path.join(segmented_images_dir, file.split('.')[0] + '_annotated.tif')   
                sam.show_masks(cmap="binary_r")
                sam.show_anns(axis="off", alpha=0.7, output=file_path)
                written_files.append(file_path)
        else:
            pass

    print()
    logger.info("The following files were written. Let's check them out!")
    for written_file in written_files:
        logger.info(written_file)
    print()


if __name__ == "__main__":

    # Start chronometer
    tic = time.time()
    logger.info('Segment images')
    logger.info('Starting...')

    # Argument and parameter specification
    parser = argparse.ArgumentParser(description="The script segments georeferenced images using SAM and samgeo tools (STDL.proj-rooftops)")
    parser.add_argument('config_file', type=str, help='Framework configuration file')
    args = parser.parse_args()

    logger.info(f"Using {args.config_file} as config file.")
 
    with open(args.config_file) as fp:
        cfg = load(fp, Loader=FullLoader)[os.path.basename(__file__)]

    # Load input parameters
    WORKING_DIR = cfg['working_dir']
    IMAGE_DIR = cfg['image_dir']
    OUTPUT_DIR = cfg['output_dir']

    SHP_EXT = cfg['vector_extension']
    CROP = cfg['image_crop']['enable']
    if CROP == True:
        SIZE = cfg['image_crop']['size']
    else:
        CROP = None

    DL_CKP = cfg['SAM']['dl_checkpoints']
    CKP_DIR = cfg['SAM']['checkpoints_dir']
    CKP = cfg['SAM']['checkpoints']

    METHOD = cfg['SAM']['large_tile']['method']
    THD_SIZE = cfg['SAM']['large_tile']['thd_size']
    TILE_SIZE = cfg['SAM']['large_tile']['tile_size']
    RESAMPLE = cfg['SAM']['large_tile']['resample']

    FOREGROUND = cfg['SAM']['foreground']
    UNIQUE = cfg['SAM']['unique']
    MASK_MULTI = cfg['SAM']['mask_multiplier']
    CUSTOM_SAM = cfg['SAM']['custom_SAM']['enable']
    CUSTOM_PARAMETERS = cfg['SAM']['custom_SAM']['custom_parameters']
    VISU = cfg['SAM']['visualisation_masks']

    main(WORKING_DIR, IMAGE_DIR, OUTPUT_DIR, SHP_EXT, CROP, 
         DL_CKP, CKP_DIR, CKP, METHOD, THD_SIZE, TILE_SIZE, RESAMPLE,
         FOREGROUND, UNIQUE, MASK_MULTI, 
         VISU, CUSTOM_SAM, CUSTOM_PARAMETERS
         )

    # Stop chronometer  
    toc = time.time()
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()