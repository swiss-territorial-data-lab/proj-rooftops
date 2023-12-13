import os
import sys
import re
import torch
from glob import glob
from loguru import logger
from tqdm import tqdm

import pandas as pd
import geopandas as gpd
from samgeo import SamGeo
sys.path.insert(1, 'scripts')
import functions.fct_misc as misc


def SAM_mask(IMAGE_DIR, OUTPUT_DIR, SIZE, CROP, SHP_ROOF, DL_CKP, CKP_DIR, CKP, BATCH, FOREGROUND, UNIQUE, MASK_MULTI, SHP_EXT, dict_params, written_files):
    
    logger.info(f"Read images file name")
    tiles = glob(os.path.join(IMAGE_DIR, '*.tif'))

    if '\\' in tiles[0]:
        tiles = [tile.replace('\\', '/') for tile in tiles]
      
    for tile in tqdm(tiles, desc='Applying SAM to tiles', total=len(tiles)):

        logger.info(f"Read images: {os.path.basename(tile)}") 
        image = tile

        # Crop the input image by pixel value
        if CROP:
            logger.info(f"Crop image with size {SIZE}") 
            image = misc.crop(image, SIZE, IMAGE_DIR)
            written_files.append(image)  
            logger.info(f"...done. A file was written: {image}") 

        egid = float(re.sub('[^0-9]','', os.path.basename(tile)))
        roofs = gpd.read_file(SHP_ROOF)
        shp_egid = roofs[roofs['EGID'] == egid]

        logger.info(f"Perform image segmentation with SAM")  
        if DL_CKP == True:
            dl_dir = os.path.join(CKP_DIR)
            if not os.path.exists(dl_dir):
                os.makedirs(dl_dir)
            ckp_dir = os.path.join(os.path.expanduser('~'), dl_dir)
        elif DL_CKP == False:
            ckp_dir = CKP_DIR
        checkpoint = os.path.join(ckp_dir, CKP)
        logger.info(f"Select pretrained model: {CKP}")   

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        sam_kwargs = {
            'points_per_side': dict_params["points_per_side"],
            'points_per_batch': dict_params["points_per_batch"],
            'pred_iou_thresh': dict_params["pred_iou_thresh"],
            'stability_score_thresh': dict_params["stability_score_thresh"], 
            'stability_score_offset': dict_params["stability_score_offset"],
            'box_nms_thresh': dict_params["box_nms_thresh"],
            'crop_n_layers': dict_params["crop_n_layers"],
            'crop_nms_thresh': dict_params["crop_nms_thresh"],
            'crop_overlap_ratio': dict_params["crop_overlap_ratio"],
            'crop_n_points_downscale_factor': dict_params["crop_n_points_downscale_factor"],
            'min_mask_region_area': dict_params["min_mask_region_area"]
            }

        sam = SamGeo(
            checkpoint=checkpoint,
            model_type='vit_h',
            device=device,
            sam_kwargs=sam_kwargs,
            )

        logger.info(f"Produce and save mask")  
        file_path=os.path.join(misc.ensure_dir_exists(os.path.join(OUTPUT_DIR, 'segmented_images')),
                                tile.split('/')[-1].split('.')[0] + '_segment.tif')       
        
        mask = file_path
        sam.generate(image, mask, batch=BATCH, foreground=FOREGROUND, unique=UNIQUE, erosion_kernel=(3,3), mask_multiplier=MASK_MULTI)
        written_files.append(file_path)  
        logger.info(f"...done. A file was written: {file_path}")

        file_path=os.path.join(misc.ensure_dir_exists(os.path.join(OUTPUT_DIR, 'segmented_images')),
                                tile.split('/')[-1].split('.')[0] + '_colormask.tif')   
        # sam.show_masks(cmap="binary_r")
        # sam.show_anns(axis="off", alpha=0.7, output=file_path)

        logger.info(f"Convert segmentation mask to vector layer")  
        if SHP_EXT == 'gpkg': 
            file_path=os.path.join(misc.ensure_dir_exists(os.path.join(OUTPUT_DIR, 'segmented_images')),
                    tile.split('/')[-1].split('.')[0] + '_segment.gpkg')       
            sam.tiff_to_gpkg(mask, file_path, simplify_tolerance=None)

            written_files.append(file_path)  
            logger.info(f"...done. A file was written: {file_path}")
        elif SHP_EXT == 'shp': 
            file_path=os.path.join(misc.ensure_dir_exists(os.path.join(OUTPUT_DIR, 'segmented_images')),
                    tile.split('/')[-1].split('.')[0] + '_segment.shp')        
            sam.tiff_to_vector(mask, file_path)
            written_files.append(file_path)  
            logger.info(f"...done. A file was written: {file_path}")


def filter(OUTPUT_DIR, SHP_ROOFS, SRS, DETECTION, SHP_EXT, written_files):

    # Get the rooftop shapes
    ROOFS_DIR, ROOFS_NAME = os.path.split(SHP_ROOFS)
    feature_path = os.path.join(ROOFS_DIR, ROOFS_NAME[:-4]  + "_EGID.shp")

    if os.path.exists(feature_path):
        logger.info(f"File {ROOFS_NAME[:-4]}_EGID.shp already exists")
        rooftops = gpd.read_file(feature_path)
    else:
        logger.info(f"File {ROOFS_NAME[:-4]}_EGID.shp does not exist")
        logger.info(f"Create it")
        gdf_roofs = gpd.read_file(SHP_ROOFS)
        logger.info(f"Dissolved shapes by EGID number")
        rooftops = gdf_roofs.dissolve('EGID', as_index=False)
        rooftops.drop(['OBJECTID', 'ALTI_MAX', 'DATE_LEVE', 'SHAPE_AREA', 'SHAPE_LEN'], axis=1)
        rooftops.to_file(feature_path)
        written_files.append(feature_path)  
        logger.info(f"...done. A file was written: {feature_path}")

    # Read all the shapefiles produced, filter them with rooftop extension and merge them into a single layer  
    logger.info(f"Read shapefiles' name")
    tiles=glob(os.path.join(OUTPUT_DIR + 'segmented_images',  '*.' + SHP_EXT))

    vector_layer = gpd.GeoDataFrame() 

    for tile in tqdm(tiles, desc='Read detection shapefiles', total=len(tiles)):

        logger.info(f"Read shapefile: {os.path.basename(tile)}")
        objects = gpd.read_file(tile)

        # Set CRS
        objects.crs = SRS
        shape_objects = objects.dissolve('value', as_index=False)
        egid = float(re.sub('[^0-9]','', os.path.basename(tile)))
        shape_egid  = rooftops[rooftops['EGID'] == egid]
        shape_egid.buffer(0)
        # shape_egid.geometry = shape_egid.geometry.buffer(1)

        misc.test_crs(shape_objects, shape_egid)

        logger.info(f"Filter detection by EGID location")
        selection = shape_objects.sjoin(shape_egid, how='inner', predicate="within")
        selection['area'] = selection.area 
        final_gdf = selection.drop(['index_right', 'OBJECTID', 'ALTI_MIN', 'ALTI_MAX', 'DATE_LEVE','SHAPE_AREA', 'SHAPE_LEN'], axis=1)
        feature_path = os.path.join(OUTPUT_DIR, f"tile_EGID_{int(egid)}_segment_selection.gpkg")
        final_gdf.to_file(feature_path)
        written_files.append(feature_path)  
        logger.info(f"...done. A file was written: {feature_path}")
        
        # Merge/Combine multiple shapefiles into one
        logger.info(f"Merge shapefiles together in a single vector layer")
        vector_layer = gpd.pd.concat([vector_layer, final_gdf])
    feature_path = os.path.join(OUTPUT_DIR, DETECTION + '.' + SHP_EXT)

    vector_layer.to_file(feature_path)
    written_files.append(feature_path)  
    logger.info(f"...done. A file was written: {feature_path}")