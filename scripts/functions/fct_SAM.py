import os, sys
import re
import torch
from glob import glob
from loguru import logger
from tqdm import tqdm

import pandas as pd
import geopandas as gpd
from samgeo import SamGeo
sys.path.insert(1, 'scripts')
import functions.fct_com as fct_com


def SAM_mask(IMAGE_DIR, OUTPUT_DIR, SIZE, CROP, SHP_ROOF, DL_CKP, CKP_DIR, CKP, BATCH, FOREGROUND, UNIQUE, MASK_MULTI, SHP_EXT, dict_params, written_files):
    
    logger.info(f"Read images file name")
    tiles=glob(os.path.join(IMAGE_DIR, '*.tif'))

    if '\\' in tiles[0]:
        tiles=[tile.replace('\\', '/') for tile in tiles]
      
    for tile in tqdm(tiles, desc='Applying SAM to tiles', total=len(tiles)):

        logger.info(f"Read images: {os.path.basename(tile)}") 
        image = tile

        # Crop the input image by pixel value
        if CROP:
            logger.info(f"Crop image with size {SIZE}") 
            image = fct_com.crop(image, SIZE, IMAGE_DIR)
            written_files.append(image)  
            logger.info(f"...done. A file was written: {image}") 

        egid = float(re.sub('[^0-9]','', os.path.basename(tile)))
        roofs=gpd.read_file(SHP_ROOF)
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
        file_path=os.path.join(fct_com.ensure_dir_exists(os.path.join(OUTPUT_DIR, 'segmented_images')),
                                tile.split('/')[-1].split('.')[0] + '_segment.tif')       
        
        mask = file_path
        sam.generate(image, mask, batch=BATCH, foreground=FOREGROUND, unique=UNIQUE, erosion_kernel=(3,3), mask_multiplier=MASK_MULTI)
        written_files.append(file_path)  
        logger.info(f"...done. A file was written: {file_path}")

        file_path=os.path.join(fct_com.ensure_dir_exists(os.path.join(OUTPUT_DIR, 'segmented_images')),
                                tile.split('/')[-1].split('.')[0] + '_colormask.tif')   
        # sam.show_masks(cmap="binary_r")
        # sam.show_anns(axis="off", alpha=0.7, output=file_path)

        logger.info(f"Convert segmentation mask to vector layer")  
        if SHP_EXT == 'gpkg': 
            file_path=os.path.join(fct_com.ensure_dir_exists(os.path.join(OUTPUT_DIR, 'segmented_images')),
                    tile.split('/')[-1].split('.')[0] + '_segment.gpkg')       
            sam.tiff_to_gpkg(mask, file_path, simplify_tolerance=None)

            written_files.append(file_path)  
            logger.info(f"...done. A file was written: {file_path}")
        elif SHP_EXT == 'shp': 
            file_path=os.path.join(fct_com.ensure_dir_exists(os.path.join(OUTPUT_DIR, 'segmented_images')),
                    tile.split('/')[-1].split('.')[0] + '_segment.shp')        
            sam.tiff_to_vector(mask, file_path)
            written_files.append(file_path)  
            logger.info(f"...done. A file was written: {file_path}")


def filter(OUTPUT_DIR, SHP_ROOFS, SRS, DETECTION, SHP_EXT, written_files):

    # Get the rooftops shapes
    ROOFS_DIR, ROOFS_NAME = os.path.split(SHP_ROOFS)
    feature_path = os.path.join(ROOFS_DIR, ROOFS_NAME[:-4]  + "_EGID.shp")

    if os.path.exists(feature_path):
        logger.info(f"File {ROOFS_NAME[:-4]}_EGID.shp already exists")
        rooftops = gpd.read_file(feature_path)
    else:
        logger.info(f"File {ROOFS_NAME[:-4]}_EGID.shp does not exist")
        logger.info(f"Create it")
        # gdf_roofs = gpd.read_file(WORKING_DIR  + '/' + ROOFS_DIR  + '/' + ROOFS_NAME)
        gdf_roofs = gpd.read_file(ROOFS_DIR  + '/' + ROOFS_NAME)
        logger.info(f"Dissolved shapes by EGID number")
        rooftops = gdf_roofs.dissolve('EGID', as_index=False)
        rooftops.drop(['OBJECTID', 'ALTI_MAX', 'DATE_LEVE', 'SHAPE_AREA', 'SHAPE_LEN'], axis=1)
        rooftops.to_file(feature_path)
        written_files.append(feature_path)  
        logger.info(f"...done. A file was written: {feature_path}")

    # Read all the shapefile produced, filter them with rooftop extension and merge them into a single layer  
    logger.info(f"Read shapefiles' name")
    tiles=glob(os.path.join(OUTPUT_DIR + 'segmented_images',  '*.' + SHP_EXT))
    if '\\' in tiles[0]:
        tiles=[tile.replace('\\', '/') for tile in tiles]

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

        fct_com.test_crs(shape_objects, shape_egid)

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


def assessment(OUTPUT_DIR, DETECTION, GT, SHP_EXT, written_files):

    # Open shapefiles
    gdf_gt = gpd.read_file(GT)
    gdf_gt = gdf_gt[gdf_gt['occupation'] == 1]
    gdf_gt['ID_GT'] = gdf_gt.index
    gdf_gt = gdf_gt.rename(columns={"area": "area_GT"})
    logger.info(f"Read GT file: {len(gdf_gt)} shapes")

    feature_path = os.path.join(OUTPUT_DIR, DETECTION + '.' + SHP_EXT)
    gdf_detec = gpd.read_file(feature_path)
    gdf_detec = gdf_detec# [gdf_detec['occupation'] == 1]
    gdf_detec['ID_DET'] = gdf_detec.index
    gdf_detec = gdf_detec.rename(columns={"area": "area_DET"})
    logger.info(f"Read detection file: {len(gdf_detec)} shapes")


    logger.info(f"Metrics computation:")
    logger.info(f" - Compute TP, FP and FN")

    tp_gdf, fp_gdf, fn_gdf = fct_com.get_fractional_sets(gdf_detec, gdf_gt)

    # Tag predictions   
    tp_gdf['tag'] = 'TP'
    fp_gdf['tag'] = 'FP'
    fn_gdf['tag'] = 'FN'

    # Compute metrics
    precision, recall, f1 = fct_com.get_metrics(tp_gdf, fp_gdf, fn_gdf)

    TP = len(tp_gdf)
    FP = len(fp_gdf)
    FN = len(fn_gdf)

    logger.info(f"   TP = {TP}, FP = {FP}, FN = {FN}")
    logger.info(f"   precision = {precision:.2f}, recall = {recall:.2f}, f1 = {f1:.2f}")
    logger.info(f" - Compute mean Jaccard index")
    iou_average = tp_gdf['IOU'].mean()
    logger.info(f"   IOU average = {iou_average:.2f}")

    # Set the final dataframe with tagged prediction
    logger.info(f"Set the final dataframe")

    tagged_preds_gdf_dict = pd.concat([tp_gdf, fp_gdf, fn_gdf])
    tagged_preds_gdf_dict = tagged_preds_gdf_dict.drop(['index_right', 'path', 'layer', 'occupation', 'geom_GT', 'geom_DET'], axis = 1)
    tagged_preds_gdf_dict = tagged_preds_gdf_dict.rename(columns={'value': 'mask_value'})

    feature_path = os.path.join(OUTPUT_DIR, f'tagged_predictions.gpkg')
    tagged_preds_gdf_dict.to_file(feature_path, driver='GPKG', index=False)
    written_files.append(feature_path)

    return f1