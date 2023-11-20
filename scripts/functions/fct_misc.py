import os
import sys
from loguru import logger

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.affinity import scale


def format_logger(logger):
    '''Format the logger from loguru
    
    -logger: logger object from loguru
    return: formatted logger object
    '''

    logger.remove()
    logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}",
            level="INFO", filter=lambda record: record["level"].no < 25)
    logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} - <green>{level}</green> - {message}",
            level="SUCCESS", filter=lambda record: record["level"].no < 30)
    logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} - <yellow>{level}</yellow> - {message}",
            level="WARNING", filter=lambda record: record["level"].no < 40)
    logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} - <red>{level}</red> - <level>{message}</level>",
            level="ERROR")
    
    return logger


logger = format_logger(logger)


def ensure_dir_exists(dirpath):
    '''
    Test if a directory exists. If not, make it.
    return: the path to the verified directory.
    '''

    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
        print(f"The directory {dirpath} was created.")

    return dirpath


def clip_labels(labels_gdf, tiles_gdf, fact=0.99):
    '''
    Clip the labels to the tiles
    Copied from the misc functions of the object detector 
    cf. https://github.com/swiss-territorial-data-lab/object-detector/blob/master/helpers/misc.py

    - labels_gdf: geodataframe with the labels
    - tiles_gdf: geodataframe of the tiles
    - fact: factor to scale the tiles before clipping
    return: a geodataframe with the labels clipped to the tiles
    '''

    tiles_gdf['tile_geometry'] = tiles_gdf['geometry']
        
    assert(labels_gdf.crs.name == tiles_gdf.crs.name)
    
    labels_tiles_sjoined_gdf = gpd.sjoin(labels_gdf, tiles_gdf, how='inner', predicate='intersects')
    
    def clip_row(row, fact=fact):
        
        old_geo = row.geometry
        scaled_tile_geo = scale(row.tile_geometry, xfact=fact, yfact=fact)
        new_geo = old_geo.intersection(scaled_tile_geo)
        row['geometry'] = new_geo

        return row

    clipped_labels_gdf = labels_tiles_sjoined_gdf.apply(lambda row: clip_row(row, fact), axis=1)
    clipped_labels_gdf.crs = labels_gdf.crs

    clipped_labels_gdf.drop(columns=['tile_geometry', 'index_right'], inplace=True)
    clipped_labels_gdf.rename(columns={'id': 'tile_id'}, inplace=True)

    return clipped_labels_gdf


def get_tilepath_from_id(tile_id, im_list):
    '''
    Get the tilepath containing the tile id from a list and check that there is no more than one.

    - tile_id: string with (part of) the tile name
    - im_list: list of the image/tile pathes
    return: matching tilepath
    '''

    matching_path=[tilepath for tilepath in im_list if tile_id in tilepath]
    if len(matching_path)>1:
        logger.critical(f'There are multiple tiles corresponding to the id {tile_id}.')
        sys.exit(1)
    elif len(matching_path)==0:
        logger.warning(f'There is no tile corresponding to the id {tile_id}.')
        return None
    else:
        tilepath=matching_path[0]

    return tilepath


def test_crs(crs1, crs2="EPSG:2056"):
    '''
    Take the crs of two dataframes and compare them. If they are not the same, stop the script.
    '''
    if isinstance(crs1, gpd.GeoDataFrame):
        crs1 = crs1.crs
    if isinstance(crs2, gpd.GeoDataFrame):
        crs2 = crs2.crs

    try:
        assert(crs1 == crs2), f"CRS mismatch between the two files ({crs1} vs {crs2})."
    except Exception as e:
        print(e)
        sys.exit(1)


def test_valid_geom(poly_gdf, correct=False, gdf_obj_name=None):
    '''
    Test if all the geometry of a dataset are valid. When it is not the case, correct the geometries with a buffer of 0 m
    if correct != False and stop with an error otherwise.

    - poly_gdf: dataframe of geometries to check
    - correct: boolean indicating if the invalid geometries should be corrected with a buffer of 0 m
    - gdf_boj_name: name of the dataframe of the object in it to print with the error message

    return: a dataframe with only valid geometries.
    '''

    try:
        assert(poly_gdf[poly_gdf.is_valid==False].shape[0]==0), \
            f"{poly_gdf[poly_gdf.is_valid==False].shape[0]} geometries are invalid{f' among the {gdf_obj_name}' if gdf_obj_name else ''}."
    except Exception as e:
        logger.error(e)
        if correct:
            logger.warning("Correction of the invalid geometries with a buffer of 0 m...")
            corrected_poly = poly_gdf.copy()
            corrected_poly.loc[corrected_poly.is_valid==False,'geometry'] = \
                            corrected_poly[corrected_poly.is_valid==False]['geometry'].buffer(0)

            return corrected_poly
        else:
            sys.exit(1)

    logger.info(f"There aren't any invalid geometries{f' among the {gdf_obj_name}' if gdf_obj_name else ''}.")

    return poly_gdf
