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


def ensure_dir_exists(dirpath):
    '''
    Test if a directory exists. If not, make it.
    return: the path to the verified directory.
    '''

    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
        print(f"The directory {dirpath} was created.")

    return dirpath


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
