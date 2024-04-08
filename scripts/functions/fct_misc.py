import os
import sys
from loguru import logger

import geopandas as gpd
import numpy as np
import pandas as pd
import pygeohash as pgh
import rasterio

from functools import reduce
from rasterio.windows import Window
from shapely.affinity import scale
from shapely.geometry import Polygon
from shapely.validation import make_valid


def format_logger(logger):
    """Format the logger from loguru

    Args:
        logger: logger object from loguru

    Returns:
        logger: formatted logger object
    """

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


def add_geohash(gdf, prefix=None, suffix=None):
    """Add geohash column to a geodaframe.

    Args:
        gdf: geodaframe
        prefix (string): custom geohash string with a chosen prefix 
        suffix (string): custom geohash string with a chosen suffix

    Returns:
        out (gdf): geodataframe with geohash column
    """

    out_gdf = gdf.copy()
    out_gdf['geohash'] = gdf.to_crs(epsg=4326).apply(geohash, axis=1)

    if prefix is not None:
        out_gdf['geohash'] = prefix + out_gdf['geohash'].astype(str)

    if suffix is not None:
        out_gdf['geohash'] = out_gdf['geohash'].astype(str) + suffix

    return out_gdf


def bbox(bounds):
    """Get bounding box of a 2D shape

    Args:
        bounds ():

    Returns:
        geometry (Polygon): polygon geometry of the bounding box
    """

    minx = bounds[0]
    miny = bounds[1]
    maxx = bounds[2]
    maxy = bounds[3]

    return Polygon([[minx, miny],
                    [maxx,miny],
                    [maxx,maxy],
                    [minx, maxy]])


def check_validity(poly_gdf, correct=False):
    '''
    Test if all the geometry of a dataset are valid. When it is not the case, correct the geometries with a buffer of 0 m
    if correct != False and stop with an error otherwise.

    - poly_gdf: dataframe of geometries to check
    - correct: boolean indicating if the invalid geometries should be corrected with a buffer of 0 m

    return: a dataframe with valid geometries.
    '''

    invalid_condition = ~poly_gdf.is_valid

    try:
        assert(poly_gdf[invalid_condition].shape[0] == 0), \
            f"{poly_gdf[invalid_condition].shape[0]} geometries are invalid on" + \
                    f" {poly_gdf.shape[0]} polygons."
    except Exception as e:
        print(e)
        if correct:
            print("Correction of the invalid geometries with the shapely function 'make_valid'...")
            invalid_poly = poly_gdf.loc[invalid_condition, 'geometry']
            poly_gdf.loc[invalid_condition, 'geometry'] = [
                make_valid(poly) for poly in invalid_poly
                ]
        else:
            sys.exit(1)

    return poly_gdf


def crop(source, size, output):

    with rasterio.open(source) as src:

        # The size in pixels of your desired window
        x1, x2, y1, y2 = size[0], size[1], size[2], size[3]

        # Create a Window and calculate the transform from the source dataset    
        window = Window(x1, y1, x2, y2)
        transform = src.window_transform(window)

        # Create a new cropped raster to write to
        profile = src.profile
        profile.update({
            'height': x2 - x1,
            'width': y2 - y1,
            'transform': transform})

        file_path = os.path.join(ensure_dir_exists(os.path.join(output, 'crop')),
                                 source.split('/')[-1].split('.')[0] + '_crop.tif')   

        with rasterio.open(file_path, 'w', **profile) as dst:
            # Read the data from the window and write it to the output raster
            dst.write(src.read(window=window))  

        return file_path
    
    
def dissolve_by_attribute(desired_file, original_file, name, attribute, buffer=0.01):
    """Dissolve shape according to a given attribute in the gdf

    Args:
        desired_file (str): path to the processed geodataframe 
        original_file (str): path to the original geodataframe on which dissolution is perfomed
        name (str): root name of the file
        attribute (key): column key on which the operation is performed
        buffer (float): size of the buffer to remove thin space due to gaps between polygons

    Returns:
        gdf: geodataframes dissolved according to the provided gdf attribute
    """

    if os.path.exists(desired_file):
        logger.info(f"File {name}_{attribute}.shp already exists")
        gdf = gpd.read_file(desired_file)
        return gdf
    
    else:
        logger.info(f"File {name}_{attribute}.shp does not exist")
        logger.info(f"Create it")
        gdf = gpd.read_file(original_file)
        
        logger.info(f"Dissolved shapes by {attribute}")
        gdf.geometry = gdf.apply(lambda row: make_valid(row.geometry) if not row.geometry.is_valid else row.geometry, axis=1)
        gdf['geometry'] = gdf['geometry'].buffer(buffer, join_style=2) # apply a small buffer to prevent thin spaces due to polygons gaps        
        dissolved_gdf = gdf.dissolve(attribute, aggfunc={'ALTI_MIN': np.min}, as_index=False)
        dissolved_gdf['geometry'] = dissolved_gdf['geometry'].buffer(-buffer, join_style=2) # apply a small buffer to prevent thin spaces due to polygons gaps        

        gdf_considered_sections = gdf[gdf.area > 2].copy()
        attribute_count_gdf = gdf_considered_sections.EGID.value_counts() \
            .reset_index().rename(columns={'count': 'nbr_elem'})
        dissolved_gdf = dissolved_gdf.merge(attribute_count_gdf, on=attribute)
        dissolved_gdf = dissolved_gdf[~dissolved_gdf.nbr_elem.isna()].reset_index()

        dissolved_gdf.to_file(desired_file)
        logger.info(f"...done. A file was written: {desired_file}")

        return dissolved_gdf


def distance_shape(geom1, geom2):
    """Compute the minimum distance between two intersecting (or not) geometries (different or not).
    Geometries accepted: POLYGON, POINT, LINE but not MULTIPOLYGON!

    Args:
        geom1 (list): list of shapes of n dimensions
        geom2 (list): list of shapes of n dimensions

    Raises:
        Error: list lenght mismatch

    Returns:
        nearest_distance (float): minimum distance between the provided two geometries
    """

    nearest_distance = []

    try:
        assert(len(geom1) == len(geom2))
    except AssertionError:
        logger.error(f"The lists of geometries have different lengths: geom1 = {len(geom1)}, geom2 = {len(geom2)}")
        sys.exit()
    
    for (i, ii) in zip(geom1, geom2):
        if i == None or ii == None:
            nearest_distance.append(None)
        else:
            nearest_distance.append(i.exterior.distance(ii))

    return nearest_distance


def drop_duplicates(gdf, subset=None):
    """Delete duplicate rows based on the values in a subset column.

    Args:
        gdf : geodataframe
        subset: columns to check for duplicates. Defaults to None.

    Returns:
        out_gdf (gdf): clean geodataframe
    """

    out_gdf = gdf.copy()
    out_gdf.drop_duplicates(subset=subset, inplace=True)

    return out_gdf


def ensure_dir_notempty(dirpath):
    """Test if a directory is empty. If it is exit the script. 

    Args:
        dirpath (str): directory path to test
    """

    if len(os.listdir(dirpath)) == 0:
        logger.error(f"{dirpath} is empty. No detection masks found") 
        logger.info("Nothing left to be done: exiting")
        sys.stderr.flush()
        sys.exit()
    else:
        pass
    

def ensure_file_exists(filepath):
    """Test if a file exists. If not, exit the script.   

    Args:
        filepath (str): file path to test
    """

    if not os.path.isfile(filepath):
        logger.error(f"{filepath} does not exist. No vector file to assess") 
        logger.info("Nothing left to be done: exiting")
        sys.stderr.flush()
        sys.exit()
    else:
        pass


def geohash(row):
    """Geohash encoding (https://en.wikipedia.org/wiki/Geohash) of a location (point).
    If geometry type is a point then (x, y) coordinates of the point are considered. 
    If geometry type is a polygon then (x, y) coordinates of the polygon centroid are considered. 
    Other geometries are not handled at the moment    

    Args:
        row: geodaframe row

    Raises:
        Error: geometry error

    Returns:
        out (str): geohash code for a given geometry
    """
    
    if row.geometry.geom_type == 'Point':
        out = pgh.encode(latitude=row.geometry.y, longitude=row.geometry.x, precision=16)
    elif row.geometry.geom_type == 'Polygon':
        out = pgh.encode(latitude=row.geometry.centroid.y, longitude=row.geometry.centroid.x, precision=16)
    else:
        logger.error(f"{row.geometry.geom_type} type is not handled (only Point or Polygon geometry type)")
        sys.exit()

    return out


def get_inputs_for_assessment(path_egids, path_roofs, labels, detections):
    """Read the different files and format the roofs, labels and detections for the assessment

    Args:
        path_egids (string): path to the list of egids in csv format
        path_roofs (string): path the geodata file for the roofs
        labels (string or GeoDataFrame): path to the labels or labels in the form of a GeoDataFrame
        detections (string or GeoDataFrame): path to the detections or labels in the form of a GeoDataFrame

    Returns:
        tuple: 
            - Dataframe: egid properties
            - Geodataframe: extend of the roofs
            - Geodataframe: labels
            - Geodataframe: detections
    """

    # Get the EGIDS of interest
    egids = pd.read_csv(path_egids)
    array_egids = egids.EGID.to_numpy()
    logger.info(f'    - {egids.shape[0]} selected EGIDs.')

    if ('EGID' in path_roofs) | ('egid' in path_roofs):
        roofs_gdf = gpd.read_file(path_roofs)
    else:
        # Get the rooftops shapes
        _, ROOFS_NAME = os.path.split(path_roofs)
        attribute = 'EGID'
        original_file_path = path_roofs
        desired_file_path = os.path.join(os.path.dirname(path_roofs), ROOFS_NAME[:-4] + "_" + attribute + ".shp")
        roofs_gdf = dissolve_by_attribute(desired_file_path, original_file_path, name=ROOFS_NAME[:-4], attribute=attribute)

    roofs_gdf['EGID'] = roofs_gdf['EGID'].astype(int)
    roofs_gdf = roofs_gdf[roofs_gdf.EGID.isin(array_egids)].copy()
    logger.info(f'    - {roofs_gdf.shape[0]} roofs')

    if isinstance(labels, str):
        labels_gdf = gpd.read_file(labels)
    elif isinstance(labels, gpd.GeoDataFrame):
        labels_gdf = labels.copy()
    else:
        labels_gdf = gpd.GeoDataFrame()

    if not labels_gdf.empty:
        labels_gdf = format_labels(labels_gdf, roofs_gdf, array_egids)


    # Read the shapefile for detections
    if isinstance(detections, str):
        detections_gdf = gpd.read_file(detections)
    elif isinstance(detections, gpd.GeoDataFrame):
        detections_gdf = detections.copy()
    else:
        logger.critical(f'Unrecognized variable type for the detections: {type(detections)}.')
        sys.exit(1)

    detections_gdf = format_detections(detections_gdf)

    return egids, roofs_gdf, labels_gdf, detections_gdf


def ensure_dir_exists(dirpath):
    """Test if a directory exists. If not, make it.  

    Args:
        dirpath (str): directory path to test

    Returns:
        dirpath (str): directory path that have been tested
    """

    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
        logger.info(f"The directory {dirpath} was created.")
    
    return dirpath


def clip_labels(labels_gdf, tiles_gdf, fact=0.9999):
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

def format_detections(detections_gdf):

    if 'occupation' in detections_gdf.columns:
        detections_gdf = detections_gdf.fillna(1)
        detections_gdf = detections_gdf[detections_gdf['occupation'].astype(int) == 1].copy()
    detections_gdf['EGID'] = detections_gdf.EGID.astype(int)
    if 'det_id' in detections_gdf.columns:
        detections_gdf['detection_id'] = detections_gdf.det_id.astype(int)
    else:
        detections_gdf['detection_id'] = detections_gdf.index
    detections_gdf = detections_gdf.explode(ignore_index=True)
    logger.info(f"    - {len(detections_gdf)} detections")

    return detections_gdf


def format_labels(labels_gdf, roofs_gdf, selected_egids_arr):

    if labels_gdf.EGID.dtype != 'int64':
        labels_gdf['EGID'] = [round(float(egid)) for egid in labels_gdf.EGID.to_numpy()]
    if 'type' in labels_gdf.columns:
        labels_gdf['type'] = labels_gdf['type'].astype(int)
        labels_gdf = labels_gdf.rename(columns={'type':'obj_class'})
        # Type 12 corresponds to free surfaces, other classes are objects
        labels_gdf.loc[labels_gdf['obj_class'] == 4, 'descr'] = 'Aero'
        labels_gdf = labels_gdf[(labels_gdf['obj_class'] != 12) & (labels_gdf.EGID.isin(selected_egids_arr))].copy()
    else:
        labels_gdf = labels_gdf[labels_gdf.EGID.isin(selected_egids_arr)].copy()
        
    # Clip labels to the corresponding roof
    for egid in selected_egids_arr:
        labels_egid_gdf = labels_gdf[labels_gdf.EGID == egid].copy()
        labels_egid_gdf = labels_egid_gdf.clip(roofs_gdf.loc[roofs_gdf.EGID == egid, 'geometry'].buffer(-0.01, join_style='mitre'), keep_geom_type=True)

        tmp_gdf = labels_gdf[labels_gdf.EGID!=egid].copy()
        labels_gdf = pd.concat([tmp_gdf, labels_egid_gdf], ignore_index=True)

    labels_gdf['label_id'] = labels_gdf.id
    labels_gdf['area'] = round(labels_gdf.area, 4)

    labels_gdf.drop(columns=['fid', 'type', 'layer', 'path'], inplace=True, errors='ignore')
    labels_gdf=labels_gdf.explode(ignore_index=True).reset_index(drop=True)

    nbr_labels=labels_gdf.shape[0]
    logger.info(f"    - {nbr_labels} labels")

    return labels_gdf


def nearest_distance(gdf1, gdf2, join_key, parameter, lsuffix, rsuffix):
    """Prepare the geometries of two gdf to be processed (compute nearest distance between two shapes)

    Args:
        gdf1: geodataframe 1
        gdf2: geodataframe 2
        join_key (str): attribute on which the join between the 2 gdf is done
        parameter: operation to be performed
        lsuffix (str)
        rsuffix (str)

    Return:
        gdf1: geodataframe 1 + new columns for the computed paramters
    """

    gdf_tmp = gdf1.join(gdf2[[join_key, 'geometry']].set_index(join_key), on=join_key, how='left', lsuffix=lsuffix, rsuffix=rsuffix, validate='m:1')

    geom1 = gdf_tmp['geometry' + rsuffix].to_numpy().tolist()
    if parameter == 'nearest_distance_centroid':
        geom2 = gdf_tmp['geometry' + lsuffix].centroid.to_numpy().tolist()
    elif parameter == 'nearest_distance_border':
        geom2 = gdf_tmp['geometry' + lsuffix].to_numpy().tolist()
    gdf1[parameter] = distance_shape(geom1, geom2)
    gdf1[parameter] = round(gdf1[parameter], 4)

    return gdf1


def roundness(gdf):
    """Compute the roundness [0,1] of a polygon https://en.wikipedia.org/wiki/Roundness#Roundness_error_definitions 

    Args:
        gdf: geodataframe

    Return:
        gdf: geodataframe + new column for the computed paramters
    """

    area = gdf.area
    perimeter = gdf.length
    gdf['roundness'] = (4 * np.pi * area) / perimeter**2.0

    return gdf

    
def test_crs(crs1, crs2="EPSG:2056"):
    """Compare coordinate reference system two geodataframes. If they are not the same, stop the script. 

    Args:
        crs1 (str): coordinate reference system of geodataframe 1
        crs2 (str): coordinate reference system of geodataframe 2 (by default "EPSG:2056")

    Raises:
        Error: crs mismatch
    """

    if isinstance(crs1, gpd.GeoDataFrame):
        crs1 = crs1.crs
    if isinstance(crs2, gpd.GeoDataFrame):
        crs2 = crs2.crs

    try:
        assert(crs1 == crs2)
    except AssertionError:
        logger.error(f"CRS mismatch between the two files ({crs1} vs {crs2}")
        sys.exit()


def fillit(row):
    """A function to fill holes below an area threshold in a polygon"""
    newgeom=None
    rings = [i for i in row["geometry"].interiors] # List all interior rings
    if len(rings)>0: # If there are any rings
        to_fill = [Polygon(ring) for ring in rings] # List the ones to fill
        if len(to_fill)>0: # If there are any to fill
            newgeom = reduce(lambda geom1, geom2: geom1.union(geom2),[row["geometry"]]+to_fill) # Union the original geometry with all holes
    if newgeom:
        return newgeom
    else:
        return row["geometry"]
    

logger = format_logger(logger)