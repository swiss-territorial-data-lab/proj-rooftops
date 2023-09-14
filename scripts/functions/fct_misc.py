import os
import sys
import rasterio
import pandas as pd
import geopandas as gpd
from loguru import logger
from shapely.geometry import Polygon
from rasterio.windows import Window


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
        assert(crs1==crs2), f"CRS mismatch between the two files ({crs1} vs {crs2})."
    except Exception as e:
        print(e)
        sys.exit(1)


def ensure_dir_exists(dirpath):
    '''
    Test if a directory exists. If not, make it.
    '''

    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
        logger.info(f"The directory {dirpath} was created.")
    
    return dirpath


def bbox(bounds):

    minx = bounds[0]
    miny = bounds[1]
    maxx = bounds[2]
    maxy = bounds[3]

    return Polygon([[minx, miny],
                    [maxx,miny],
                    [maxx,maxy],
                    [minx, maxy]])


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


def dissolve_by_attribute(desired_file, original_file, written_files, name, attribute):
    """Dissolve shape according to a given attribute in the gdf

    Args:
        desired_file (path): path to the processed geodataframe 
        original_file (path): path to the original geodataframe on which dissolution is perfomed
        name (str): root name of the file
        attribute (key): column key on which the operation is performed

    Returns:
        gdf: geodataframes dissolved according to the provided gdf attribute
    """

    if os.path.exists(desired_file):
        logger.info(f"File {name}_{attribute}.shp already exists")
        gdf = gpd.read_file(desired_file)
    else:
        logger.info(f"File {name}_{attribute}.shp does not exist")
        logger.info(f"Create it")
        gdf = gpd.read_file(original_file)
        logger.info(f"Dissolved shapes by {attribute}")
        gdf = gdf.dissolve(attribute, as_index=False)
        gdf['geometry'] = gdf['geometry'].buffer(0.0001)       
        gdf.to_file(desired_file)
        written_files.append(desired_file)  
        logger.info(f"...done. A file was written: {desired_file}")

    return gdf, written_files


def distance_shape(geom1, geom2):
    """Compute the minimum distance between two intersecting (or not) geometries (different or not).
    Geometries accepted: POLYGON, POINT, LINE but not MULTIPOLYGON!

    Args:
        geom1 (list): list of shapes of n dimensions
        geom2 (list): list of shapes of n dimensions

    Returns:
        nearest_distance (float): minimum distance between the provided two geometries
    """

    nearest_distance = []
    
    for (i, ii) in zip(geom1, geom2):
        if i == None or ii == None:
            nearest_distance.append(None)
        else:
            nearest_distance.append(i.exterior.distance(ii))

    return nearest_distance