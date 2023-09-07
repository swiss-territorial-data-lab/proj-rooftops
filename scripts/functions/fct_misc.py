import os
import sys
import rasterio
import pandas as pd
import geopandas as gpd
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
    import logger


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