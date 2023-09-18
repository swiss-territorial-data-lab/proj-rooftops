import os
import sys

import numpy as np

from rasterio.features import rasterize
from shapely.geometry import Polygon


def poly_from_utm(polygon, transform):
    """Apply the image transform to the polygons

    Args:
        polygon (shapely geometry): polygon to transform
        transform (rasterio transform): transform to be applied

    Returns:
        shapely geometry: the converted polygon
    """
    poly_pts = []
    
    for i in np.array(polygon.exterior.coords):
        
        # Convert polygons to the image CRS
        poly_pts.append(~transform * tuple(i))
        
    # Generate a polygon object
    new_poly = Polygon(poly_pts)

    return new_poly


def polygons_to_raster_mask(polygons, image_meta):
    """Make a mask with the polygons for an image of the size and transfrom given in the meta

    Args:
        polygons (MultiPolygon geometry): polygons to rasterize
        image_meta (dict): rasterio meta with the height, width and transform of the output tile.

    Returns:
        numpy array, dict: output mask and its meta
    """

    im_size = (image_meta['height'], image_meta['width'])

    reprojected_polygons = [poly_from_utm(geom, image_meta['transform']) for geom in polygons.geoms]
    mask = rasterize(shapes=reprojected_polygons, out_shape=im_size)

    mask_meta = image_meta.copy()
    mask_meta.update({'count': 1, 'dtype': 'uint8'})

    return mask, mask_meta