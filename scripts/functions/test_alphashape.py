# Copied from https://alphashape.readthedocs.io/en/latest/_modules/alphashape/optimizealpha.html
# Changed line 45 to allow the iteration through MultiPoints.
# Added try and error around line 39 to avoid exiting on error during the optimization.

__all__ = ['optimizealpha']
import sys
import logging
import shapely
from shapely.geometry import MultiPoint
from shapely.errors import GEOSException
from shapely.validation import make_valid
import trimesh
from typing import Union, Tuple, List
import rtree  # Needed by trimesh
import numpy as np
try:
    import geopandas
    USE_GP = True
except ImportError:
    USE_GP = False


def _testalpha(points: Union[List[Tuple[float]], np.ndarray], alpha: float):
    """
    Evaluates an alpha parameter.

    This helper function creates an alpha shape with the given points and alpha
    parameter.  It then checks that the produced shape is a Polygon and that it
    intersects all the input points.

    Args:
        points: data points
        alpha: alpha value

    Returns:
        bool: True if the resulting alpha shape is a single polygon that
            intersects all the input data points.
    """
    from alphashape import alphashape
    try:
        polygon = alphashape(points, alpha)
    except shapely.errors.GEOSException as e:
        if 'side location conflict' in str(e):
            return False
        else:
            raise Exception(e)
    if isinstance(polygon, shapely.geometry.polygon.Polygon):
        if not isinstance(points, MultiPoint):
            points = MultiPoint(list(points))
        try:
            return all([polygon.intersects(point) for point in points.geoms])
        except GEOSException:
            logging.error('Alphashape producing an invalid polygon. Correction with the shapely function "make_valid".')
            polygon = make_valid(polygon)
            return all([polygon.intersects(point) for point in points.geoms])
    elif isinstance(polygon, trimesh.base.Trimesh):
        return len(polygon.faces) > 0 and all(
            trimesh.proximity.signed_distance(polygon, list(points)) >= 0)
    else:
        return False


def optimizealpha(points: Union[List[Tuple[float]], np.ndarray],
                  max_iterations: int = 10000, lower: float = 0.,
                  upper: float = sys.float_info.max, silent: bool = False):
    """
    Solve for the alpha parameter.

    Attempt to determine the alpha parameter that best wraps the given set of
    points in one polygon without dropping any points.

    Note:  If the solver fails to find a solution, a value of zero will be
    returned, which when used with the alphashape function will safely return a
    convex hull around the points.

    Args:

        points: an iterable container of points
        max_iterations (int): maximum number of iterations while finding the
            solution
        lower: lower limit for optimization
        upper: upper limit for optimization
        silent: silence warnings

    Returns:

        float: The optimized alpha parameter

    """
    # Convert to a shapely multipoint object if not one already
    if USE_GP and isinstance(points, geopandas.GeoDataFrame):
        points = points['geometry']

    # Set the bounds
    assert lower >= 0, "The lower bounds must be at least 0"
    # Ensure the upper limit bounds the solution
    assert upper <= sys.float_info.max, (
        f'The upper bounds must be less than or equal to {sys.float_info.max} '
        'on your system')

    if _testalpha(points, upper):
        if not silent:
            logging.error('the max float value does not bound the alpha '
                          'parameter solution')
        return 0.

    # Begin the bisection loop
    counter = 0
    while (upper - lower) > np.finfo(float).eps * 2:
        # Bisect the current bounds
        test_alpha = (upper + lower) * .5

        # Update the bounds to include the solution space
        if _testalpha(points, test_alpha):
            lower = test_alpha
        else:
            upper = test_alpha

        # Handle exceeding maximum allowed number of iterations
        counter += 1
        if counter > max_iterations:
            if not silent:
                logging.warning('maximum allowed iterations reached while '
                                'optimizing the alpha parameter')
            lower = 0.
            break
    return lower