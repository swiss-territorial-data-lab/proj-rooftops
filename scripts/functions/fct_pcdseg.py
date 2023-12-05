import os, sys
from loguru import logger

import numpy as np
import pandas as pd
import geopandas as gpd

import matplotlib.pyplot as plt

from scipy.spatial import ConvexHull, QhullError
from shapely.errors import GEOSException
from shapely.geometry import Polygon, MultiPolygon

import alphashape

sys.path.insert(1, 'scripts')
from functions.test_alphashape import optimizealpha


def vectorize_concave(df, plan_groups, epsg=2056, alpha_ini=None, visu=False):
    """Vectorize clustered points to a concave polygon

    Args:
        df (geodataframe): coordinates of the points
        plan_groups (list): name of the groups
        epsg (int, optional): EPSG number of the CRS. Defaults to 2056.
        alpha_ini (float, optional): alpha coefficient for the alpha shape algorithm. Defaults to None.
        visu (bool, optional): Produce visualisation of the results. Defaults to False.

    Returns:
        geodataframe: polygons produced with the clustered points.
    """

    try:
        object_type = df['type'].unique()[0]
    except IndexError as e:
        logger.error('No elements to vectorize. Returning an empty dataframe.')
        return gpd.GeoDataFrame()

    if len(df.type.unique())>1:
        logger.warning('Several different types were passed to the function "vectorize_concave".')

    # Intialize polygon dataframe 
    polygon_df = pd.DataFrame()
    # Iterrate over all the provided group of points
    for group in plan_groups:

        points_df = df[df['group'] == group]
        points_df = points_df.drop(['Unnamed: 0', 'Z', 'group', 'type'], axis=1) 
        points = points_df.to_numpy()

        # Produce alpha shapes point, i.e. bounding polygons containing a set of points.
        if not alpha_ini:
            # Tune alpha if undefined
            alpha = optimizealpha(points, upper=10, max_iterations=1000, silent=True)
            alpha_shape = alphashape.alphashape(points, alpha=alpha)
        else:
            alpha = alpha_ini
            optimized_alpha = False
            try:
                alpha_shape = alphashape.alphashape(points, alpha=alpha)
            except GEOSException:
                try:
                    # Try a second time with alpha=1 before doing the time-costly optimization of alpha.
                    alpha_shape = alphashape.alphashape(points, alpha=1)
                except GEOSException:
                    alpha = optimizealpha(points, upper=10, max_iterations=1000, silent=True)
                    alpha_shape = alphashape.alphashape(points, alpha=alpha)
                    optimized_alpha = True
            except QhullError:
                # Ignore Qhull error if the polygon is negligeable.
                if (points_df.X.max() - points_df.X.min() < 1) and (points_df.Y.max() - points_df.Y.min() < 1):
                    continue
        
            # If the result is empty and alpha was not optimized, optimize alpha to obtain a better result.
            if alpha_shape.is_empty and not optimized_alpha:
                alpha = optimizealpha(points, upper=10, max_iterations=1000, silent=True)
                alpha_shape = alphashape.alphashape(points, alpha=alpha)
                optimized_alpha = True

        if optimized_alpha:
            logger.info(f'The alpha value had to be optimized. The final value is {round(alpha, 3)}.')

        # The bounding points produced can be vizualize for control
        if visu:
            fig, ax = plt.subplots()
            ax.scatter(*zip(*points))
            # ax.add_patch(PolygonPatch(alpha_shape, alpha = 0.2))  # Used to work... not working at the moment, need to be fixed
            plt.show()

        # Transform alpha shape to polygon geometry
        if alpha_shape.geom_type == 'Polygon':
            poly = Polygon(alpha_shape)
        elif alpha_shape.geom_type == 'MultiPolygon':
            poly = MultiPolygon(alpha_shape)
        elif alpha_shape.geom_type in ['LineString', 'Point']:
            continue
        else:
            logger.critical(f'The created polygon has not a managed geometry type: {alpha_shape.geom_type}')
            sys.exit(1)

        # Build the final dataframe
        area = poly.area
        pcd_df = pd.DataFrame({'class': [object_type], 'area': [area], 'geometry': [poly]})

        polygon_df = pd.concat([polygon_df, pcd_df], ignore_index=True)
    
    if polygon_df.empty:
        polygon_gdf = gpd.GeoDataFrame()
        logger.warning('Vectorization retruned an empty dataframe.')
    else:
        polygon_gdf = gpd.GeoDataFrame(polygon_df, crs='EPSG:{}'.format(epsg), geometry='geometry')

    return polygon_gdf


def vectorize_convex(df, plan_groups, epsg=2056, visu=False):
    """Vectorize clustered points to a convex polygon

    Args:
        df (geodataframe): coordinates of the points
        plan_groups (list): name of the groups
        epsg (int, optional): EPSG number of the CRS. Defaults to 2056.
        visu (bool, optional): Produce visualisation of the results. Defaults to False.

    Returns:
        geodataframe: polygons produced with the clustered points.
    """

    object_type=df['type'].unique()[0]
    if len(df.type.unique()) > 1:
        logger.warning('Several different types were passsed to the function "vectorize_concave".')

    logger.info(f"Compute 2D vector from points groups of type {object_type}:")

    # Intialize polygon dataframe 
    polygon_df = pd.DataFrame()
    idx = []

    # Iterrate over all the provided group of points
    for group in plan_groups:

        # 
        points = df[df['group'] == group]
        points = points.drop(['Unnamed: 0', 'Z', 'group', 'type'], axis=1) 
        points = points.to_numpy()

        hull = ConvexHull(points)
        # area = hull.volume

        if visu:
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 3))
            for ax in (ax1, ax2):
                ax.plot(points[:, 0], points[:, 1], '.', color='k')
                if ax == ax1:
                    ax.set_title('Given points')
                else:
                    ax.set_title('Convex hull')
                    for simplex in hull.simplices:
                        ax.plot(points[simplex, 0], points[simplex, 1], 'c')
                    ax.plot(points[hull.vertices, 0], points[hull.vertices, 1], 'o', mec='r', color='none', lw=1, markersize=10)
            plt.show()

        polylist = []
        for idx in hull.vertices: #Indices of points forming the vertices of the convex hull.
            polylist.append(points[idx])

        # Transform list to polygon geometry
        poly = Polygon(polylist)

        # Build the final dataframe
        area = poly.area
        logger.info(f"   - Group: {group}, number of vertices: {len(polylist)}, area: {(area):.2f} m2")
        pcd_df = pd.DataFrame({'class': [object_type], 'area': [area], 'geometry': [poly]})

        polygon_df = pd.concat([polygon_df, pcd_df], ignore_index=True)
    
    polygon_gdf = gpd.GeoDataFrame(polygon_df, crs='EPSG:{}'.format(epsg), geometry='geometry')

    return polygon_gdf