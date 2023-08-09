from loguru import logger

import numpy as np
import pandas as pd
import geopandas as gpd

import matplotlib.pyplot as plt

from descartes import PolygonPatch
from scipy.spatial import ConvexHull
from shapely.errors import GEOSException
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import alphashape


def vectorize_concave(df, plan_groups, epsg=2056, alpha_ini=None, visu = False):

    object_type=df['type'].unique()[0]
    if len(df.type.unique())>1:
        logger.warning('Several different types were passed to the function "vectorize_concave".')

    # Intialize polygon dataframe 
    polygon_df = pd.DataFrame()
    # Iterrate over all the provided group of points
    for group in plan_groups:

        points = df[df['group'] == group]
        points = points.drop(['Unnamed: 0', 'Z', 'group', 'type'], axis = 1) 
        points = points.to_numpy()

        # Produce alpha shapes point, i.e. bounding polygons containing a set of points. alpha parameter can be tuned
        if not alpha_ini:
            alpha = alphashape.optimizealpha(points)
            logger.info(f"   - alpha shape value = {alpha}")
        else:
            alpha=alpha_ini
        try:
            alpha_shape = alphashape.alphashape(points, alpha = alpha)
        except GEOSException:
            alpha_shape = alphashape.alphashape(points, alpha = 1)

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

        # Build the final dataframe
        area = poly.area
        pcd_df = pd.DataFrame({'class': [object_type], 'area': [area], 'geometry': [poly]})

        polygon_df = pd.concat([polygon_df, pcd_df], ignore_index=True)
    
    polygon_gdf = gpd.GeoDataFrame(polygon_df, crs='EPSG:{}'.format(epsg), geometry='geometry')

    return polygon_gdf


def vectorize_convex(df, plan_groups, epsg=2056, visu=False):

    object_type=df['type'].unique()[0]
    if len(df.type.unique())>1:
        logger.warning('Several different types were passsed to the function "vectorize_concave".')

    logger.info(f"Compute 2D vector from points groups of type {object_type}:")

    # Intialize polygon dataframe 
    polygon_df = pd.DataFrame()
    idx = []

    # Iterrate over all the provided group of points
    for group in plan_groups:

        # 
        points = df[df['group'] == group]
        points = points.drop(['Unnamed: 0', 'Z', 'group', 'type'], axis = 1) 
        points = points.to_numpy()

        hull = ConvexHull(points)
        # area = hull.volume

        if visu:
            fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize = (10, 3))
            for ax in (ax1, ax2):
                ax.plot(points[:, 0], points[:, 1], '.', color='k')
                if ax == ax1:
                    ax.set_title('Given points')
                else:
                    ax.set_title('Convex hull')
                    for simplex in hull.simplices:
                        ax.plot(points[simplex, 0], points[simplex, 1], 'c')
                    ax.plot(points[hull.vertices, 0], points[hull.vertices, 1], 'o', mec = 'r', color = 'none', lw = 1, markersize = 10)
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


def IOU(pol1_xy, pol2_xy):
    # Define each polygon
    polygon1_shape = pol1_xy
    polygon2_shape = pol2_xy

    # Calculate intersection and union, and tne IOU
    polygon_intersection = polygon1_shape.intersection(polygon2_shape).area
    polygon_union = polygon1_shape.area + polygon2_shape.area - polygon_intersection
    return polygon_intersection / polygon_union


def get_fractional_sets(the_preds_gdf, the_labels_gdf):

    preds_gdf = the_preds_gdf.copy()
    labels_gdf = the_labels_gdf.copy()
    
    if len(labels_gdf) == 0:
        fp_gdf = preds_gdf.copy()
        tp_gdf = gpd.GeoDataFrame()
        fn_gdf = gpd.GeoDataFrame()       
        return tp_gdf, fp_gdf, fn_gdf
    
    try:
        assert(preds_gdf.crs == labels_gdf.crs), f"CRS Mismatch: predictions' CRS = {preds_gdf.crs}, labels' CRS = {labels_gdf.crs}"
    except Exception as e:
        raise Exception(e)
        

    # we add a dummy column to the labels dataset, which should not exist in predictions too;
    # this allows us to distinguish matching from non-matching predictions
    labels_gdf['dummy_id'] = labels_gdf.index
    
    # TRUE POSITIVES
    left_join = gpd.sjoin(preds_gdf, labels_gdf, how='left', predicate='intersects', lsuffix='left', rsuffix='right')
    
    tp_gdf = left_join[left_join.dummy_id.notnull()].copy()
    tp_gdf.drop_duplicates(subset=['dummy_id'], inplace=True)
    tp_gdf.drop(columns=['dummy_id'], inplace=True)
    
    # FALSE POSITIVES -> potentially "new" swimming pools
    fp_gdf = left_join[left_join.dummy_id.isna()].copy()
    assert(len(fp_gdf[fp_gdf.duplicated()]) == 0)
    fp_gdf.drop(columns=['dummy_id'], inplace=True)
    
    # FALSE NEGATIVES -> potentially, objects that are not actual swimming pools!
    right_join = gpd.sjoin(preds_gdf, labels_gdf, how='right', predicate='intersects', lsuffix='left', rsuffix='right')
    fn_gdf = right_join[right_join.index_left.isna()].copy()
    fn_gdf.drop_duplicates(subset=['dummy_id'], inplace=True)
    
    return tp_gdf, fp_gdf, fn_gdf


def get_metrics(tp_gdf, fp_gdf, fn_gdf):
    
    TP = len(tp_gdf)
    FP = len(fp_gdf)
    FN = len(fn_gdf)
    # print(TP, FP, FN)
    
    if TP == 0:
        return 0, 0, 0

    precision = TP / (TP + FP)
    recall    = TP / (TP + FN)
    f1 = 2*precision*recall/(precision+recall)
    
    return precision, recall, f1