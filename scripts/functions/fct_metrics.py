
import os, sys
import pandas as pd
import geopandas as gpd
from loguru import logger

    

def intersection_over_union(polygon1_shape, polygon2_shape):
    """Determine the intersection area over union area (IOU) of two polygons

    Args:
        polygon1_shape (geometry): first polygon
        polygon2_shape (geometry): second polygon

    Returns:
        int: Unrounded ratio between the intersection and union area
    """

    # Calculate intersection and union, and the IOU
    polygon_intersection = polygon1_shape.intersection(polygon2_shape).area
    polygon_union = polygon1_shape.area + polygon2_shape.area - polygon_intersection
    return polygon_intersection / polygon_union


def apply_iou_threshold_one_to_one(tp_gdf_ini, threshold=0):
    """Apply the IoU threshold on the TP detection to only keep the ones with sufficient intersection over union.
    Each detection can correspond to several labels.

    Args:
        tp_gdf_ini (geodataframe): geodataframe of the potiential true positive detection
        threshold (int, optional): threshold to apply on the IoU. Defaults to 0.

    Returns:
        geodataframes: geodataframes of the true positive and of the flase positives intersecting labels.
    """

    # Filter detection based on IOU value
    # Keep only max IOU value for each detection
    tp_grouped_gdf = tp_gdf_ini.groupby(['ID_DET'], group_keys=False).apply(lambda g:g[g.IOU==g.IOU.max()])
    
    # Detection with IOU lower than threshold value are considered as FP and removed from TP list   
    fp_gdf_temp = tp_grouped_gdf[tp_grouped_gdf['IOU'] < threshold]
    id_det_fp = fp_gdf_temp['ID_DET'].unique().tolist()
    tp_gdf_temp = tp_grouped_gdf[~tp_grouped_gdf['ID_DET'].isin(id_det_fp)]

    # For each label, only keep the pred with the best IOU.
    sorted_tp_gdf_temp = tp_gdf_temp.sort_values(by='IOU')
    tp_gdf=sorted_tp_gdf_temp.drop_duplicates(['ID_GT'], keep='last', ignore_index=True)

    # Save the dropped preds due to multiple detections of the same label.
    id_det_not_dropped = tp_gdf.ID_DET.unique()
    fp_gdf_temp = pd.concat([fp_gdf_temp, sorted_tp_gdf_temp[~sorted_tp_gdf_temp.ID_DET.isin(id_det_not_dropped)]], ignore_index=True)

    return tp_gdf, fp_gdf_temp


def apply_iou_threshold_one_to_many(tp_gdf_ini, threshold=0):
    """Apply the IoU threshold on the TP detection to only keep the ones with sufficient intersection over union.
    Each detection can only correspond to one label.

    Args:
        tp_gdf_ini (geodataframe): geodataframe of the potiential true positive detection
        threshold (int, optional): threshold to apply on the IoU. Defaults to 0.

    Returns:
        geodataframes: geodataframes of the true positive and of the flase positives intersecting labels.
    """
    
    # Compare the global IOU of the detection based on all the matching labels
    sum_detections_gdf=tp_gdf_ini.groupby(['ID_DET'])['IOU'].sum().reset_index()
    true_detections_gdf=sum_detections_gdf[sum_detections_gdf['IOU']>threshold]
    true_detections_index=true_detections_gdf['ID_DET'].unique().tolist()

    # Check that the label is at least 25% under the prediction.
    tp_gdf_ini['label_in_pred']=round(tp_gdf_ini['geom_GT'].intersection(tp_gdf_ini['geom_DET']).area/tp_gdf_ini['geom_GT'].area, 3)
    tp_gdf_temp=tp_gdf_ini[(tp_gdf_ini['ID_DET'].isin(true_detections_index)) & (tp_gdf_ini['label_in_pred'] > 0.25)]

    # For each label, only keep the pred with the best IOU.
    sorted_tp_gdf_temp = tp_gdf_temp.sort_values(by='IOU')
    tp_gdf=sorted_tp_gdf_temp.drop_duplicates(['ID_GT'], keep='last', ignore_index=True)
    id_det_tp=tp_gdf['ID_DET'].unique().tolist()

    fp_gdf_temp=tp_gdf_ini[~tp_gdf_ini['ID_DET'].isin(id_det_tp)]
    fp_gdf_temp = fp_gdf_temp.groupby(['ID_DET'], group_keys=False).apply(lambda g:g[g.IOU == g.IOU.max()])

    return tp_gdf, fp_gdf_temp


def get_fractional_sets(preds_gdf, labels_gdf, method='one-to-one'):
    """Separate the predictions and labels between TP, FP and FN based on their overlap and the passed IoU score.
    One prediction can either correspond to one (one-to-one) or several (one-to-many) labels.

    Args:
        preds_gdf (geodataframe): geodataframe of the prediction with the id "ID_DET"
        labels_gdf (geodataframe): threshold to apply on the IoU to determine TP and FP
        method (str, optional): string with the possible values 'one-to-one' or 'one-to-many' indicating if a prediction can or not correspond to several labels. 
                Defaults to 'one-to-one'.

    Raises:
        Exception: CRS mismatch

    Returns:
        geodataframes: geodataframes of the true positives, false postivies and false negatives
    """

    if len(labels_gdf) == 0:
        fp_gdf = preds_gdf.copy()
        tp_gdf = gpd.GeoDataFrame()
        fn_gdf = gpd.GeoDataFrame()       
        return tp_gdf, fp_gdf, fn_gdf
    
    try:
        assert(preds_gdf.crs == labels_gdf.crs), f"CRS mismatch: predictions' CRS = {preds_gdf.crs}, labels' CRS = {labels_gdf.crs}"
    except Exception as e:
        raise Exception(e)

    # CREATE ADDITIONAL COLUMN FOR TP, FP AND FN CLASSIFICATION AND IOU COMPUTATION
    labels_gdf['geom_GT'] = labels_gdf.geometry
    preds_gdf['geom_DET'] = preds_gdf.geometry

    # TRUE POSITIVES
    left_join = gpd.sjoin(preds_gdf, labels_gdf, how='left', predicate='intersects', lsuffix='left', rsuffix='right')
    tp_gdf_temp = left_join[left_join.ID_GT.notnull()].copy()

    # IOU computation between GT geometry and Detection geometry
    geom1 = tp_gdf_temp['geom_DET'].to_numpy().tolist()
    geom2 = tp_gdf_temp['geom_GT'].to_numpy().tolist()
    iou = []
    for (i, ii) in zip(geom1, geom2):
        iou.append(intersection_over_union(i, ii))
    tp_gdf_temp['IOU'] = iou

    iou_threshold=0.1
    if method=='one-to-many':
        tp_gdf, fp_gdf_temp = apply_iou_threshold_one_to_many(tp_gdf_temp, iou_threshold)
    else:
        tp_gdf, fp_gdf_temp = apply_iou_threshold_one_to_one(tp_gdf_temp, iou_threshold)


    # FALSE POSITIVES -> potentially object not referenced in ground truth or mistakes
    fp_gdf = left_join[left_join.ID_GT.isna()].copy()
    fp_gdf = pd.concat([fp_gdf, fp_gdf_temp])
    assert(len(fp_gdf[fp_gdf.duplicated()]) == 0)


    # FALSE NEGATIVES -> objects that have been missed by the detection algorithm
    right_join = gpd.sjoin(labels_gdf, preds_gdf, how='left', predicate='intersects', lsuffix='left', rsuffix='right')
    
    id_gt_tp=tp_gdf['ID_GT'].unique().tolist()
    suppressed_tp=tp_gdf_temp[~tp_gdf_temp['ID_GT'].isin(id_gt_tp)]
    id_gt_filter = suppressed_tp['ID_GT'].unique().tolist()
    
    fn_too_low_hit_gdf = right_join[right_join['ID_GT'].isin(id_gt_filter)]
    fn_no_hit_gdf = right_join[right_join.ID_DET.isna()].copy()
    fn_gdf = pd.concat([fn_no_hit_gdf, fn_too_low_hit_gdf])
   
    fn_gdf.drop_duplicates(subset=['ID_GT'], inplace=True)

    # Tag predictions   
    tp_gdf['tag'] = 'TP'
    fp_gdf['tag'] = 'FP'
    fn_gdf['tag'] = 'FN'

    return tp_gdf, fp_gdf, fn_gdf


def get_metrics(tp_gdf, fp_gdf, fn_gdf):
    """Determine the metrics based on the TP, FP and FN

    Args:
        tp_gdf (geodataframe): true positive detections
        fp_gdf (geodataframe): false positive detections
        fn_gdf (geodataframe): false negative labels

    Returns:
        float: precision, recall and f1 score
    """
    
    TP = len(tp_gdf)
    FP = len(fp_gdf)
    FN = len(fn_gdf)
    #print(TP, FP, FN)
    
    if TP == 0:
        return 0, 0, 0

    precision = TP / (TP + FP)
    recall    = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    
    return precision, recall, f1