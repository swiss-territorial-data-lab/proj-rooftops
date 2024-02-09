import geopandas as gpd
import pandas as pd


def intersection_over_union(polygon1_shape, polygon2_shape):
    """Determine the intersection area over union area (IoU) of two polygons

    Args:
        polygon1_shape (geometry): first polygon
        polygon2_shape (geometry): second polygon

    Returns:
        int: Unrounded ratio between the intersection and union area
    """

    # Calculate intersection and union, and the IoU
    polygon_intersection = polygon1_shape.intersection(polygon2_shape).area
    polygon_union = polygon1_shape.area + polygon2_shape.area - polygon_intersection

    return polygon_intersection / polygon_union


def apply_iou_threshold_one_to_one(tp_gdf_ini, threshold=0.1):
    """Apply the IoU threshold on the TP detection to only keep the ones with sufficient intersection over union.
    Each detection can correspond to several labels.

    Args:
        tp_gdf_ini (geodataframe): geodataframe of the potiential true positive detection
        threshold (int, optional): threshold to apply on the IoU. Defaults to 0.1.

    Returns:
        geodataframes: geodataframes of the true positive and of the flase positives intersecting labels.
    """

    # Filter detection based on IoU value
    # Keep only max IoU value for each detection
    tp_grouped_gdf = tp_gdf_ini.groupby(['detection_id'], group_keys=False).apply(lambda g:g[g.IoU==g.IoU.max()])
    
    # Detection with IoU lower than threshold value are considered as FP and removed from TP list   
    fp_gdf_temp = tp_grouped_gdf[tp_grouped_gdf['IoU'] < threshold]
    detection_id_fp = fp_gdf_temp['detection_id'].unique().tolist()
    tp_gdf_temp = tp_grouped_gdf[~tp_grouped_gdf['detection_id'].isin(detection_id_fp)]

    # For each label, only keep the pred with the best IoU.
    sorted_tp_gdf_temp = tp_gdf_temp.sort_values(by='IoU')
    tp_gdf = sorted_tp_gdf_temp.drop_duplicates(['label_id'], keep='last', ignore_index=True)

    # Save the dropped preds due to multiple detections of the same label.
    detection_id_not_dropped = tp_gdf.detection_id.unique()
    fp_gdf_temp = pd.concat([fp_gdf_temp, sorted_tp_gdf_temp[~sorted_tp_gdf_temp.detection_id.isin(detection_id_not_dropped)]], ignore_index=True)

    return tp_gdf, fp_gdf_temp


def apply_iou_threshold_one_to_many(tp_gdf_ini, threshold=0.1):
    """Apply the IoU threshold on the TP detection to only keep the ones with sufficient intersection over union.
    Each detection can only correspond to one label.

    Args:
        tp_gdf_ini (geodataframe): geodataframe of the potiential true positive detection
        threshold (int, optional): threshold to apply on the IoU. Defaults to 0.1.

    Returns:
        geodataframes: geodataframes of the true positive and of the flase positives intersecting labels.
    """
    
    # Compare the global IoU of the detection based on all the matching labels
    sum_detections_gdf = tp_gdf_ini.groupby(['detection_id'])['IoU'].sum().reset_index()
    true_detections_gdf = sum_detections_gdf[sum_detections_gdf['IoU'] > threshold]
    true_detections_index = true_detections_gdf['detection_id'].unique().tolist()

    # Check that the label is at least 25% under the prediction.
    tp_gdf_ini['label_in_pred'] = round(tp_gdf_ini['label_geometry'].intersection(tp_gdf_ini['detection_geometry']).area/tp_gdf_ini['label_geometry'].area, 3)
    tp_gdf_temp = tp_gdf_ini[(tp_gdf_ini['detection_id'].isin(true_detections_index)) & (tp_gdf_ini['label_in_pred'] > 0.25)]

    # For each label, only keep the pred with the best IoU.
    sorted_tp_gdf_temp = tp_gdf_temp.sort_values(by='IoU')
    tp_gdf = sorted_tp_gdf_temp.drop_duplicates(['label_id'], keep='last', ignore_index=True)
    detection_id_tp = tp_gdf['detection_id'].unique().tolist()

    fp_gdf_temp = tp_gdf_ini[~tp_gdf_ini['detection_id'].isin(detection_id_tp)]
    fp_gdf_temp = fp_gdf_temp.groupby(['detection_id'], group_keys=False).apply(lambda g:g[g.IoU == g.IoU.max()])

    return tp_gdf, fp_gdf_temp


def get_fractional_sets(detections_gdf, labels_gdf, method='one-to-one', iou_threshold=0.1):
    """Separate the predictions and labels between TP, FP and FN based on their overlap and the passed IoU score.
    One prediction can either correspond to one (one-to-one) or several (one-to-many) labels.

    Args:
        detections_gdf (geodataframe): geodataframe of the detections with the id "detection_id"
        labels_gdf (geodataframe): geodataframe of the labels with the id "label_id"
        method (str, optional): string with the possible values 'one-to-one' or 'one-to-many' indicating if a prediction can or not correspond to several labels. 
                Defaults to 'one-to-one'.
        iou_thrshold (float, optional): threshold to apply on the IoU to determine the tags. Defaults to 0.1.

    Raises:
        Exception: CRS mismatch

    Returns:
        geodataframes: geodataframes of the true positives, false postivies and false negatives
    """

    if len(labels_gdf) == 0:
        fp_gdf = detections_gdf.copy()
        tp_gdf = gpd.GeoDataFrame()
        fn_gdf = gpd.GeoDataFrame()       
        return tp_gdf, fp_gdf, fn_gdf
    
    try:
        assert(detections_gdf.crs == labels_gdf.crs), f"CRS mismatch: predictions' CRS = {detections_gdf.crs}, labels' CRS = {labels_gdf.crs}"
    except Exception as e:
        raise Exception(e)

    # CREATE ADDITIONAL COLUMN FOR TP, FP AND FN CLASSIFICATION AND IoU COMPUTATION
    labels_gdf['label_geometry'] = labels_gdf.geometry
    detections_gdf['detection_geometry'] = detections_gdf.geometry

    # TRUE POSITIVES
    left_join = gpd.sjoin(detections_gdf, labels_gdf, how='left', predicate='intersects', lsuffix='_det', rsuffix='_label')
    tp_gdf_temp = left_join[left_join.label_id.notnull()].copy()

    # IoU computation between label geometry and detection geometry
    geom1 = tp_gdf_temp['detection_geometry'].to_numpy().tolist()
    geom2 = tp_gdf_temp['label_geometry'].to_numpy().tolist()
    iou = []
    for (i, ii) in zip(geom1, geom2):
        iou.append(intersection_over_union(i, ii))
    tp_gdf_temp['IoU'] = iou

    if method == 'one-to-many':
        tp_gdf, fp_gdf_temp = apply_iou_threshold_one_to_many(tp_gdf_temp, iou_threshold)
    else:
        tp_gdf, fp_gdf_temp = apply_iou_threshold_one_to_one(tp_gdf_temp, iou_threshold)


    # FALSE POSITIVES -> potentially object not referenced in ground truth or mistakes
    fp_gdf = left_join[left_join.label_id.isna()].copy()
    fp_gdf = pd.concat([fp_gdf, fp_gdf_temp])
    assert(len(fp_gdf[fp_gdf.duplicated()]) == 0)

    # FALSE NEGATIVES -> objects that have been missed by the detection algorithm
    right_join = gpd.sjoin(labels_gdf, detections_gdf, how='left', predicate='intersects', lsuffix='_label', rsuffix='_det')
    
    id_label_tp = tp_gdf['label_id'].unique().tolist()
    suppressed_tp = tp_gdf_temp[~tp_gdf_temp['label_id'].isin(id_label_tp)]
    id_label_filter = suppressed_tp['label_id'].unique().tolist()
    
    fn_too_low_hit_gdf = right_join[right_join['label_id'].isin(id_label_filter)]
    fn_no_hit_gdf = right_join[right_join.detection_id.isna()].copy()
    fn_gdf = pd.concat([fn_no_hit_gdf, fn_too_low_hit_gdf])
   
    fn_gdf.drop_duplicates(subset=['label_id'], inplace=True)

    # Tag predictions   
    tp_gdf['tag'] = 'TP'
    fp_gdf['tag'] = 'FP'
    fn_gdf['tag'] = 'FN'

    return tp_gdf, fp_gdf, fn_gdf


def get_metrics(TP, FP, FN):
    """Determine the metrics precision, recall and f1-score based on the TP, FP and FN

    Args:
        TP (float): number of true positive detections
        FP (float): number of false positive detections
        FN (float): number of false negative labels

    Returns:
        metrics_dic (dic): dictionary returning count + metrics
    """
        
    precision = 0. if TP == 0.0 else TP / (TP + FP)
    recall = 0. if TP == 0.0 else TP / (TP + FN)
    f1 = 0. if precision == 0.0 or recall == 0.0 else 2 * precision * recall / (precision + recall)

    metrics_dict = dict(TP=TP, FP=FP, FN=FN,
                        precision=precision, recall=recall, f1=f1
                        )

    return metrics_dict