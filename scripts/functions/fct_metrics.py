import os
import sys
from loguru import logger

import geopandas as gpd
import pandas as pd
import networkx as nx
from collections import OrderedDict
from fractions import Fraction
   

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


def apply_iou_threshold_one_to_one(tp_gdf_ini, threshold=0):
    """Apply the IoU threshold on the TP detection to only keep the ones with sufficient intersection over union.
    Each detection can correspond to several labels.

    Args:
        tp_gdf_ini (geodataframe): geodataframe of the potiential true positive detection
        threshold (int, optional): threshold to apply on the IoU. Defaults to 0.

    Returns:
        geodataframes: geodataframes of the true positive and of the flase positives intersecting labels.
    """

    # Filter detection based on IoU value
    # Keep only max IoU value for each detection
    tp_grouped_gdf = tp_gdf_ini.groupby(['detection_id'], group_keys=False).apply(lambda g:g[g.IOU==g.IOU.max()])
    
    # Detection with IoU lower than threshold value are considered as FP and removed from TP list   
    fp_gdf_temp = tp_grouped_gdf[tp_grouped_gdf['IOU'] < threshold]
    detection_id_fp = fp_gdf_temp['detection_id'].unique().tolist()
    tp_gdf_temp = tp_grouped_gdf[~tp_grouped_gdf['detection_id'].isin(detection_id_fp)]

    # For each label, only keep the pred with the best IoU.
    sorted_tp_gdf_temp = tp_gdf_temp.sort_values(by='IOU')
    tp_gdf = sorted_tp_gdf_temp.drop_duplicates(['label_id'], keep='last', ignore_index=True)

    # Save the dropped preds due to multiple detections of the same label.
    detection_id_not_dropped = tp_gdf.detection_id.unique()
    fp_gdf_temp = pd.concat([fp_gdf_temp, sorted_tp_gdf_temp[~sorted_tp_gdf_temp.detection_id.isin(detection_id_not_dropped)]], ignore_index=True)

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
    
    # Compare the global IoU of the detection based on all the matching labels    
    sum_detections_gdf = tp_gdf_ini.groupby(['detection_id'])['IOU'].sum().reset_index()
    true_detections_gdf = sum_detections_gdf[sum_detections_gdf['IOU']>threshold]
    true_detections_index = true_detections_gdf['detection_id'].unique().tolist()

    # Check that the label is at least 25% under the prediction.
    tp_gdf_ini['label_in_pred'] = round(tp_gdf_ini['label_geometry'].intersection(tp_gdf_ini['detection_geometry']).area/tp_gdf_ini['label_geometry'].area, 3)
    tp_gdf_temp = tp_gdf_ini[(tp_gdf_ini['detection_id'].isin(true_detections_index)) & (tp_gdf_ini['label_in_pred'] > 0.25)]

    # For each label, only keep the pred with the best IoU.
    sorted_tp_gdf_temp = tp_gdf_temp.sort_values(by='IOU')
    tp_gdf = sorted_tp_gdf_temp.drop_duplicates(['label_id'], keep='last', ignore_index=True)
    detection_id_tp = tp_gdf['detection_id'].unique().tolist()

    fp_gdf_temp = tp_gdf_ini[~tp_gdf_ini['detection_id'].isin(detection_id_tp)]
    fp_gdf_temp = fp_gdf_temp.groupby(['detection_id'], group_keys=False).apply(lambda g:g[g.IOU == g.IOU.max()])

    return tp_gdf, fp_gdf_temp


def get_fractional_sets(detections_gdf, labels_gdf, method='one-to-one'):
    """Separate the predictions and labels between TP, FP and FN based on their overlap and the passed IoU score.
    One prediction can either correspond to one (one-to-one) or several (one-to-many) labels.

    Args:
        detections_gdf (geodataframe): geodataframe of the detection with the id "detection_idection"
        labels_gdf (geodataframe): threshold to apply on the IoU to determine TP and FP
        method (str, optional): string with the possible values 'one-to-one' or 'one-to-many' indicating if a prediction can or not correspond to several labels. 
                Defaults to 'one-to-one'.

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
    left_join = gpd.sjoin(detections_gdf, labels_gdf, how='left', predicate='intersects', lsuffix='left', rsuffix='right')
    tp_gdf_temp = left_join[left_join.label_geometry.notnull()].copy()

    # IoU computation between label geometry and detection geometry
    geom1 = tp_gdf_temp['label_geometry'].to_numpy().tolist()
    geom2 = tp_gdf_temp['detection_geometry'].to_numpy().tolist()
    iou = []
    for (i, ii) in zip(geom1, geom2):
        iou.append(intersection_over_union(i, ii))
    tp_gdf_temp['IOU'] = iou

    iou_threshold = 0.1
    if method == 'one-to-many':
        tp_gdf, fp_gdf_temp = apply_iou_threshold_one_to_many(tp_gdf_temp, iou_threshold)
    else:
        tp_gdf, fp_gdf_temp = apply_iou_threshold_one_to_one(tp_gdf_temp, iou_threshold)


    # FALSE POSITIVES -> potentially object not referenced in ground truth or mistakes
    fp_gdf = left_join[left_join.label_geometry.isna()].copy()
    fp_gdf = pd.concat([fp_gdf, fp_gdf_temp])
    assert(len(fp_gdf[fp_gdf.duplicated()]) == 0)


    # FALSE NEGATIVES -> objects that have been missed by the detection algorithm
    right_join = gpd.sjoin(labels_gdf, detections_gdf, how='left', predicate='intersects', lsuffix='left', rsuffix='right')
    
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


def get_jaccard_index(labels_gdf, detections_gdf, attribute):
    """Compute the IoU (Jaccard index) of all the detection by roof (EGID)

    Args:
        detections_gdf (geodataframe): geodataframe of the detection with "detection_geometry" and "EGID" columns
        labels_gdf (geodataframe): geodataframe of the ground truth with "detection_geometry" and "EGID" columns

    Returns:
        labels_egid_gdf: geodataframes of all the labels merged by roof
        detections_egid_gdf: geodataframes of all the detections merged by roof and with the IoU by roof
    """

    detections_egid_gdf = detections_gdf.dissolve(by=attribute, as_index=False) 
    labels_egid_gdf = labels_gdf.dissolve(by=attribute, as_index=False) 

    geom1 = detections_egid_gdf.geometry.values.tolist()
    geom2 = labels_egid_gdf.geometry.values.tolist()
    iou = []
    for (i, ii) in zip(geom1, geom2):
        iou.append(intersection_over_union(i, ii))
    detections_egid_gdf['IOU_' + attribute] = iou

    return labels_egid_gdf, detections_egid_gdf


def tag(gt, dets, gt_buffer, gt_prefix, dets_prefix, threshold):
    """Tag labels and detections with "charges". 
    This method reserves the label and detection numbers by not duplicating or omitting to count a label or detection.
    A fractionnal "charge" will be assigned to labels/detections belonging to an identified group
    cf https://tech.stdl.ch/PROJ-TREEDET/#24-post-processing-assessment-algorithm-and-metrics-computation for more information

    Args:
        gt (geodataframe): geodataframe of the detection with the id "detection_idection"
        dets (geodataframe): threshold to apply on the IoU to determine TP and FP
        gt_buffer (float): buffer (in meter) applied to gt shape to avoid shape sharing border to be assigned to the same group if no common detection overlap them
        gt_prefix (str): prefix used to identified labels groups 
        det_prefix (str): prefix used to identified detections groups 

    Returns:
        gt (gdf): geodataframes of the tagged labels with the associated group id, TP charge and FN charge
        dets (gdf): geodataframes of the tagged detections with the associated group id, TP charge and FN charge
    """

    ### --- helper functions --- ###
    def make_groups():
        """Identify groups based on pairing nodes with NetworkX. The Graph is a collection of nodes.
        Nodes are hashable objects (geohash (str)).

        Returns:
            groups (list): list of connected geohash groups
        """

        g = nx.Graph()
        for row in l_join[l_join.geohash_y.notnull()].itertuples():
            g.add_edge(row.geohash_x, row.geohash_y)

        groups = list(nx.connected_components(g))

        return groups


    def assess_group(group):
        """Count the number of GT label and detection by identified group and provide FN and FP charge.

        Args:
            group (list): list of geohash (GT and detection) of a single group

        Returns:
            (dic): count of GT, detection by groups and associated FP and FN charges
        """
        
        # init
        cnt_gt = 0
        cnt_dets = 0
        FP_charge = 0
        FN_charge = 0
    
        for el in group:
            if el.startswith(dets_prefix):
                cnt_dets += 1
            if el.startswith(gt_prefix):
                cnt_gt += 1
 
        if cnt_dets > cnt_gt:
            FP_charge = cnt_dets - cnt_gt
        
        if cnt_dets < cnt_gt:
            FN_charge = cnt_gt - cnt_dets

        return dict(cnt_gt=cnt_gt, cnt_dets=cnt_dets, FP_charge=FP_charge, FN_charge=FN_charge)


    def assign_groups(row):
        """Assign a group number to GT and detection of a geodataframe

        Args:
            row (row): geodataframe row

        Returns:
            row (row): row with a new 'group_id' column
        """

        group_index = {node: i for i, group in enumerate(groups) for node in group}
    
        try:
            row['group_id'] = group_index[row['geohash']]
        except: 
            row['group_id'] = None
        
        return row


    def assign_charges(row):
        """Assign a charge to GT and detection of a geodataframe

        Args:
            row (row): geodataframe row

        Returns:
            row (row): row with new columns: GT = TP_charge and FN_charge and Detection = TP_charge and FP_charge
        """

        for k, v in charges_dict[row['geohash']].items():
            row[k] = v

        return row

    ### --- main --- ###
    assert 'geohash' in gt.columns.tolist()
    assert 'geohash' in dets.columns.tolist()

    # init
    _gt = gt.copy()
    _dets = dets.copy()
    _gt['geometry'] = _gt.geometry.buffer(gt_buffer, join_style=2)

    charges_dict = {}

    # spatial joins
    l_join = gpd.sjoin(_dets, _gt, how='left', predicate='intersects', lsuffix='x', rsuffix='y')
    r_join = gpd.sjoin(_dets, _gt, how='right', predicate='intersects', lsuffix='x', rsuffix='y')

    # trivial False Positives
    trivial_FPs = l_join[l_join.geohash_y.isna()]
    for tup in trivial_FPs.itertuples():
        charges_dict = {
            **charges_dict,
            tup.geohash_x: {
                'FP_charge': Fraction(1, 1),
                'TP_charge': Fraction(0, 1)
            }
        }

    # trivial False Negatives
    trivial_FNs = r_join[r_join.geohash_x.isna()]
    for tup in trivial_FNs.itertuples():
        charges_dict = {
            **charges_dict,
            tup.geohash_y: {
                'FN_charge': Fraction(1, 1),
                'TP_charge': Fraction(0, 1)
            }
        }

    # less trivial cases
    groups = make_groups()

    for group in groups:
        geom1 = gt[gt['geohash'].isin(group)].geometry.values.tolist()
        geohash1 = gt[gt['geohash'].isin(group)].geohash.values.tolist()        
        geom2 = dets[dets['geohash'].isin(group)].geometry.values.tolist()
        geohash2 = dets[dets['geohash'].isin(group)].geohash.values.tolist()

        # Filter detection based on intersection/overlap fraction threshold with the GT 
        for (i, ii, iii, iv) in zip(geom1, geom2, geohash1, geohash2):
            # % of overlap of GT and detection shape 
            polygon1_shape = i
            polygon2_shape = ii
            intersection = polygon1_shape.intersection(polygon2_shape).area
            if intersection / polygon1_shape.area <= threshold and intersection / polygon2_shape.area <= threshold:
                group.remove(iv)
                charges_dict = {
                    **charges_dict,
                    iv: {
                    'FP_charge': Fraction(1, 1),
                    'TP_charge': Fraction(0, 1)
                    }
                }

        group_assessment = assess_group(group)
        this_group_charges_dict = {}

        for el in group:
            if el.startswith(dets_prefix):
                this_group_charges_dict[el] = {
                    'TP_charge': Fraction(min(group_assessment['cnt_gt'], group_assessment['cnt_dets']), group_assessment['cnt_dets']),
                    'FP_charge': Fraction(group_assessment['FP_charge'], group_assessment['cnt_dets'])      
                }
        
            if el.startswith(gt_prefix):
                this_group_charges_dict[el] = {
                    'TP_charge': Fraction(min(group_assessment['cnt_gt'], group_assessment['cnt_dets']), group_assessment['cnt_gt']),
                    'FN_charge': Fraction(group_assessment['FN_charge'], group_assessment['cnt_gt'])
                }
        
        charges_dict = {**charges_dict, **this_group_charges_dict}

    # remove the buffer applied before group assignement to recover original geometry 
    _gt['geometry'] = _gt.geometry.buffer(-gt_buffer, join_style=2)

    _gt = _gt.apply(lambda row: assign_groups(row), axis=1)
    _dets = _dets.apply(lambda row: assign_groups(row), axis=1)

    _gt = _gt.apply(lambda row: assign_charges(row), axis=1)
    _dets = _dets.apply(lambda row: assign_charges(row), axis=1)

    return _gt[gt.columns.to_list() + ['group_id', 'TP_charge', 'FN_charge']], _dets[dets.columns.to_list() + ['group_id', 'TP_charge', 'FP_charge']]


def get_count(tagged_gt, tagged_dets=pd.DataFrame({'TP_charge':[], 'FP_charge':[]})):
    """Sum the TP, FP and FN charge for all the labels and detections

    Args:
        tagged_gt (gdf): geodataframe with TP and FP charges
        tagged_dets (gdf): geodataframe with TP and FN charges

    Returns:
        TP, FP, FN (float): values of TP, FP and FN
    """

    assert 'TP_charge' in tagged_dets.columns.tolist()
    assert 'TP_charge' in tagged_gt.columns.tolist()
    assert 'FP_charge' in tagged_dets.columns.tolist()
    assert 'FN_charge' in tagged_gt.columns.tolist()
    
    TP = float(tagged_dets.TP_charge.sum())
    FP = float(tagged_dets.FP_charge.sum())

    _TP = float(tagged_gt.TP_charge.sum()) # x-check
    FN = float(tagged_gt.FN_charge.sum())
    
    try:
        assert _TP == TP, f"{_TP} != {TP}"
    except AssertionError:
        logger.error(f"{_TP} != {TP}")
        sys.exit()

    return TP, FP, FN


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

    metrics_dic = dict(TP=TP, FP=FP, FN=FN,
                        TPplusFN=TP+FN, TPplusFP=TP+FP,
                        precision=precision, recall=recall, f1=f1
                        )

    return metrics_dic