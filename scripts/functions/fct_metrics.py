
import os
import sys
from loguru import logger

import geopandas as gpd
import numpy as np
import pandas as pd

import networkx as nx
from fractions import Fraction
from shapely import unary_union
from shapely.geometry import GeometryCollection
from shapely.validation import make_valid
    

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


def apply_iou_threshold_one_to_one(tp_gdf_ini, threshold=0.1):
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
    tp_gdf=sorted_tp_gdf_temp.drop_duplicates(['label_id'], keep='last', ignore_index=True)

    # Save the dropped preds due to multiple detections of the same label.
    id_det_not_dropped = tp_gdf.ID_DET.unique()
    fp_gdf_temp = pd.concat([fp_gdf_temp, sorted_tp_gdf_temp[~sorted_tp_gdf_temp.ID_DET.isin(id_det_not_dropped)]], ignore_index=True)

    return tp_gdf, fp_gdf_temp


def apply_iou_threshold_one_to_many(tp_gdf_ini, threshold=0.1):
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
    tp_gdf_ini['label_in_pred']=round(tp_gdf_ini['label_geometry'].intersection(tp_gdf_ini['detection_geometry']).area/tp_gdf_ini['label_geometry'].area, 3)
    tp_gdf_temp=tp_gdf_ini[(tp_gdf_ini['ID_DET'].isin(true_detections_index)) & (tp_gdf_ini['label_in_pred'] > 0.25)]

    # For each label, only keep the pred with the best IOU.
    sorted_tp_gdf_temp = tp_gdf_temp.sort_values(by='IOU')
    tp_gdf=sorted_tp_gdf_temp.drop_duplicates(['label_id'], keep='last', ignore_index=True)
    id_det_tp=tp_gdf['ID_DET'].unique().tolist()

    fp_gdf_temp=tp_gdf_ini[~tp_gdf_ini['ID_DET'].isin(id_det_tp)]
    fp_gdf_temp = fp_gdf_temp.groupby(['ID_DET'], group_keys=False).apply(lambda g:g[g.IOU == g.IOU.max()])

    return tp_gdf, fp_gdf_temp


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
    except AssertionError as e:
        logger.critical(f"Difference in the count of TP between the labels and the detections: {str(e)}")
        sys.exit()

    return TP, FP, FN


def get_fractional_sets(dets_gdf, labels_gdf, method='one-to-one', iou_threshold=0.1):
    """Separate the predictions and labels between TP, FP and FN based on their overlap and the passed IoU score.
    One prediction can either correspond to one (one-to-one) or several (one-to-many) labels.

    Args:
        dets_gdf (geodataframe): geodataframe of the detections with the id "ID_DET"
        labels_gdf (geodataframe): geodataframe of the labels with the id "label_id"
        method (str, optional): string with the possible values 'one-to-one' or 'one-to-many' indicating if a prediction can or not correspond to several labels. 
                Defaults to 'one-to-one'.
        iou_thrshold (flaot, optional): threshold to apply on the IoU to determine the tags. Defaults to 0.1.

    Raises:
        Exception: CRS mismatch

    Returns:
        geodataframes: geodataframes of the true positives, false postivies and false negatives
    """

    if len(labels_gdf) == 0:
        fp_gdf = dets_gdf.copy()
        tp_gdf = gpd.GeoDataFrame()
        fn_gdf = gpd.GeoDataFrame()       
        return tp_gdf, fp_gdf, fn_gdf
    
    try:
        assert(dets_gdf.crs == labels_gdf.crs), f"CRS mismatch: predictions' CRS = {dets_gdf.crs}, labels' CRS = {labels_gdf.crs}"
    except Exception as e:
        raise Exception(e)

    # CREATE ADDITIONAL COLUMN FOR TP, FP AND FN CLASSIFICATION AND IOU COMPUTATION
    labels_gdf['label_geometry'] = labels_gdf.geometry
    dets_gdf['detection_geometry'] = dets_gdf.geometry

    # TRUE POSITIVES
    left_join = gpd.sjoin(dets_gdf, labels_gdf, how='left', predicate='intersects', lsuffix='_det', rsuffix='_label')
    tp_gdf_temp = left_join[left_join.label_id.notnull()].copy()

    # IOU computation between GT geometry and Detection geometry
    geom1 = tp_gdf_temp['detection_geometry'].to_numpy().tolist()
    geom2 = tp_gdf_temp['label_geometry'].to_numpy().tolist()
    iou = []
    for (i, ii) in zip(geom1, geom2):
        iou.append(intersection_over_union(i, ii))
    tp_gdf_temp['IOU'] = iou

    if method=='one-to-many':
        tp_gdf, fp_gdf_temp = apply_iou_threshold_one_to_many(tp_gdf_temp, iou_threshold)
    else:
        tp_gdf, fp_gdf_temp = apply_iou_threshold_one_to_one(tp_gdf_temp, iou_threshold)


    # FALSE POSITIVES -> potentially object not referenced in ground truth or mistakes
    fp_gdf = left_join[left_join.label_id.isna()].copy()
    fp_gdf = pd.concat([fp_gdf, fp_gdf_temp])
    assert(len(fp_gdf[fp_gdf.duplicated()]) == 0)


    # FALSE NEGATIVES -> objects that have been missed by the detection algorithm
    right_join = gpd.sjoin(labels_gdf, dets_gdf, how='left', predicate='intersects', lsuffix='left', rsuffix='right')
    
    id_gt_tp=tp_gdf['label_id'].unique().tolist()
    suppressed_tp=tp_gdf_temp[~tp_gdf_temp['label_id'].isin(id_gt_tp)]
    id_gt_filter = suppressed_tp['label_id'].unique().tolist()
    
    fn_too_low_hit_gdf = right_join[right_join['label_id'].isin(id_gt_filter)]
    fn_no_hit_gdf = right_join[right_join.ID_DET.isna()].copy()
    fn_gdf = pd.concat([fn_no_hit_gdf, fn_too_low_hit_gdf])
   
    fn_gdf.drop_duplicates(subset=['label_id'], inplace=True)

    # Tag predictions   
    tp_gdf['tag'] = 'TP'
    fp_gdf['tag'] = 'FP'
    fn_gdf['tag'] = 'FN'

    return tp_gdf, fp_gdf, fn_gdf


def get_free_area(labels_gdf, detections_gdf, roofs_gdf, attribute='EGID'):
    """Compute the occupied and free surface area of all the labels and detection by roof (EGID)

    Args:
        detections_gdf (geodataframe): geodataframe of the detections
        labels_gdf (geodataframe): geodataframe of the ground truth
        roofs_gdf (geodataframe): geodataframe of the roofs
        attribute (string): attribute to dissolve by. Defaults to 'EGID'.

    Returns:
        labels_free_gdf: geodataframes of all the labels merged by roof with the occupied and free surface area by roof
        detections_free_gdf: geodataframes of all the detections merged by roof with the occupied and free surface area by roof
    """

    detections_by_attribute_gdf = detections_gdf.dissolve(by=attribute, as_index=False) 
    labels_by_attribute_gdf = labels_gdf.dissolve(by=attribute, as_index=False) 
    roofs_by_attribute_gdf = roofs_gdf.dissolve(by=attribute, as_index=False)
    roofs_by_attribute_gdf['roof_area'] = roofs_by_attribute_gdf.area

    # Add value to empty gdf
    if detections_by_attribute_gdf['geometry'].empty:
        keys_list = detections_by_attribute_gdf.to_dict()
        dic = dict.fromkeys(keys_list, 0)
        detections_by_attribute_gdf = pd.DataFrame.from_dict(dic, orient='index').T
        detections_by_attribute_gdf['occup_area'] = 0
        detections_by_attribute_gdf['EGID'] = labels_by_attribute_gdf['EGID']
    else:
        detections_by_attribute_gdf['occup_area'] = detections_by_attribute_gdf.area

    if labels_by_attribute_gdf['geometry'].empty:
        keys_list = labels_by_attribute_gdf.to_dict()
        dic = dict.fromkeys(keys_list, 0)
        labels_by_attribute_gdf = pd.DataFrame.from_dict(dic, orient='index').T
        labels_by_attribute_gdf['occup_area'] = 0
    else:
        labels_by_attribute_gdf['occup_area'] = labels_by_attribute_gdf.area


    detections_with_area_gdf=pd.merge(detections_by_attribute_gdf, roofs_by_attribute_gdf[['EGID', 'roof_area']], on='EGID')
    detections_with_area_gdf['free_area'] = detections_with_area_gdf.roof_area - detections_with_area_gdf.occup_area

    labels_with_area_gdf=pd.merge(labels_by_attribute_gdf, roofs_by_attribute_gdf[['EGID', 'roof_area']], on='EGID')
    labels_with_area_gdf['free_area'] = labels_with_area_gdf.roof_area - labels_with_area_gdf.occup_area

    return labels_with_area_gdf, detections_with_area_gdf


def get_jaccard_index(labels_gdf, detections_gdf, attribute='EGID'):
    """Compute the IoU (Jaccard index) of all the detection by roof (EGID)

    Args:
        detections_gdf (geodataframe): geodataframe of the detection with "detection_geometry" and "EGID" columns
        labels_gdf (geodataframe): geodataframe of the ground truth with "detection_geometry" and "EGID" columns
        attribute (string): attribute to dissolve by before calculating the jaccard_index. Defaults to "EGID"

    Returns:
        detections_by_attr_gdf: geodataframes of all the detections merged by roof and with the IoU by roof
    """

    labels_by_attr_gdf = labels_gdf.dissolve(by=attribute, as_index=False)
    detections_by_attr_gdf = detections_gdf.dissolve(by=attribute, as_index=False)
 
    geom1 = labels_by_attr_gdf.geometry.to_numpy().tolist()
    geom2 = [
        detections_by_attr_gdf.loc[detections_by_attr_gdf.EGID == egid, 'geometry'].iloc[0]
        if egid in detections_by_attr_gdf.EGID.unique() else GeometryCollection()
        for egid in labels_by_attr_gdf.EGID.to_numpy()
    ]
    iou = []
    for (i, ii) in zip(geom1, geom2):
        iou.append(intersection_over_union(i, ii))
    labels_by_attr_gdf['IOU_' + attribute] = iou

    return labels_by_attr_gdf


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


def relative_error_df(df, target, measure):
    """Compute relative error between 2 df columns

    Args:
        df: dataframe
        target_col (string): name of the target column in the df
        measure_col (string): name of the measured column in the df

    Returns:
        out (df): dataframe relative error computed
    """

    re = abs(df[measure] - df[target]) / df[target]
    re.replace([np.inf], 1.0, inplace=True)

    return re


def tag(gt, dets, threshold, method, buffer=0.001, gt_prefix='gt_', dets_prefix='dt_', group_attribute=None):
    """Tag labels and detections with "charges". 
    This method reserves the label and detection numbers by not duplicating or omitting to count a label or detection.
    A fractionnal "charge" will be assigned to labels/detections belonging to an identified group
    cf https://tech.stdl.ch/PROJ-TREEDET/#24-post-processing-assessment-algorithm-and-metrics-computation for more information

    Args:
        gt (geodataframe): geodataframe of the ground truth
        dets (geodataframe): geodataframe of the detections
        threshold (float):  threshold to apply on the percentage of overlap to determine TP and FP
        method (string): method to use for the charge attribution, possiblities are "charges" and "fusion"
        buffer (float, optional): buffer (in meter) applied to shapes to avoid them sharing border and being assigned to the same group without proper overlap. Defaults to 0.001.
        gt_prefix (str, optional): prefix used to identified labels groups. Defaults to "gt_".
        det_prefix (str, optional): prefix used to identified detections groups. Defaults to "dt_".

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
        for row in l_join[l_join.geohash_gt.notnull()].itertuples():
            g.add_edge(row.geohash_dt, row.geohash_gt)

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
    _gt['geometry'] = _gt.geometry.buffer(buffer, join_style=2)
    _gt = _gt[~_gt.is_empty].copy()
    _dets = dets.copy()
    _dets['geometry'] = _dets.geometry.buffer(buffer, join_style=2)
    _dets = _dets[~_dets.is_empty].copy()

    charges_dict = {}

    # spatial joins
    l_join = gpd.sjoin(_dets, _gt, how='left', predicate='intersects', lsuffix='dt', rsuffix='gt')
    r_join = gpd.sjoin(_dets, _gt, how='right', predicate='intersects', lsuffix='dt', rsuffix='gt')
    if group_attribute:
        l_join.loc[l_join[group_attribute + '_dt'] != l_join[group_attribute + '_gt'], 'geohash_gt'] = np.nan
        r_join.loc[r_join[group_attribute + '_dt'] != r_join[group_attribute + '_gt'], 'geohash_dt'] = np.nan

    # trivial False Positives
    trivial_FPs = l_join[l_join.geohash_gt.isna()].copy()
    for tup in trivial_FPs.itertuples():
        charges_dict = {
            **charges_dict,
            tup.geohash_dt: {
                'FP_charge': Fraction(1, 1),
                'TP_charge': Fraction(0, 1)
            }
        }

    # trivial False Negatives
    trivial_FNs = r_join[r_join.geohash_dt.isna()].copy()
    for tup in trivial_FNs.itertuples():
        charges_dict = {
            **charges_dict,
            tup.geohash_gt: {
                'FN_charge': Fraction(1, 1),
                'TP_charge': Fraction(0, 1)
            }
        }

    # less trivial cases
    groups = make_groups()

    for group in groups:
        all_geoms_gt = gt[gt['geohash'].isin(group)].geometry.values.tolist()
        all_geohashes_gt = gt[gt['geohash'].isin(group)].geohash.values.tolist()        
        all_geoms_dets = dets[dets['geohash'].isin(group)].geometry.values.tolist()
        all_geohashes_dets = dets[dets['geohash'].isin(group)].geohash.values.tolist()

        # filter detections and labels based on intersection area fraction
        keep_geohashes_gt = []
        keep_geohashes_dets = []

        geom_gt = unary_union(all_geoms_gt)
        
        for (geom_det, geohash_det) in zip(all_geoms_dets, all_geohashes_dets):
            polygon_gt_shape = geom_gt
            polygon_det_shape = geom_det
            if polygon_gt_shape.intersects(polygon_det_shape):
                intersection = polygon_gt_shape.intersection(polygon_det_shape).area
            else:
                continue
            # keep element if intersection overlap % of GT and detection shape relative to the detection area is >= THD
            if intersection / polygon_det_shape.area >= threshold:
                keep_geohashes_dets.append(geohash_det)

        for (geom_gt, geohash_gt) in zip(all_geoms_gt, all_geohashes_gt):
            for (geom_det, geohash_det) in zip(all_geoms_dets, all_geohashes_dets):
                polygon_gt_shape = geom_gt
                polygon_det_shape = geom_det
                if polygon_gt_shape.intersects(polygon_det_shape):
                    intersection = polygon_gt_shape.intersection(polygon_det_shape).area
                else:
                    continue
                # keep element if intersection overlap % of GT and detection shape relative to the detection area is >= THD or the detection
                # is already a TP.
                if intersection / polygon_det_shape.area >= threshold or ((geohash_det in keep_geohashes_dets) and (intersection / polygon_gt_shape.area >= 0.5)):
                    keep_geohashes_gt.append(geohash_gt)

        # list of elements to be deleted that do not meet the threshold conditions for the intersection zone 
        remove_geohashes_gt = [x for x in all_geohashes_gt if x not in np.unique(keep_geohashes_gt).astype(str)]
        remove_geohashes_dets = [x for x in all_geohashes_dets if x not in np.unique(keep_geohashes_dets).astype(str)]
        
        # remove elements without enough overlap and attribute TP, FP and FN charges 
        for i in remove_geohashes_gt:
            group.remove(i)
            charges_dict = {
                **charges_dict,
                i: {
                'FN_charge': Fraction(1, 1),
                'TP_charge': Fraction(0, 1)
                }
            }
        for i in remove_geohashes_dets:
            group.remove(i)
            charges_dict = {
                **charges_dict,
                i: {
                'FP_charge': Fraction(1, 1),
                'TP_charge': Fraction(0, 1)
                }
            }
        
        # attriute TP, FP and FN charges of remaining elements
        group_assessment = assess_group(group)
        this_group_charges_dict = {}
        
        for el in group:
            if el.startswith(dets_prefix):
                if method == 'charges':
                    this_group_charges_dict[el] = {
                        'TP_charge': Fraction(min(group_assessment['cnt_gt'], group_assessment['cnt_dets']), group_assessment['cnt_dets']),
                        'FP_charge': Fraction(group_assessment['FP_charge'], group_assessment['cnt_dets'])
                        }
                        
                else:
                    this_group_charges_dict[el] = {
                        'TP_charge': group_assessment['cnt_gt'],
                        'FP_charge': 0
                        }        
            if el.startswith(gt_prefix):
                if method == 'charges':
                    this_group_charges_dict[el] = {
                        'TP_charge': Fraction(min(group_assessment['cnt_gt'], group_assessment['cnt_dets']), group_assessment['cnt_gt']),
                        'FN_charge': Fraction(group_assessment['FN_charge'], group_assessment['cnt_gt'])
                        }
                else:
                    this_group_charges_dict[el] = {
                        'TP_charge': 1, 
                        'FN_charge': 0
                        }  
        
        charges_dict = {**charges_dict, **this_group_charges_dict}

    # remove the buffer applied before group assignement to recover original geometry 
    _gt['geometry'] = _gt.geometry.buffer(-buffer, join_style=2)
    _dets['geometry'] = _dets.geometry.buffer(-buffer, join_style=2)

    _gt = _gt.apply(lambda row: assign_groups(row), axis=1)
    _dets = _dets.apply(lambda row: assign_groups(row), axis=1)

    _gt = _gt.apply(lambda row: assign_charges(row), axis=1)
    _dets = _dets.apply(lambda row: assign_charges(row), axis=1)

    if method == 'fusion':
        unique_dets_gdf = _dets[_dets['group_id'].isna()] 
        dissolved_dets_gdf = _dets.dissolve(by='group_id', as_index=False)
        fused_dets_gdf = pd.concat([unique_dets_gdf, dissolved_dets_gdf]).reset_index(drop=True)

        return _gt[gt.columns.to_list() + ['group_id', 'TP_charge', 'FN_charge']], fused_dets_gdf[dets.columns.to_list() + ['group_id', 'TP_charge', 'FP_charge']]

    else:
        return _gt[gt.columns.to_list() + ['group_id', 'TP_charge', 'FN_charge']], _dets[dets.columns.to_list() + ['group_id', 'TP_charge', 'FP_charge']]