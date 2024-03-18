
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

sys.path.insert(1, 'scripts')
import functions.fct_misc as misc


def area_comparisons(egid_surfaces_df, surfaces_df, attribute_surfaces_df, surface_type):

    if surface_type == 'occupied':
        surface_type = 'occup'
    elif surface_type != 'free':
        logger.critical('The surface type is not valid. Please pass "occupied" or "free".')
        sys.exit(1)

    # Compute relative error
    # by EGID
    egid_surfaces_df[f'{surface_type}_rel_error'] = misc.relative_error_df(egid_surfaces_df, target=f'{surface_type}_area_labels', measure=f'{surface_type}_area_dets')
    # total
    surfaces_df[f'{surface_type}_rel_error'] = misc.relative_error_df(surfaces_df, target=f'{surface_type}_area_labels', measure=f'{surface_type}_area_dets')
    # by attriubte
    attribute_surfaces_df[f'{surface_type}_rel_error'] = misc.relative_error_df(attribute_surfaces_df, target=f'{surface_type}_area_labels', measure=f'{surface_type}_area_dets')
    
    return egid_surfaces_df, surfaces_df, attribute_surfaces_df


def area_estimation(objects_df, egid_surfaces_df, surface_type, object_type, BINS, roof_attributes, surfaces_df=None, attribute_surfaces_df=None):

    if surface_type == 'occupied':
        surface_type = 'occup'
    elif surface_type != 'free':
        logger.critical('The surface type is not valid. Please pass "occupied" or "free".')
        sys.exit(1)

    if object_type == 'detections':
        object_type = 'dets'
    elif object_type != 'labels':
        logger.critical('The object type is not valid. Please pass "detections" or "labels".')
        sys.exit(1)

    egid_surfaces_df[f'{surface_type}_area_{object_type}'] = [
        objects_df.loc[objects_df.EGID == egid, f'{surface_type}_area'].iloc[0]
        if egid in objects_df.EGID.unique() else 0
        for egid in egid_surfaces_df.EGID.unique() 
    ]

    # Warn in case of negative values in surface computation
    nbr_tmp = egid_surfaces_df.loc[egid_surfaces_df[f'{surface_type}_area_{object_type}'] < 0].shape[0]
    if nbr_tmp > 0:
        logger.warning(f'{nbr_tmp} calculated {surface_type} surfaces for the {object_type} are smaller than 0. Those are set to 0.')
        egid_surfaces_df.loc[egid_surfaces_df[f'{surface_type}_area_{object_type}'] < 0, f'{surface_type}_area_{object_type}'] = 0.0

    # Attribute bin to surface area
    bin_labels = [f"{BINS[i]}-{BINS[i+1]}" for i in range(len(BINS)-1)]

    egid_surfaces_df[f'ratio_{surface_type}_area_{object_type}'] = egid_surfaces_df[f'{surface_type}_area_{object_type}']/egid_surfaces_df['total_area']
    egid_surfaces_df[f'bin_{surface_type}_area_{object_type} (%)'] = pd.cut(
        egid_surfaces_df[f'ratio_{surface_type}_area_{object_type}'] * 100, BINS, right=False, labels=bin_labels
    )

    # Get the global area
    if not isinstance(attribute_surfaces_df, pd.DataFrame):
        surfaces_df = pd.DataFrame.from_records([{'attribute': 'EGID', 'value': 'ALL'}])
    surfaces_df[f'{surface_type}_area_{object_type}'] = [egid_surfaces_df[f'{surface_type}_area_{object_type}'].sum()]
    surfaces_df['total_area'] = egid_surfaces_df['total_area'].sum()
    surfaces_df[f'ratio_{surface_type}_area_{object_type}'] = surfaces_df[f'{surface_type}_area_{object_type}'] / surfaces_df['total_area']

    # Compute area by roof attributes
    tmp_df = pd.DataFrame()
    surface_types = [f'{surface_type}_area_{object_type}', 'total_area', f'ratio_{surface_type}_area_{object_type}']
    attribute_surface_dict = {'attribute': [], 'value': []}

    for attribute in roof_attributes:
        for val in sorted(egid_surfaces_df[attribute].unique()):
            attribute_surface_dict['attribute'] = attribute
            attribute_surface_dict['value'] = val
            for var in surface_types:
                total_area = egid_surfaces_df.loc[egid_surfaces_df[attribute] == val, f'total_area'].sum()
                sum_surface = egid_surfaces_df.loc[egid_surfaces_df[attribute] == val, f'{surface_type}_area_{object_type}'].sum() if var!='total_area'\
                    else total_area
                attribute_surface_dict[var] = sum_surface if var!=f'ratio_{surface_type}_area_{object_type}' \
                    else sum_surface / total_area

            tmp_df = pd.concat([tmp_df, pd.DataFrame(attribute_surface_dict, index=[0])], ignore_index=True)

    if not isinstance(attribute_surfaces_df, pd.DataFrame):
        attribute_surfaces_df = tmp_df.copy()
    else:
        tmp_df.drop(columns=['total_area'], inplace=True)
        attribute_surfaces_df = attribute_surfaces_df.merge(tmp_df, on=['value', 'attribute'])

    return egid_surfaces_df, surfaces_df, attribute_surfaces_df


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
    Each detection can only correspond to one label.
    
    Args:
        tp_gdf_ini (geodataframe): geodataframe of the potiential true positive detection
        threshold (int, optional): threshold to apply on the IoU. Defaults to 0.1.

    Returns:
        geodataframes: geodataframes of the true positive and of the flase positives intersecting labels.
    """

    # Filter detection based on IoU value
    # Keep only max IoU value for each detection
    tp_grouped_gdf = tp_gdf_ini.groupby(['detection_id'], group_keys=False).apply(lambda g:g[g.IoU == g.IoU.max()])
    
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
    Each detection can correspond to several labels.

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


def get_free_area(objects_gdf, roofs_gdf, attribute='EGID'):
    """Compute the occupied and free surface area of all the labels and detection by roof (EGID)

    Args:
        objects_gdf (geodataframe): geodataframe of the detections or the ground truth
        roofs_gdf (geodataframe): geodataframe of the roofs
        attribute (string): attribute to dissolve by. Defaults to 'EGID'.

    Returns:
        labels_free_gdf: geodataframes of all the labels merged by roof with the occupied and free surface area by roof
        objects_free_gdf: geodataframes of all the detections merged by roof with the occupied and free surface area by roof
    """

    
    roofs_by_attribute_gdf = roofs_gdf.dissolve(by=attribute, as_index=False)
    roofs_by_attribute_gdf['roof_area'] = roofs_by_attribute_gdf.area

    objects_by_attribute_gdf = objects_gdf.dissolve(by=attribute, as_index=False) 
    # Add values to empty gdf
    if objects_by_attribute_gdf['geometry'].empty:
        keys_list = objects_by_attribute_gdf.to_dict()
        dic = dict.fromkeys(keys_list, 0)
        objects_by_attribute_gdf = pd.DataFrame.from_dict(dic, orient='index').T
        objects_by_attribute_gdf['occup_area'] = 0
        objects_by_attribute_gdf['EGID'] = roofs_by_attribute_gdf['EGID']
    else:
        objects_by_attribute_gdf['occup_area'] = objects_by_attribute_gdf.area

    objects_with_area_gdf=pd.merge(objects_by_attribute_gdf, roofs_by_attribute_gdf[['EGID', 'roof_area']], on='EGID')
    objects_with_area_gdf['free_area'] = objects_with_area_gdf.roof_area - objects_with_area_gdf.occup_area

    return objects_with_area_gdf


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
    labels_by_attr_gdf['IoU_' + attribute] = iou

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

    # prepare data: apply negative buffer to avoid the intersection of touching shapes and remove empty geometries
    _gt = gt.copy()
    _gt['geometry'] = _gt.geometry.buffer(-buffer, join_style=2)
    _gt = _gt[~_gt.is_empty].copy()
    _dets = dets.copy()
    _dets['geometry'] = _dets.geometry.buffer(-buffer, join_style=2)
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
        
        # remove elements from the label and detection lists without enough overlap and attribute TP, FP and FN charges 
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
    _gt['geometry'] = _gt.geometry.buffer(buffer, join_style=2)
    _dets['geometry'] = _dets.geometry.buffer(buffer, join_style=2)

    _gt = _gt.apply(lambda row: assign_groups(row), axis=1)
    _dets = _dets.apply(lambda row: assign_groups(row), axis=1)

    _gt = _gt.apply(lambda row: assign_charges(row), axis=1)
    _dets = _dets.apply(lambda row: assign_charges(row), axis=1)

    if method == 'fusion':
        unique_detections_gdf = _dets[_dets['group_id'].isna()] 
        dissolved_detections_gdf = _dets.dissolve(by='group_id', as_index=False)
        fused_detections_gdf = pd.concat([unique_detections_gdf, dissolved_detections_gdf]).reset_index(drop=True)

        return _gt[gt.columns.to_list() + ['group_id', 'TP_charge', 'FN_charge']], fused_detections_gdf[dets.columns.to_list() + ['group_id', 'TP_charge', 'FP_charge']]

    else:
        return _gt[gt.columns.to_list() + ['group_id', 'TP_charge', 'FN_charge']], _dets[dets.columns.to_list() + ['group_id', 'TP_charge', 'FP_charge']]
