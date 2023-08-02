
import os, sys
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import rasterio
from rasterio.windows import Window



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

        file_path=os.path.join(ensure_dir_exists(os.path.join(output, 'crop')),
                                 source.split('/')[-1].split('.')[0] + '_crop.tif')   

        with rasterio.open(file_path, 'w', **profile) as dst:
            # Read the data from the window and write it to the output raster
            dst.write(src.read(window=window))  

        return file_path
    

def IOU(pol1_xy, pol2_xy):
    # Define each polygon
    polygon1_shape = pol1_xy
    polygon2_shape = pol2_xy

    # print(polygon1_shape, polygon2_shape)
    # Calculate intersection and union, and tne IOU
    polygon_intersection = polygon1_shape.intersection(polygon2_shape).area
    polygon_union = polygon1_shape.area + polygon2_shape.area - polygon_intersection
    return round(polygon_intersection / polygon_union, 3)


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

def test_crs(crs1, crs2 = "EPSG:2056"):
    '''
    Take the crs of two dataframes and compare them. If they are not the same, stop the script.
    '''
    if isinstance(crs1, gpd.GeoDataFrame):
        crs1=crs1.crs
    if isinstance(crs2, gpd.GeoDataFrame):
        crs2=crs2.crs

    try:
        assert(crs1 == crs2), f"CRS mismatch between the two files ({crs1} vs {crs2})."
    except Exception as e:
        print(e)
        sys.exit(1)

def ensure_dir_exists(dirpath):
    '''
    Test if a directory exists. If not, make it.
    return: the path to the verified directory.
    '''

    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
        print(f"The directory {dirpath} was created.")

    return dirpath



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

    # CREATE ADDITIONAL COLUMN FOR TP, FP AND FN CLASSIFICATION AND IOU COMPUTATION
    labels_gdf['geom_GT'] = labels_gdf.geometry
    preds_gdf['geom_DET'] = preds_gdf.geometry

    # TRUE POSITIVES
    left_join = gpd.sjoin(preds_gdf, labels_gdf, how='left', predicate='intersects', lsuffix='left', rsuffix='right')
    tp_gdf_temp = left_join[left_join.ID_GT.notnull()].copy()

    # IOU computation between GT geometry and Detection geometry
    geom1 = (tp_gdf_temp['geom_DET'].to_numpy()).tolist()
    geom2 = (tp_gdf_temp['geom_GT'].to_numpy()).tolist()
    iou = []
    for (i, ii) in zip(geom1, geom2):
        iou.append(IOU(i, ii))
    tp_gdf_temp['IOU'] = iou

    # Filter detection based on IOU value
    # Keep only max IOU value for each detection mask
    tp_gdf = tp_gdf_temp.groupby(['value'], group_keys=False).apply(lambda g:g[g.IOU == g.IOU.max()])
    
    # Detection with IOU lower than threshold value are considered as FP and removed from TP list   
    threshold_iou = 0.1
    fp_gdf_temp = tp_gdf[tp_gdf['IOU'] < threshold_iou]
    val_fp = fp_gdf_temp['value'].unique().tolist()
    tp_gdf = tp_gdf[~tp_gdf['value'].isin(val_fp)]


    # FALSE POSITIVES -> potentially object not referenced in ground truth or mistakes
    fp_gdf = left_join[left_join.ID_GT.isna()].copy()
    fp_gdf = pd.concat([fp_gdf, fp_gdf_temp])
    assert(len(fp_gdf[fp_gdf.duplicated()]) == 0)


    # FALSE NEGATIVES -> objects that have been missed by the algorithm
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
    
    TP = len(tp_gdf)
    FP = len(fp_gdf)
    FN = len(fn_gdf)
    #print(TP, FP, FN)
    
    if TP == 0:
        return 0, 0, 0

    precision = TP / (TP + FP)
    recall    = TP / (TP + FN)
    f1 = 2*precision*recall/(precision+recall)
    
    return precision, recall, f1
