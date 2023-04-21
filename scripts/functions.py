import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from shapely.geometry import shape,Polygon,MultiPolygon,mapping, Point
from loguru import logger

def vectorize(df, array, type, visu):
    
    df_object = pd.DataFrame({'class':[type]})
    df_poly = pd.DataFrame()
    logger.info(f"Compute 2D vector from points groups of type {type}:")
    idx = []
    for i in range(len(array)):
        points = df[df['group'] == array[i]]
        points = points.drop(['Unnamed: 0','Z','group','type'], axis=1) 
        points = points.to_numpy()

        hull = ConvexHull(points)
        area = hull.volume

        if visu == 'True':
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
            polylist.append(points[idx]) #Append this index point to list
        logger.info(f"Group: {array[i]}, number of vertices: {len(polylist)}, area: {(area):.2f}")
        poly = Polygon(polylist)
        df_object['area'] = area # Assuming the OP's x,y coordinates
        df_object['geometry'] = poly
        df_poly = df_poly.append(df_object, ignore_index=True)

    return df_poly
