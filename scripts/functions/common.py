
from shapely.geometry import Polygon

def bbox(bounds):

    minx = bounds[0]
    miny = bounds[1]
    maxx = bounds[2]
    maxy = bounds[3]

    return Polygon([[minx, miny],
                    [maxx,miny],
                    [maxx,maxy],
                    [minx, maxy]])