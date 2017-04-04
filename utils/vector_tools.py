# ===============================================================================
# Copyright 2017 dgketchum
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===============================================================================
import os
from osgeo import ogr


def lat_lon_to_ogr_point(lat, lon):
    point = ogr.Geometry(ogr.wkbPoint)
    point.AddPoint(lat, lon)
    return point


def read_shapes(shape):
    reader = ogr.Open(shape)
    layer = reader.GetLayer(0)
    shape_geos = []
    for i in range(layer.GetFeatureCount()):
        feature = layer.GetFeature(i)
        shape_geos.append(feature)

    return shape_geos


def find_point_poly_intersect(shapes, points):
    points = [point for point in read_shapes(points)]
    multi_polygon = read_shapes(shapes)
    path_row_pairs = []
    for i, pt in enumerate(points):
        for poly in multi_polygon:
            point = ogr.Geometry(pt['geometry'])
            if point.within(ogr.Geometry(poly['geometry'])):
                path, row = get_path_row(ogr.Geometry(points[i]['geometry']))
                path_row_pairs.append((path, row))

    return path_row_pairs


def get_path_row(layer, shape=None):

    path_list = []

    if isinstance(layer, ogr.Layer):
        try:
            for feature in layer:
                path = str(feature.GetField('PATH'))
                row = str(feature.GetField('ROW'))
                path_list.append((path.rjust(3, '0'), row.rjust(3, '0')))
        except TypeError:
            pass

        print 'number of tiles : {}'.format(len(path_list))
        return path_list

    else:
        raise NotImplementedError('get_path_row requires osgeo.ogr.Layer instance')


if __name__ == '__main__':
    home = os.path.expanduser('~')
    print 'home: {}'.format(home)

    # ===============================================================================
