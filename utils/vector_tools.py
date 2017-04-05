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


def shp_to_org_features(shape):
    reader = ogr.Open(shape)
    layer = reader.GetLayer()
    features = []
    for i in range(layer.GetFeatureCount()):
        feature = layer.GetFeature(i)
        features.append(feature)
    return features


def shp_to_ogr_geometries(shape):
    reader = ogr.Open(shape)
    layer = reader.GetLayer()
    geometries = []
    for feature in layer:
        geom = feature.GetGeometryRef()
        geometries.append(geom)
    return geometries


def get_pr_from_field(shapefile):
    path_list = []
    for feature in shapefile:
        path = str(feature.GetField('PATH'))
        row = str(feature.GetField('ROW'))
        path_list.append((path.rjust(3, '0'), row.rjust(3, '0')))
    return path_list


def get_pr_from_lat_lon_shp_intersect(lat, lon, poly_shapefile):
    path_list = []
    point = lat_lon_to_ogr_point(lat, lon)
    poly_features = shp_to_org_features(poly_shapefile)
    poly_geo_refs = [poly.GetGeometryRef() for poly in poly_features]
    for j, polygon in enumerate(poly_geo_refs):
        if point.Within(polygon):
            path, row = poly_features[j].GetField('PATH'), poly_features[j].GetField('ROW')
            print path, row

    print 'Path list: {}'.format(path_list)
    return path_list


def get_multipoint_multipath_shp_intesects(pt_shapefile, poly_shapefile):
    path_list = []
    pt_features = shp_to_org_features(pt_shapefile)
    poly_features = shp_to_org_features(poly_shapefile)
    pt_geo_refs = [pt.GetGeometryRef() for pt in pt_features]
    poly_geo_refs = [poly.GetGeometryRef() for poly in poly_features]
    for point in pt_geo_refs:
        for j, polygon in enumerate(poly_geo_refs):
            if point.Within(polygon):
                path, row = poly_features[j].GetField('PATH'), poly_features[j].GetField('ROW')
                path_list.append((path, row))
    print 'number of tiles: {}'.format(len(path_list))
    return path_list


if __name__ == '__main__':
    home = os.path.expanduser('~')
    print 'home: {}'.format(home)
    flux_sites = os.path.join(home, 'images', 'vector_data', 'MT_SPCS_vector', 'amf_mt_SPCS.shp')
    poly = os.path.join(home, 'images', 'vector_data', 'MT_SPCS_vector', 'MT_row_paths.shp')
    lat, lon = 44.91, -106.55
    print get_pr_from_lat_lon_shp_intersect(lat, lon, poly)

# ===============================================================================
