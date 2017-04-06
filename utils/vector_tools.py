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


def lat_lon_to_ogr_point(lon, lat):
    point = ogr.Geometry(ogr.wkbPoint)
    point.AddPoint(lon, lat)
    return point


def points_to_shapefile(points_x_y, output_file):
    geo = ogr.Geometry(ogr.wkbPoint)
    for pt in points_x_y:
        geo.AddPoint(pt[0], pt[1])

    driver = ogr.GetDriverByName('Esri Shapefile')
    ds = driver.CreateDataSource(output_file)
    layer = ds.CreateLayer('', None, ogr.wkbPoint)
    defn = layer.GetLayerDefn()
    feat = ogr.Feature(defn)
    feat.SetField('id', 123)
    feat.SetGeometry(geo)
    layer.CreateFeature(feat)


def points_to_ogr_polygon(args):
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for point in args:
        ring.AddPoint(point[0], point[1])
    ring_poly = ogr.Geometry(ogr.wkbPolygon)
    ring_poly.AddGeometry(ring)
    return ring_poly


def shp_poly_to_pts_list(poly, include_z_vals=False):
    ds = ogr.Open(poly)
    layer1 = ds.GetLayer()
    print layer1.GetExtent()
    for feat in layer1:
        geom = feat.GetGeometryRef()
        ring = geom.GetGeometryRef(0)
        point_ct = ring.GetPointCount()
    points = []
    for p in xrange(point_ct):
        lon, lat, z = ring.GetPoint(p)
        points.append((lon, lat, z))
        if include_z_vals:
            print 'Points from shape: {}'.format(points)
            return points
    latlon = []
    for ll in points:
        lat, lon = ll[0], ll[1]
        latlon.append((lat, lon))
    print 'Points from shape: {}'.format(latlon)
    return latlon


def poly_to_shp(polygon, output_file, field_attr_dict=None):
    driver = ogr.GetDriverByName('Esri Shapefile')
    ds = driver.CreateDataSource(output_file)
    layer = ds.CreateLayer('', None, ogr.wkbPolygon)
    defn = layer.GetLayerDefn()
    feat = ogr.Feature(defn)
    if field_attr_dict:
        for key, val in field_attr_dict:
            feat.SetFieldInteger64('ID', key)
            if type(val) is dict:
                for second_key, sub_val in val:
                    if second_key == 'PATH':
                        feat.SetFieldInteger64('PATH', sub_val)
                    elif second_key == 'ROW':
                        feat.SetFieldInteger64('ROW', sub_val)
                    else:
                        print 'There are fields not set: {} has {}'.format(second_key, sub_val)

    feat.SetField('id', 123)
    feat.SetGeometry(polygon)
    layer.CreateFeature(feat)
    print ds
    return None


def shp_to_ogr_features(shape):
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


def get_pr_multipath(points, poly_shapefile):

    path_list = []

    if isinstance(points, tuple):
        pt_geo_refs = lat_lon_to_ogr_point(points[0], points[1])
    elif isinstance(points, list):
        pt_geo_refs = []
        for pt in points:
            pt_geo_refs.append(lat_lon_to_ogr_point(pt[0], pt[1]))
    elif isinstance(points, str):
        pt_features = shp_to_ogr_features(points)
        poly_features = shp_to_ogr_features(poly_shapefile)
        pt_geo_refs = [pt.GetGeometryRef() for pt in pt_features]
        poly_geo_refs = [pl.GetGeometryRef() for pl in poly_features]
    else:
        raise NotImplementedError('Function takes first arg type tuple, list, or string')

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
    poly = os.path.join(home, 'images', 'vector_data', 'MT_SPCS_vector', 'US_MJ_tile.shp')
    out_file = os.path.join(home, 'images', 'test_data', 'points_out.shp')
    lat, lon = 44.91, -106.55
    print get_pr_multipath(flux_sites, poly)

# ===============================================================================
