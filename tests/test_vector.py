#! /usr/bin/env python
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
import unittest
from tempfile import mkdtemp
from osgeo import ogr

from utils import vector_tools


class TestVector(unittest.TestCase):
    def setUp(self):
        self.temp_foler = mkdtemp()
        self.out_shp = os.path.join(self.temp_foler, 'out.shp')
        self.lat = 46.99
        self.lon = -109.61
        self.poly_interior_points = [(-110.4, 48.3), (-108.1, 47.9), (-108.7, 46.6),
                                     (-110.8, 46.9)]
        self.tup = (38, 27)
        self.lst = [(38, 27)]
        self.points = [(488245.30243298115, 292668.1008154524), (488238.7897641528, 292669.6144005801),
                       (488416.7260886859, 293328.3133520024), (490445.02364716836, 300846.8477201403),
                       (529685.3760580911, 450127.9607610224), (531409.2720712859, 456862.50326944795),
                       (531788.1465979685, 458344.7094121693), (531803.6518557359, 458341.1182646604),
                       (712217.0961924061, 419630.0785964296), (711273.0359873119, 415925.9308217316),
                       (670521.1234916867, 260742.92484704658), (668624.2008179231, 253730.22862013045),
                       (488245.30243298115, 292668.1008154524)]
        self.field_attrs = dict(ID=1, PATH=38, ROW=27)

    def test_point(self):
        # test with AmeriFlux US-MJ-1
        point = vector_tools.lat_lon_to_ogr_point(self.lat, self.lon)
        self.assertTrue(type(point), ogr.Geometry)
        with self.assertRaises(TypeError):
            type(point) is not ogr.Geometry

    def test_pt_topoly_geo(self):
        self.poly = vector_tools.points_to_ogr_polygon(self.points)
        self.assertTrue(type(self.poly), ogr.Geometry)
        with self.assertRaises(TypeError):
            type(self.poly) is not ogr.Geometry

    def test_poly_to_shp(self):
        self.shp = vector_tools.points_to_shapefile(self.poly, self.out_shp)
        self.assertTrue(ogr.Open(self.shp), ogr.DataSource)

    def test_shp_to_feat(self):
        feat = vector_tools.shp_to_ogr_features(self.shp)
        self.assertTrue(type(feat), ogr.Feature)

    def test_shp_to_geom(self):
        geo = vector_tools.shp_to_ogr_geometries(self.shp)
        self.assertTrue(type(geo[0]), ogr.Geometry)
        self.assertTrue(type(geo), list)

    def test_pathrow_from_field(self):
        path_list = vector_tools.get_pr_from_field(self.shp)
        self.assertTrue(type(path_list), list)
        self.assertTrue(type(path_list[0]), tuple)
        for item in path_list:
            self.assertEqual(len(str(item[0])), 3)
            self.assertEqual(len(str(item[1])), 3)

    def test_point_multipath(self):
        tup_result = vector_tools.get_pr_multipath(self.tup, self.shp)
        lst_result = vector_tools.get_pr_multipath(self.lst, self.shp)
        points_shp = vector_tools.points_to_shapefile()
        shp_result = vector_tools.get_pr_multipath()


if __name__ == '__main__':
    unittest.main()

# ===============================================================================
