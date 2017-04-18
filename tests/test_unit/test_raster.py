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
import ogr
import gdal
import shutil
import unittest
import numpy as np
import pkg_resources
from tempfile import mkdtemp

from utils import raster_tools as rt


class RasterTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_folder = mkdtemp()
        self.mtspcs_tif = 'LT05_L1GS_036029_20060523_test_MTSPCS.tif'
        self.mtspcs_file = pkg_resources.resource_filename('data', self.mtspcs_tif)
        self.mtspcs_dataset = gdal.Open(self.mtspcs_file)
        self.mtspcs_dtype = self.mtspcs_dataset.GetRasterBand(1).DataType
        self.mtspcs_geo_known = {'cols': self.mtspcs_dataset.RasterXSize, 'rows': self.mtspcs_dataset.RasterYSize,
                                 'bands': self.mtspcs_dataset.RasterCount,
                                 'data_type': self.mtspcs_dtype,
                                 'projection': self.mtspcs_dataset.GetProjection(),
                                 'geotransform': self.mtspcs_dataset.GetGeoTransform(),
                                 'resolution': self.mtspcs_dataset.GetGeoTransform()[1]}
        self.mtspcs_arr = np.array(self.mtspcs_dataset.GetRasterBand(1).ReadAsArray(), dtype=float)

        self.wgs_tif = 'LT05_L1GS_036029_20060523_test_WGS84.tif'
        self.wgs_file = pkg_resources.resource_filename('data', self.wgs_tif)
        self.wgs_dataset = gdal.Open(self.wgs_file)
        self.wgs_arr = np.array(self.wgs_dataset.GetRasterBand(1).ReadAsArray(), dtype=float)
        self.wgs_dtype = self.mtspcs_dataset.GetRasterBand(1).DataType
        self.wgs_geo_known = {'cols': self.wgs_dataset.RasterXSize, 'rows': self.wgs_dataset.RasterYSize,
                              'bands': self.wgs_dataset.RasterCount,
                              'data_type': self.wgs_dtype,
                              'projection': self.wgs_dataset.GetProjection(),
                              'geotransform': self.wgs_dataset.GetGeoTransform(),
                              'resolution': self.mtspcs_dataset.GetGeoTransform()[1]}

        self.wgs_tile_on = pkg_resources.resource_filename('data',
                                                           'wrs2_036029_WGS.shp')
        self.wgs_tile_off = pkg_resources.resource_filename('data',
                                                            'US_MJ_tile_WGS.shp')
        self.raster_dir = os.path.dirname(self.wgs_file)

    def tearDown(self):
        shutil.rmtree(self.temp_folder)

    def test_setup_types(self):
        self.assertIsInstance(self.wgs_file, str)
        self.assertIsInstance(self.wgs_dataset, gdal.Dataset)
        self.assertIsInstance(self.wgs_arr, np.ndarray)

        self.assertIsInstance(self.mtspcs_file, str)
        self.assertIsInstance(self.mtspcs_dataset, gdal.Dataset)
        self.assertIsInstance(self.mtspcs_arr, np.ndarray)

    def test_raster_to_array(self):
        arr = rt.raster_to_array(self.mtspcs_file)
        self.assertEqual(arr.sum(), self.mtspcs_arr.sum())
        self.assertEqual(arr.shape, self.mtspcs_arr.shape)

    def test_raster_geo(self):
        mtspcs_geo_expected = rt.get_raster_geo_attributes(self.mtspcs_file)
        self.assertIsInstance(mtspcs_geo_expected, dict)

        for key, val in mtspcs_geo_expected.iteritems():
            self.assertEqual(self.mtspcs_geo_known[key], val)
            if key in ['cols', 'rows', 'bands', 'data_type', 'resolution']:
                self.assertEqual(self.wgs_geo_known[key], val)
            elif key in ['projection', 'geotransform']:
                self.assertNotEqual(self.wgs_geo_known[key], val)
            else:
                raise NotImplementedError('You have a key in {} that is unaccounted for.'.format(mtspcs_geo_expected))

        wgs_geo_expected = rt.get_raster_geo_attributes(self.wgs_file)
        for key, val in wgs_geo_expected.iteritems():
            if key in ['projection', 'geotransform']:
                self.assertEqual(self.wgs_geo_known[key], val)
            elif key in ['cols', 'rows', 'bands', 'data_type', 'resolution']:
                pass
            else:
                raise NotImplementedError('You have a key in {} that is unaccounted for.'.format(mtspcs_geo_expected))

    def test_get_polygon_from_raster(self):
        poly = rt.get_polygon_from_raster(self.wgs_file)
        self.assertIsInstance(poly, ogr.Geometry)

    def test_find_pol_ras_intersect(self):
        raster_list = rt.find_poly_ras_intersect(self.wgs_tile_on, self.raster_dir)
        self.assertEqual(raster_list, [self.wgs_tif])


if __name__ == '__main__':
    unittest.main()

# ===============================================================================
