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

import unittest
import gdal
import numpy as np
import shutil
from tempfile import mkdtemp
from pkgutil import get_data
import pkg_resources

from utils import raster_tools as rt


class RasterTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_folder = mkdtemp()
        self.mtspcs_file = pkg_resources.resource_filename('data',
                                                           'LT05_L1GS_036029_20060523_test_MTSPCS.tif')
        self.mtspcs_dataset = gdal.Open(self.mtspcs_file)
        self.mtspcs_dtype = self.mtspcs_dataset.GetRasterBand(1).DataType
        self.wgs_arr = np.array(self.wgs_dataset.GetRasterBand(1).ReadAsArray(), dtype=float)
        self.mtspcs_geo = {'cols': self.mtspcs_dataset.RasterXSize, 'rows': self.mtspcs_dataset.RasterYSize,
                           'bands': self.mtspcs_dataset.RasterCount,
                           'data_type': self.mtspcs_dtype,
                           'projection': self.mtspcs_dataset.GetProjection(),
                           'geotransform': self.mtspcs_dataset.GetGeoTransform(),
                           'resolution': self.mtspcs_dataset.GetGeoTransform()[1]}
        self.mtspcs_arr = np.array(self.mtspcs_dataset.GetRasterBand(1).ReadAsArray(), dtype=float)

        self.wgs_file = pkg_resources.resource_filename('data',
                                                        'LT05_L1GS_036029_20060523_test_WGS84.tif')
        self.wgs_dataset = gdal.Open(self.wgs_file)
        self.wgs_dtype = self.mtspcs_dataset.GetRasterBand(1).DataType
        self.mtspcs_geo = {'cols': self.wgs_dataset.RasterXSize, 'rows': self.wgs_dataset.RasterYSize,
                           'bands': self.wgs_dataset.RasterCount,
                           'data_type': self.wgs_dtype,
                           'projection': self.wgs_dataset.GetProjection(),
                           'geotransform': self.wgs_dataset.GetGeoTransform(),
                           'resolution': self.mtspcs_dataset.GetGeoTransform()[1]}

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
        self.assertAlmostEqual(arr.sum(), self.mtspcs_arr.sum())


if __name__ == '__main__':
    unittest.main()

# ===============================================================================
