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

# standard library imports ======================================================
import os
import gdal
import osr


def merge_rasters(in_folder, out_location, out_proj):
    tifs = [os.path.join(in_folder, x) for x in os.listdir(in_folder) if x.endswith('.tif')]

    print 'tif files: \n {}'.format(tifs)
    tif_string = ' '.join(tifs)
    print 'tif string to save: {}'.format(tif_string)

    t_srs = osr.SpatialReference()
    t_srs.ImportFromEPSG(out_proj)
    t_proj = t_srs.ExportToWkt()
    print 'target srs strirng: {}'.format(t_proj)

    for tif in tifs:
        dataset = gdal.Open(tif)
        band = dataset.GetRasterBand(1)
        band_ct = dataset.RasterCount
        geo_t = dataset.GetGeoTransform()
        proj_info = dataset.GetProjection()
        src_srs = osr.SpatialReference()
        src_srs.ImportFromWkt(proj_info)
        src_proj = src_srs.ExportToWkt()
        print 'source srs: {}'.format(src_proj)

        driver = gdal.GetDriverByName('GTiff')
        out_name = tif.replace('.tif', '_32100.tif')
        print 'out name: {}'.format(out_name)
        # dest = driver.Create(out_name, dataset.RasterXSize, dataset.RasterYSize,
        #                      band_ct, band.DataType)
        dest = driver.ReprojectImage(dataset, dest, src_proj, t_srs, gdal.GRA_Cubic)
        dest.SetGeoTransform(geo_t)
        dest.SetProjection(t_proj)
        out_band = dest.GetRasterBand(1)
        dest = None

        # driver = gdal.GetDriverByName('GTiff')
        # out_data_set = driver.Create(filename, self._geo['cols'], self._geo['rows'],
        #                              self._geo['bands'], self._geo['data_type'])
        # out_data_set.SetGeoTransform(self._geo['geotransform'])
        # out_data_set.SetProjection(self._geo['projection'])
        # output_band = out_data_set.GetRasterBand(1)
        # output_band.WriteArray(array_to_save, 0, 0)
        #
        # raster_geo_dict = {'cols': dataset.RasterXSize, 'rows': dataset.RasterYSize, 'bands': dataset.RasterCount,
        #            'data_type': band.DataType, 'projection': dataset.GetProjection(),
        #            'geotransform': dataset.GetGeoTransform(), 'resolution': dataset.GetGeoTransform()[1]}



if __name__ == '__main__':
    home = os.path.expanduser('~')
    # print 'home: {}'.format(home)
    images = os.path.join(home, 'images')
    tiles = os.path.join(images, 'DEM', 'elevation_NED30M_id_22371_01', 'elevation')
    merge_rasters(tiles, os.path.join(images, 'DEM'), out_proj=32100)


# ============= EOF ============================================================
