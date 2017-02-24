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
import numpy as np
import gdal
from subprocess import call


# local imports ======================================================


def merge_rasters(in_folder, out_location, out_name):
    tifs = [os.path.join(in_folder, x) for x in os.listdir(in_folder) if x.endswith('.tif')]
    print 'tif files: \n {}'.format(tifs)
    tif_string = ' '.join(tifs)
    print 'tif string to save: {}'.format(tif_string)

    source_epsg = 4326
    target_epsg = 32100

    print 'working space: {}'.format(os.getcwd())
    os.chdir(in_folder)
    vrt_str = 'vrt_merge_id.vrt'

    vrts = 'gdalbuildvrt -allow_projection_difference {} {}'.format(vrt_str, tif_string)
    call(vrts, shell=True)

    warp = 'gdal_warp -s_srs {} -t_srs {} -tr 30 -r cubic -srcnodata 0.0 dstnodata 0.0 \n' \
           '{} {}'.format(source_epsg,
                          target_epsg, vrt_str, os.path.join(out_location, out_name))

    print 'warp cmd: {}'.format(warp)
    call(warp, shell=True)


if __name__ == '__main__':
    home = os.path.expanduser('~')
    print 'home: {}'.format(home)
    images = os.path.join(home, 'images')
    tiles = os.path.join(images, 'DEM', 'elevation_NED30M_id_22371_01', 'elevation')
    merge_rasters(tiles, os.path.join(images, 'DEM'), 'id_dem_full_30m_proj.tif')


# ============= EOF ============================================================
