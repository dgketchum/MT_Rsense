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

    f = open(os.path.join(out_location, 'saveTest.txt'), 'w')
    out_path = os.path.join(out_location, out_name) + ' '
    print 'outpath: {}'.format(out_path)
    f.write(out_path)
    f.write(tif_string)
    f.close()

    # merge = 'gdal_merge.py -o {}, {}'.format(os.path.join(out_location, out_name), tif_string)
    # print 'merge cmd: {}'.format(merge)
    # call(merge, shell=True)

if __name__ == '__main__':
    home = os.path.expanduser('~')
    print 'home: {}'.format(home)
    images = os.path.join(home, 'images')
    tiles = os.path.join(images, 'DEM', 'elevation_NED30M_id_22371_01', 'elevation')
    merge_rasters(tiles, os.path.join(images, 'DEM'), 'mt_dem_full_30m.tif')


# ============= EOF ============================================================

