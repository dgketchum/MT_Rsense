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
from subprocess import call
from itertools import izip

# local imports ======================================================


def recursive_raster_merge(in_folder, out_loc, out_named, level=0):

    # this still places tile incorrectly
    tifs = [os.path.join(in_folder, x) for x in os.listdir(in_folder) if x.endswith('.tif')]

    level += 1
    print 'length of tif list: {}'.format(len(tifs))
    print 'tif list\n{}'.format(tifs)
    tifs.append(None)
    pairs = izip(tifs[::2], tifs[1::2])
    new_dir = out_loc + str(level)
    for i in pairs:
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)
        merge_rasters(i, new_dir)
    recursive_raster_merge(new_dir, new_dir, out_named, level=level)


def merge_rasters(tif_subset, out_location):

    try:
        tif_string = ' '.join(tif_subset)
        # print '\ntif inputs: {}'.format(tif_string)

        name_1, name_2 = tif_subset[0].split('.')[0], tif_subset[1].split('.')[0]
        name_1, name_2 = name_1[-5:], name_2[-5:]
        out_string = '{}.tif'.format('_'.join([name_1, name_2]))
        print 'saving... {}'.format(out_string)

        merge = 'gdal_merge.py -co COMPRESS=DEFLATE -o {} {}'.format(os.path.join(out_location, out_string), tif_string)
        print 'merge cmd: {}'.format(merge)

    except TypeError:
        out_string = tif_subset[0][-3:]
        merge = 'gdal_merge.py -o {} {}'.format(os.path.join(out_location, out_string), tif_subset)

    call(merge, shell=True)


if __name__ == '__main__':
    home = os.path.expanduser('~')
    print 'home: {}'.format(home)
    images = os.path.join(home, 'images')
    tiles = os.path.join(images, 'DEM', 'elevation_NED30M_id_22371_01', 'elevation')
    recursive_raster_merge(tiles, tiles, 'mt_dem_full_30m_test2.tif')


# ============= EOF ============================================================
