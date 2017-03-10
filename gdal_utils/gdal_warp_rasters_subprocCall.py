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
import osr, gdal
from subprocess import call
from gdal_utils.gdal_funcs import wkt2epsg

# local imports ======================================================
gdal.UseExceptions()


def merge_rasters(in_folder, out_location, out_name, out_proj):
    tifs = [os.path.join(in_folder, x) for x in os.listdir(in_folder) if x.endswith('.tif')]

    srs_dict = {}

    t_srs = osr.SpatialReference()
    t_srs.ImportFromEPSG(out_proj)
    t_proj = t_srs.ExportToProj4()
    t_wkt = t_srs.ExportToWkt()
    t_epsg = wkt2epsg(t_wkt)
    print 'attempt at get target epsg: {}'.format(t_epsg)
    # t_epsg_str = 'EPSG:{}'.format(t_epsg)

    # create dict with separate entry for each srs
    for tif in tifs:
        dataset = gdal.Open(tif)
        proj_info = dataset.GetProjection()
        src_srs = osr.SpatialReference()
        src_srs.ImportFromWkt(proj_info)
        src_epsg = int(src_srs.GetAuthorityCode(None))

        vrt_str = tif.replace('.tif', '.vrt')

        if src_epsg not in srs_dict:
            srs_dict[src_epsg] = [vrt_str]
        else:
            srs_dict[src_epsg].append(vrt_str)

        # create vrt for each .tif files
        build_vrt = """gdalbuildvrt -overwrite -r bilinear {} {}""".format(vrt_str, tif)
        call(build_vrt, shell=True)

    out_reproj_tifs = []
    for key, val in srs_dict.iteritems():

        for vrt in val:
            if os.path.isfile(vrt):
                print '{} exists'.format(vrt)
            else:
                print '{} does not exist'.format(vrt)

        src_srs = osr.SpatialReference()
        src_srs.ImportFromEPSG(key)
        src_proj = src_srs.ExportToProj4()
        src_epsg = int(src_srs.GetAuthorityCode(None))
        # src_epsg_str = 'EPSG:{}'.format(src_epsg)
        src_wkt = src_srs.ExportToWkt()
        src_epsg = wkt2epsg(src_wkt)
        print 'attempt at get source epsg: {}'.format(src_epsg)

        val = ' '.join(val)
        print 'authority: EPSG {}'.format(src_srs.GetAuthorityCode(None))
        print 'source proj4: {} \n vrt string: {}'.format(src_proj, val)

        print 'working directory: {}'.format(os.getcwd())

        # combine all .vrt of the same srs into one vrt
        out_vrt_file = os.path.join(out_location, '{}.vrt'.format(str(key)))
        build_vrt = """gdalbuildvrt -overwrite -r bilinear {} {}""".format(out_vrt_file, val)
        call(build_vrt, shell=True)

        # reproject all combined .vrt into target projection in GeoTiff
        out_file = os.path.join(out_location, '{}_{}'.format(str(key), out_name))
        out_reproj_tifs.append(out_file)
        print 'output file: {}'.format(out_file)
        warp = """gdalwarp -s_srs '{}' -t_srs '{}' -overwrite -tr 30 30 -r bilinear -dstnodata 0.0 {} {}""" \
            .format(src_proj, t_proj, out_vrt_file, out_file)
        print 'warp vrt-> tif cmd: {}\n'.format(warp)
        call(warp, shell=True)

    # non-working code to merge geometry into an empty raster then mosaic it with warp
    # DEM rasters on which this was developed were too big for merge
    # # create empty tif, merge cmd finishes with {out} {in}
    # print 'reprojected tifs: {}'.format(out_reproj_tifs)
    # merge = """gdal_merge.py -createonly {} {}""".format(' '.join(out_reproj_tifs), out_name)
    # print 'merge cmd: {}\n'.format(merge)
    #
    # # mosaic reprojected tif files into merged tif via warp, warp cmd finishes with {in} {out}
    # warp = """gdalwarp -s_srs '{}' -t_srs '{}' -tr 30 30 -r cubic -srcnodata 0 -dstnodata 0.0 {} {}""" \
    #     .format(t_proj, t_proj, out_name, ' '.join(out_reproj_tifs))
    # call(warp, shell=True)
    #
    # print 'warp mosaic final image  cmd: {}\n'.format(warp)
    # call(warp, shell=True)

if __name__ == '__main__':
    home = os.path.expanduser('~')
    print 'home: {}'.format(home)
    images = os.path.join(home, 'images')
    tiles = os.path.join(images, 'DEM', 'elevation_NED30M_mt_20033_01', 'elevation')
    merge_rasters(tiles, os.path.join(images, 'DEM'), 'mt_dem.tif', out_proj=32100)


# ============= EOF ============================================================
