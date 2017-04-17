"""
This module downloads landsat data.  Get the wrs (ascending)
from http://landsat.usgs.gov/worldwide_reference_system_WRS.php
select an area you want images for, save the selection and
pass shapefile to this program,
or just choose location coordinates
"""
import os
import requests.packages.urllib3
from osgeo import ogr
from datetime import datetime

from landsat.search import Search
from landsat.downloader import Downloader, RemoteFileDoesntExist
from landsat.image import Simple

from utils import olivier
from utils import usgs_download

from utils.vector_tools import get_pr_from_field, get_pr_multipath
from utils.web_tools import convert_lat_lon_wrs2pr

requests.packages.urllib3.disable_warnings()


def download_landsat(start_end_tuple, satellite='L8', path_row_tuple=None, lat_lon_tuple=None,
                     shape=None, output_path=None, seek_multipath=False, multipath_points=None,
                     dry_run=False, max_cloud=None, return_scenes=100, usgs_creds=None):

    start_date, end_date = start_end_tuple[0], start_end_tuple[1]
    print 'Date range: {} to {}'.format(start_date, end_date)

    if shape and not seek_multipath:
        # assumes shapefile has a 'path' and a 'row' field
        ds = ogr.Open(shape)
        lyr = ds.GetLayer()
        image_index = get_pr_from_field(lyr)
        print 'Downloading landsat by row/path shapefile: {}'.format(shape)

    # for case with path,row shapefile and point(s) attempts to get path, row overlapping scenes
    # thereby increasing images/time
    elif seek_multipath:
        image_index = get_pr_multipath(multipath_points, shape)
        print 'Downloading landsat for multipath'
        print 'shapefile: {}'.format(shape)
        print 'points shapefile: {}'.format(multipath_points)

    elif lat_lon_tuple:
        # for case of lat and lon
        image_index = [convert_lat_lon_wrs2pr(lat, lon)]
        print 'Downloading landsat by lat/lon: {}, {}'.format(lat, lon)

    elif path_row_tuple:
        # for case of given path row tuple
        path, row = path_row_tuple[0], path_row_tuple[1]
        image_index = [(path, row)]
        print 'Downloading landsat by path/row: {}, {}'.format(path, row)

    else:
        raise NotImplementedError('Must give path/row tuple, lat/lon tuple plus row/path \n'
                                  'shapefile, or a path/rows shapefile!')

    print 'Image Ind: {}'.format(image_index)

    for tile in image_index:
        path, row = tile[0], tile[1]
        searcher = Search()
        destination_path = os.path.join(output_path, 'd_{}_{}'.format(path, row))
        os.chdir(output_path)

        downer = Downloader(verbose=False, download_dir=destination_path)

        candidate_scenes = searcher.search(paths_rows='{},{},{},{}'.format(path, row, path, row),
                                           start_date=start_date,
                                           end_date=end_date,
                                           cloud_min=0,
                                           cloud_max=max_cloud,
                                           limit=return_scenes)

        print 'total images for tile {} is {}'.format(tile, candidate_scenes['total_returned'])

        x = 0

        if candidate_scenes['status'] == 'SUCCESS':
            for scene_image in candidate_scenes['results']:
                print 'Downloading:', (str(scene_image['sceneID']))
                if not dry_run:
                    try:
                        print 'Downloading tile {} of {}'.format(x, candidate_scenes['total_returned'])
                        downer.download([str(scene_image['sceneID'])])
                        Simple(
                            os.path.join(output_path, destination_path,
                                         '{}.tar.bz'.format(str(scene_image['sceneID']))))
                        x += 1
                    except RemoteFileDoesntExist:
                        print 'Skipping:', (str(scene_image['sceneID']))

        else:
            print 'nothing'

    print 'done'


if __name__ == '__main__':
    home = os.path.expanduser('~')
    print 'home: {}'.format(home)
    start = datetime(2013, 5, 1).strftime('%Y-%m-%d')
    end = datetime(2013, 9, 30).strftime('%Y-%m-%d')
    output = os.path.join(home, 'images', 'Landsat_8')
    flux_sites = os.path.join(home, 'images', 'vector_data', 'MT_SPCS_vector', 'amf_mt_SPCS.shp')
    poly = os.path.join(home, 'images', 'vector_data', 'MT_SPCS_vector', 'MT_row_paths.shp')
    lat, lon = 44.91, -106.55
    path_int, row_int = 36, 25
    download_landsat((start, end), shape=poly, seek_multipath=True, dry_run=True,
                     output_path=output, max_cloud=70, multipath_points=flux_sites)

    # ===============================================================================
