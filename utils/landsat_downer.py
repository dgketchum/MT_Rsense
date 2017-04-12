"""
This module downloads landsat data.  Get the wrs (ascending)
from http://landsat.usgs.gov/worldwide_reference_system_WRS.php
select an area you want images for, save the selection and
pass shapefile to this program,
or just choose location coordinates
"""
import os
from osgeo import ogr
from datetime import datetime

from utils import usgs_download

from vector_tools import get_pr_from_field, get_pr_multipath
from web_tools import lat_lon_wrs2pr_convert


def download_landsat(start_end_tuple, satellite, path_row_tuple=None, lat_lon_tuple=None,
                     shape=None, output_path=None, seek_multipath=False, multipath_points=None,
                     usgs_creds=None):

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
        image_index = [lat_lon_wrs2pr_convert(lat_lon_tuple)]
        print 'Downloading landsat by lat/lon: {}'.format(lat_lon_tuple)

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

        destination_path = os.path.join(output_path, 'd_{}_{}'.format(path, row))

        if not os.path.exists(destination_path):
            os.mkdir(destination_path)

        scenes_list = usgs_download.get_candidate_scenes_list((path, row), satellite, start, end)

        usgs_download.down_usgs_by_list(scenes_list, destination_path, usgs_creds)


if __name__ == '__main__':
    home = os.path.expanduser('~')
    start = datetime(2007, 5, 1).strftime('%Y-%m-%d')
    end = datetime(2007, 5, 30).strftime('%Y-%m-%d')
    start_end = start, end
    satellite = 'LT5'
    output = os.path.join(home, 'images', satellite)
    usgs_creds = os.path.join(home, 'images', 'usgs.txt')
    path_row = 36, 25
    download_landsat((start, end), satellite=satellite.replace('andsat_', ''),
                     path_row_tuple=path_row, output_path=output, usgs_creds=usgs_creds)


# ===============================================================================
