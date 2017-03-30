"""
This module downloads landsat data.  Get the wrs (ascending)
from http://landsat.usgs.gov/worldwide_reference_system_WRS.php
select an area you want images for, save the selection and
pass shapefile to this program,
or just choose location coordinates
"""
from landsat.image import Simple
from landsat.downloader import Downloader
from landsat.downloader import RemoteFileDoesntExist
from landsat.search import Search
from os.path import join
from osgeo import ogr
import os
import requests.packages.urllib3

requests.packages.urllib3.disable_warnings()


def download_landsat((start_date, end_date), (path, row)=None, (lat, lon)=None, shape=None, output_path=None):
    if shape:
        shp_filename = 'C:/Recharge_GIS/Landsat_Paths/NM_wrs2_desc.shp'
        ds = ogr.Open(shp_filename)
        lyr = ds.GetLayer()
        image_index = get_pathrow(lyr)
    elif (lat, lon):
        latitude = 32.88205
        longitude = -105.17832
    elif (path, row):
        path = None
        row = None
    else:
        raise NotImplementedError('You must give path/row tuple, lat/lon tuple, or a path/row shapefile!')

    return_scenes = 100
    max_cloud_prcnt = 20

    for tile in image_index:
        path, row = tile[0], tile[1]
        srcher = Search()
        dest_path = 'd_{}_{}'.format(path, row)
        os.chdir(output_path)
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        # if os.listdir(dest_path) == []:
        if os.listdir(dest_path)
            print
        print '{} is empty'.format(dest_path)

        dwner = Downloader(verbose=False, download_dir='{}\\{}'.format(output_path, dest_path))

        candidate_scenes = srcher.search(paths_rows='{},{},{},{}'.format(path, row, path, row),
                                         start_date=start_date,
                                         end_date=end_date,
                                         cloud_min=0,
                                         cloud_max=max_cloud_prcnt,
                                         limit=return_scenes)
        print 'total images for tile {} is {}'.format(tile, candidate_scenes['total_returned'])

        x = 0
        if candidate_scenes['status'] == 'SUCCESS':
            for scene_image in candidate_scenes['results']:
                print 'Downloading:', (str(scene_image['sceneID']))
                print 'Downloading tile {} of {}'.format(x, candidate_scenes['total_returned'])
                try:
                    dwner.download([str(scene_image['sceneID'])])
                    Simple(join('{}\\{}'.format(output_path, dest_path), str(scene_image['sceneID']) + '.tar.bz'))
                    x += 1
                except RemoteFileDoesntExist:
                    print 'Skipping:', (str(scene_image['sceneID']))

        else:
            print 'nothing'

    print 'done'


def get_pathrow(layer):
    path_list = []
    for feat in layer:
        path = str(feat.GetField('PATH'))
        row = str(feat.GetField('ROW'))
        path_list.append((path.rjust(3, '0'), row.rjust(3, '0')))
    print path_list
    print 'number of tiles : {}'.format(len(path_list))
    return path_list
