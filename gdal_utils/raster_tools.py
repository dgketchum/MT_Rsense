# ===============================================================================
# Copyright 2016 dgketchum
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
"""
The purpose of this module is to provide some simple tools needed for raster processing.


"""
from osgeo import gdal, ogr
from numpy import array, asarray
from numpy.ma import masked_where, nomask
from datetime import datetime
import os


def get_raster_polygon(raster):
    tile_id = os.path.basename(raster)
    # print 'tile number: {}'.format(tile_id)
    # print 'get poly tile: {}'.format(tile_id)
    # get raster geometry
    tile = gdal.Open(raster)
    # print 'tile is type: {}'.format(tile)
    transform = tile.GetGeoTransform()
    pixel_width = transform[1]
    pixel_height = transform[5]
    cols = tile.RasterXSize
    rows = tile.RasterYSize

    x_left = transform[0]
    y_top = transform[3]
    x_right = x_left + cols * pixel_width
    y_bottom = y_top - rows * pixel_height

    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(x_left, y_top)
    ring.AddPoint(x_left, y_bottom)
    ring.AddPoint(x_right, y_top)
    ring.AddPoint(x_right, y_bottom)
    ring.AddPoint(x_left, y_top)
    raster_geo = ogr.Geometry(ogr.wkbPolygon)
    raster_geo.AddGeometry(ring)
    # print 'found poly tile geo: {}'.format(raster_geo)
    return raster_geo


def find_poly_ras_intersect(shape, raster_dir, extension='.tif'):
    """  Finds all the tiles falling within raster object
    the get shape geometry should be seperated from the intesect check,
    currently causes a exit code 139 on unix box

    :param polygon:
    :param extension:
    :param raster_dir:
    """

    print 'starting shape: {}'.format(shape)

    # get vector geometry
    polygon = ogr.Open(shape)
    layer = polygon.GetLayer()
    feature = layer.GetFeature(0)
    vector_geo = feature.GetGeometryRef()
    print 'vector geometry: {}'.format(vector_geo)

    tiles = [os.path.join(raster_dir, x) for x in
             os.listdir(os.path.join(raster_dir)) if x.endswith(extension)]

    raster_list = []
    for tile in tiles:
        raster_geo = get_raster_polygon(tile)
        if raster_geo.Intersect(vector_geo):
            print 'tile: {} intersects {}'.format(os.path.basename(tile), os.path.basename(shape))
            raster_list.append(tile)

    return raster_list


def convert_raster_to_array(input_raster_path, raster=None, band=1):
    """
    Convert .tif raster into a numpy numerical array.

    :param input_raster_path: Path to raster.
    :param raster: Raster name with *.tif
    :param band: Band of raster sought.
    :return: Numpy array.
    """
    try:
        raster_open = gdal.Open(os.path.join(input_raster_path, raster))
    except TypeError:
        raster_open = gdal.Open(input_raster_path)
    ras = array(raster_open.GetRasterBand(band).ReadAsArray(), dtype=float)
    return ras


def get_raster_geo_attributes(root):
    """
    Creates a dict of geographic attributes from any of the pre-processed standardized rasters.

    :param root: Path to a folder with pre-processed standardized rasters.
    :return: dict of geographic attributes.
    """
    # statics = [filename for filename in os.listdir(statics_path) if filename.endswith('.tif')]
    # file_name = statics[0]
    file_name = next((fn for fn in os.listdir(root) if fn.endswith('.tif')), None)
    dataset = gdal.Open(os.path.join(root, file_name))

    band = dataset.GetRasterBand(1)
    raster_geo_dict = {'cols': dataset.RasterXSize, 'rows': dataset.RasterYSize, 'bands': dataset.RasterCount,
                       'data_type': band.DataType, 'projection': dataset.GetProjection(),
                       'geotransform': dataset.GetGeoTransform(), 'resolution': dataset.GetGeoTransform()[1]}
    return raster_geo_dict


def apply_mask(mask_path, arr):
    out = None
    file_name = next((fn for fn in os.listdir(mask_path) if fn.endswith('.tif')), None)
    if file_name is not None:
        mask = convert_raster_to_array(mask_path, file_name)
        idxs = asarray(mask, dtype=bool)
        out = arr[idxs].flatten()
    return out


def remake_array(mask_path, arr):
    out = None
    file_name = next((filename for filename in os.listdir(mask_path) if filename.endswith('.tif')), None)
    if file_name is not None:
        mask_array = convert_raster_to_array(mask_path, file_name)
        masked_arr = masked_where(mask_array == 0, mask_array)
        masked_arr[~masked_arr.mask] = arr.ravel()
        masked_arr.mask = nomask
        arr = masked_arr.filled(0)
        out = arr

    return out


def save_daily_pts(wfile, day, data):
    """
        data =x_cord, y_cord, ndvi, temp, precip, etr, petr, nlcd, dem, slope, aspect
    """

    year = day.strftime('%Y')
    this_month = day.strftime('%m')
    month_day = day.strftime('%d')
    print('Saving daily NDVI data for {}-{}-{}'.format(year, this_month, month_day))
    for row in data:
        timestamp = '{},{},{}'.format(year, this_month, month_day)
        # use map(str, row) to convert each element from a float to a str
        row = ','.join(map(str, row))
        wfile.write('{},{}\n'.format(timestamp, row))

    # in case you are worried about data not getting written to file if the script crashes mid-execution
    wfile.flush()
    # comment this out if you are not worried. The only reason i thought you might be is because of the use of 'a' to
    # open a file


def save_daily_pts_old(filename, day, ndvi, temp, precip, etr, petr, nlcd, dem, slope, aspect):
    year = day.strftime('%Y')
    this_month = day.strftime('%m')
    month_day = day.strftime('%d')
    print('Saving daily NDVI data for {}-{}-{}'.format(year, this_month, month_day))
    for a, b, c, d, e, f, g, h, i in zip(ndvi, temp, precip, etr, petr, nlcd, dem, slope, aspect):
        with open(filename, "a") as wfile:
            wfile.write(
                '{},{},{},{},{},{},{},{},{},{},{},{} \n'.format(year, this_month, month_day, a, b, c, d, e, f, g, h, i))


# def save_daily_pts(filename, day, ndvi, temp, precip, etr, petr, nlcd, dem, slope, aspect):
#     year = day.strftime('%Y')
#     this_month = day.strftime('%m')
#     month_day = day.strftime('%d')
#     for datum in zip(ndvi, temp, precip, etr, petr, nlcd, dem, slope, aspect):
#         with open(filename, "a") as wfile:
#             wfile.write('{},{},{},{],{},{},{},{},{},{},{},{}'.format(year,this_month,month_day,*datum))


def array_to_raster(save_array, out_path, geo):
    key = None
    pass
    driver = gdal.GetDriverByName('GTiff')
    out_data_set = driver.Create(out_path, geo['cols'], geo['rows'],
                                 geo['bands'], geo['data_type'])
    out_data_set.SetGeoTransform(geo['geotransform'])
    out_data_set.SetProjection(geo['projection'])
    output_band = out_data_set.GetRasterBand(1)
    output_band.WriteArray(save_array, 0, 0)
    print 'written array {} mean value: {}'.format(key, save_array.mean())

    return None


def make_results_dir(out_root, shapes):
    """
    Creates a directory tree of empty folders that will recieve ETRM model output rasters.

    :param out_root:
    :param shapes: Folder contains sub-directories with shapefiles of geographies to be analyzed.
    :return: dict of directory paths
    """

    empties = ('annual_rasters', 'monthly_rasters', 'simulation_tot_rasters', 'annual_tabulated',
               'monthly_tabulated', 'daily_tabulated', 'daily_rasters')
    now = datetime.now()
    tag = now.strftime('%Y_%m_%d')

    out_root = os.path.join(out_root, 'ETRM_Results_{}'.format(tag))

    results_directories = {'root': out_root}

    if not os.path.isdir(out_root):
        os.makedirs(out_root)
        for item in empties:
            empty = os.path.join(out_root, item)
            os.makedirs(empty)
            results_directories[item] = empty

    else:
        results_directories = {item: os.path.join(out_root, item) for item in empties}

    region_types = os.listdir(shapes)
    for tab_folder in ('annual_tabulated', 'monthly_tabulated', 'daily_tabulated'):
        d = {}
        for region_type in region_types:
            a, b = region_type.split('_P')
            dst = os.path.join(out_root, tab_folder, a)
            os.makedirs(dst)
            d[a] = dst

    results_directories[tab_folder] = d

    print 'results dirs: \n{}'.format(results_directories)
    return results_directories


if __name__ == '__main__':
    home = os.path.expanduser('~')
    print 'home: {}'.format(home)
    terrain = os.path.join(home, 'images', 'terrain', 'ned_tiles', 'dem')
    shape = os.path.join(home, 'images', 'vector_data', 'Flux_locations', '37027_L8_Z12.shp')
    find_poly_ras_intersect(shape, terrain)

# =================================== EOF =========================
