import os
from subprocess import check_call

import fiona
from shapely.geometry import shape

PROJ4 = '+proj=aea +lat_0=23 +lon_0=-96 +lat_1=29.5 +lat_2=45.5 +x_0=0 +y_0=0 +ellps=GRS80' \
        ' +towgs84=0,0,0,0,0,0,0 +units=m +no_defs'
warp = '/home/dgketchum/miniconda3/envs/opnt/bin/gdalwarp'


def clip_raster(basin, raw_data_dir, clipped_dir, buffer_extent=None):
    with fiona.open(basin, 'r') as basn:
        geo = [f['geometry'] for f in basn][0]
        geo = shape(geo)
        if buffer_extent:
            geo = geo.buffer(buffer_extent)

    bnd = geo.bounds

    scaled = ['sand', 'clay', 'loam', 'awc', 'ksat']
    not_scaled = ['landfire_cover', 'landfire_type', 'nlcd', 'elevation']
    layers = scaled + not_scaled

    _files = [os.path.join(raw_data_dir, x) for x in os.listdir(raw_data_dir) if x.endswith('.tif')]

    for in_ras in _files:

        _var = os.path.basename(in_ras).split('.')[0]

        if _var not in layers and 'prism' not in in_ras:
            print(_var, ' not in {}'.format(layers))
            continue

        if _var == 'elevation' or _var in scaled:
            resamp = 'average'

        else:
            resamp = 'nearest'

        out_ras = os.path.join(clipped_dir, '{}.tif'.format(_var))

        cmd = [warp, '-of', 'GTiff', '-r', resamp, '-overwrite',
               '-te', str(bnd[0]), str(bnd[1]), str(bnd[2]), str(bnd[3]),
               '-multi', '-wo', '-wo NUM_THREADS=8',
               in_ras, out_ras]

        check_call(cmd)


if __name__ == '__main__':
    d = '/media/research/IrrigationGIS/Montana'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS/Montana'
    src = os.path.join(d, 'statewide_rasters')
    project = os.path.join(d, 'upper_yellowstone', 'gsflow_prep')
    out_rasters = os.path.join(project, 'rasters')
    basin_ = os.path.join(project, 'domain', 'carter_basin.shp')
    clip_raster(basin_, src, out_rasters, buffer_extent=None)
# ========================= EOF ======================= =============================================
