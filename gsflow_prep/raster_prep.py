import os

import fiona
import rasterio
from rasterio.mask import mask

from shapely.geometry import shape, mapping

PROJ4 = '+proj=aea +lat_0=23 +lon_0=-96 +lat_1=29.5 +lat_2=45.5 +x_0=0 +y_0=0 +ellps=GRS80' \
        ' +towgs84=0,0,0,0,0,0,0 +units=m +no_defs'


def clip_raster(basin, raw_data_dir, clipped_dir, buffer_extent=None):
    with fiona.open(basin, 'r') as basn:
        geo = [f['geometry'] for f in basn]
        if buffer_extent:
            geo = [mapping(shape(g).buffer(buffer_extent)) for g in geo]

    scaled = ['sand', 'clay', 'loam', 'awc', 'ksat']
    not_scaled = ['landfire_cover', 'landfire_type', 'nlcd', 'elevation']
    layers = scaled + not_scaled

    _files = [os.path.join(raw_data_dir, x) for x in os.listdir(raw_data_dir) if x.endswith('.tif')]

    for in_ras in _files:
        _var = os.path.basename(in_ras).split('.')[0]
        if _var not in layers:
            print(_var, ' not in {}'.format(layers))
            continue

        out_file = os.path.join(clipped_dir, '{}.tif'.format(_var))

        with rasterio.open(in_ras) as src:
            out_image, out_transform = rasterio.mask.mask(src, geo, crop=True)
            out_meta = src.meta

        out_meta.update({'driver': 'GTiff',
                         'height': out_image.shape[1],
                         'width': out_image.shape[2],
                         'transform': out_transform})

        if _var in scaled:
            out_image = out_image / 100.
            out_meta['dtype'] = 'float32'

        if _var == 'elevation':
            out_meta['dtype'] = 'float32'

        with rasterio.open(out_file, 'w', **out_meta) as dest:
            dest.write(out_image)
            print('wrote {}'.format(out_file))


if __name__ == '__main__':
    d = '/media/research/IrrigationGIS/Montana'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS/Montana'
    src = os.path.join(d, 'statewide_rasters')
    project = os.path.join(d, 'upper_yellowstone', 'gsflow_prep')
    out_rasters = os.path.join(project, 'rasters')
    basin_ = os.path.join(project, 'uyws_carter', 'domain', 'uyws_basin.shp')
    clip_raster(basin_, src, out_rasters, buffer_extent=5000)
# ========================= EOF ======================= =============================================
