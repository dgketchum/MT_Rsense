import os

import xarray
import fiona
from pandas import date_range, to_datetime, Series
from shapely.geometry import shape
from pyproj import Proj
import rasterio

from utils.thredds import GridMet, TopoWX
from utils.bounds import GeoBounds


def mt_county_gridmet_stacks(shapes_dir, out_dir, start, end):
    vars_ = ['pr', 'pet', 'tmmn', 'tmmx']
    shps = [os.path.join(shapes_dir, x) for x in os.listdir(shapes_dir) if x.endswith('.shp')]
    for shp_ in shps:
        for var in vars_:
            name_ = os.path.basename(shp_).split('.')[0]
            out_ = os.path.join(out_dir, '{}_{}_{}_{}.nc'.format(var, start[:4], end[:4], name_))
            if os.path.exists(out_):
                continue
            gridmet_subset_stack(shp_, '1997-01-01', '2006-12-13', var, out_, epsg=None)


def gridmet_subset_stack(extent, start, end, variable, filename, epsg, template_raster):
    with rasterio.open(template_raster, 'r') as ras:
        profile = ras.profile
    with fiona.open(extent, 'r') as src:
        for f in src:
            polygon = shape(f['geometry'])
            w, s, e, n = (x for x in polygon.bounds)
            bounds = GeoBounds(w, s, e, n)
            if epsg:
                bounds = bounds.to_geographic(epsg)
    gridmet = GridMet(variable=variable, start=start, end=end, bbox=bounds,
                      clip_feature=polygon)
    ds = gridmet.full_array(start, end)
    ds.to_netcdf(filename)
    print(filename)


def nc_point_extract(points, nc, out_dir):
    ds = xarray.open_dataset(nc)
    with fiona.open(points, 'r') as src:
        for f in src:
            name = int(f['properties']['id'])
            coords = f['geometry']['coordinates']
            in_proj = Proj(src.crs['init'])
            geo_coords = in_proj(coords[0], coords[1], inverse=True)
            point_data = ds.sel(lon=geo_coords[0], lat=geo_coords[1], method='nearest')['precipitation_amount'].values
            time_ = ds['time'].values
            filename = os.path.join(out_dir, 'gridmet_pr_{}.csv'.format(name))
            Series(point_data, index=to_datetime(time_), name='precip').to_csv(filename)
            print(filename)


if __name__ == '__main__':
    d = '/home/dgketchum/IrrigationGIS'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS'

    shapes = os.path.join(d, 'boundaries', 'tl_2017_us_state.shp')
    rasters = os.path.join(d, 'climate_data', 'topowx')
    template = '/home/dgketchum/IrrigationGIS/climate_data/gridmet_elev.tif'

    for y in range(1990, 2023):
        r = os.path.join(rasters, 'tmmn_{}.tif'.format(y))
        gridmet_subset_stack(shapes, '{}-01-01'.format(y), '{}-12-31'.format(y),
                             'tmmn', filename=rasters, epsg=4326, template_raster=template)
# ========================= EOF ====================================================================
