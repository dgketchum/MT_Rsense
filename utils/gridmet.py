import os

import xarray
import fiona
from pandas import date_range, to_datetime, Series
from shapely.geometry import shape
from pyproj import Proj

from thredds import GridMet
from bounds import GeoBounds


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


def gridmet_subset_stack(extent, start, end, variable, filename, epsg):
    with fiona.open(extent, 'r') as src:
        for f in src:
            polygon = shape(f['geometry'])
            w, s, e, n = (x for x in polygon.bounds)
            bounds = GeoBounds(w, s, e, n)
            if epsg:
                bounds = bounds.to_geographic(epsg)
    gridmet = GridMet(variable=variable, start=start, end=end, bbox=bounds,
                      clip_feature=polygon)
    ds = gridmet.subset_nc(return_array=True)
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
    d = '/media/research/IrrigationGIS'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS'

    shapes_co = os.path.join(d, 'boundaries', 'mt_counties', 'mt_counties.shp')
    shapes_dir_out = os.path.join(d, 'boundaries', 'mt_counties', 'individual_shapes')
    nc_dir = os.path.join(d, 'climate', 'gridmet_nc', 'mt_counties')
    mt_county_gridmet_stacks(shapes_dir_out, nc_dir, '1997-01-01', '2006-12-31')
# ========================= EOF ====================================================================
