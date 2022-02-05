import os

import xarray
import fiona
from pandas import date_range, to_datetime, Series
from shapely.geometry import shape
from pyproj import Proj

from thredds import GridMet
from bounds import GeoBounds


def gridmet_subset_stack(extent_shp, start, end, variable, filename):
    with fiona.open(extent_shp, 'r') as src:
        for f in src:
            polygon = shape(f['geometry'])
            w, s, e, n = (x for x in polygon.bounds)
            bounds = GeoBounds(w, s, e, n).to_geographic(5071)
    dt_range = date_range(start, end)
    years = [x for x in range(dt_range[0].year, dt_range[-1].year + 1)]
    sets = []
    for y in years:
        dates = [x for x in dt_range if x.year == y]
        s, e = dates[0], dates[-1]
        print('gridmet {} downloading {} to {}'.format(variable, s, e))
        gridmet = GridMet(variable=variable, start=s, end=e, bbox=bounds,
                          clip_feature=polygon)

        ds = gridmet.subset_nc(return_array=True)
        sets.append(ds)
    xarray.concat(sets, dim='time').to_netcdf(filename)


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
    d = '/home/dgketchum/Downloads/uyws'
    basin = os.path.join(d, 'uyws_basin.shp')
    target_points = os.path.join(d, 'uyws_fake_stations.shp')
    var = 'pr'
    out_ = os.path.join(d, 'gridmet_{}_{}_{}.nc'.format(var, '1991', '2020'))
    gridmet_subset_stack(basin, '1991-01-01', '2020-12-13', var, out_)
    nc_point_extract(target_points, out_, d)
# ========================= EOF ====================================================================
