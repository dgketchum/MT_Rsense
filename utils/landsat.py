import os
import time

import numpy as np
from dask.diagnostics import ProgressBar
import xarray as xr
import rioxarray
from dask import config
from pandas import date_range

from dask.array import ma
from dask.array import count_nonzero, ones_like, array, broadcast_to, where
from pandas import to_datetime

config.set({'array.slicing.split_large_chunks': True})


def read_raster(t, d):
    da = rioxarray.open_rasterio(t, parse_coordinates=True)
    da = da.squeeze()
    da = da.drop_vars('band')
    da['time'] = to_datetime(d)
    return da


def get_landsat_etf_time_series(tif_dir, out_tif, year=2017):
    l = [os.path.join(tif_dir, x) for x in os.listdir(tif_dir) if x.endswith('.tif') and '_{}'.format(year) in x]
    d = [x.split('.')[0][-8:] for x in l]
    dstr = ['{}-{}-{}'.format(x[:4], x[4:6], x[-2:]) for x in d]
    srt = sorted([(x, y) for x, y in zip(l, dstr)], key=lambda x: x[1])
    names = [os.path.basename(x[0].split('.')[0]) for x in srt]
    da = None
    first = True
    for name, (tif, date) in zip(names, srt):
        if first:
            da = read_raster(tif, date)
            first = False
        else:
            d = read_raster(tif, date)
            da = xr.concat([da, d], dim='time')

    da = da[:, 2455:2747, 5161:5363]

    count = count_nonzero(da, axis=0)
    da = da.astype(float)
    da = da.where(da > 0)
    da = da / 100.

    dates_ = [to_datetime(x) for x in da.time.values]
    idx = date_range(dates_[0], dates_[-1])
    dt_strings = ['{}-{}-{}'.format(idx[x].year, idx[x].month, idx[x].day) for x, _ in enumerate(idx)]
    da = da.reindex({'time': idx})

    da = da.chunk(chunks={'time': -1, 'x': 60, 'y': 60})
    da = da.interpolate_na(dim='time', method='linear')
    da = da.interpolate_na(dim='time', method='nearest', fill_value='extrapolate')

    idx = [x.dayofyear for x in idx]
    broadcast = broadcast_to(array(idx), ones_like(da).T.shape).T
    seq = broadcast * ones_like(da)

    count = broadcast_to(count, da.shape)
    da_m = ma.masked_where(count == 0, da)

    doy = da.where(da > 0.5, 0)
    doy = da.where(doy < 0.5, seq)

    with ProgressBar():
        doy = doy.max(axis=0).compute()
        da = da.where(da_m, da).compute()

    da = da.assign_attrs({'long_name': dt_strings})
    doy.rio.to_raster(out_tif.replace('.tif', '_doy.tif'))
    da.rio.to_raster(out_tif)


if __name__ == '__main__':
    yr_ = 2016
    root = '/media/research/IrrigationGIS/Montana/water_rights/landsat/etf'
    tif = os.path.join(root, 'masked')
    tif_o = os.path.join(root, 'merged', 'etf_{}_test_.tif'.format(yr_))
    t = time.process_time()
    get_landsat_etf_time_series(tif, tif_o, year=yr_)
    elapsed_time = time.process_time() - t
    print(elapsed_time)
# ========================= EOF ====================================================================
