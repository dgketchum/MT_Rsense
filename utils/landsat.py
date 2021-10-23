import os
import time
import multiprocessing
import psutil
import numpy as np
import xarray as xr
import rioxarray
from pandas import date_range

from dask.distributed import Client, LocalCluster
from dask.diagnostics import ProgressBar
from dask import config
from dask.array import ma
from dask.array import count_nonzero, ones_like, array, broadcast_to
from pandas import to_datetime

MEM = psutil.virtual_memory().total / 1e9
CORES = multiprocessing.cpu_count()
print(CORES, 'cores', MEM, 'memory')
config.set({'array.slicing.split_large_chunks': True})


def read_raster(t, d):
    da = rioxarray.open_rasterio(t, parse_coordinates=True)
    da = da.squeeze()
    da = da.drop_vars('band')
    da['time'] = to_datetime(d)
    return da


def get_landsat_etf_time_series(tif_dir, out_tif, chunk='auto', year=2017):
    cluster = LocalCluster(memory_limit='{}GB'.format(MEM - 5), processes=False)
    client = Client(cluster)
    l = [os.path.join(tif_dir, x) for x in os.listdir(tif_dir) if x.endswith('.tif') and '_{}'.format(year) in x]
    d = [x.split('.')[0][-8:] for x in l]
    dstr = ['{}-{}-{}'.format(x[:4], x[4:6], x[-2:]) for x in d]
    srt = sorted([(x, y) for x, y in zip(l, dstr)], key=lambda x: x[1])
    names = [os.path.basename(x[0].split('.')[0]) for x in srt]

    def read_rasters(names, file_list):
        da = None
        first = True
        for name, (tif, date) in zip(names, file_list):
            if first:
                da = read_raster(tif, date)
                first = False
            else:
                d = read_raster(tif, date)
                da = xr.concat([da, d], dim='time')
        return da

    def resample_time_sereies(arr):
        arr = arr.chunk(chunks={'time': -1, 'x': chunk, 'y': chunk})
        count = count_nonzero(arr, axis=0)
        arr = arr.astype(float)
        arr = arr.where(arr > 0)
        arr = arr / 100.
        arr = arr.where(count > 1., 0.)

        dates_ = [to_datetime(x) for x in dstr]
        idx = date_range(dates_[0], dates_[-1])
        dt_strings = ['{}-{}-{}'.format(idx[x].year, idx[x].month, idx[x].day) for x, _ in enumerate(idx)]
        arr = arr.reindex({'time': idx})
        arr = arr.chunk(chunks={'time': -1, 'x': chunk, 'y': chunk})

        arr = arr.interpolate_na(dim='time', method='linear')
        arr = arr.interpolate_na(dim='time', method='nearest', fill_value='extrapolate')
        arr = arr.assign_attrs({'long_name': dt_strings})

        return arr

    def get_doy_index(arr):
        dates_ = [to_datetime(x) for x in dstr]
        idx = date_range(dates_[0], dates_[-1])
        idx = [x.dayofyear for x in idx]
        broadcast = broadcast_to(array(idx), ones_like(arr).T.shape).T
        seq = broadcast * ones_like(arr)
        doy_idx = arr.where(arr > 0.5, 0)
        doy_idx = arr.where(doy_idx < 0.5, seq)
        doy_idx = doy_idx.max(axis=0)
        return doy_idx

    da = read_rasters(names, srt)
    da = da[:, 2455:2747, 5161:5363]

    da = resample_time_sereies(da)
    doy = get_doy_index(da)

    with ProgressBar():
        # doy.rio.to_raster(out_tif.replace('.tif', '_doy.tif'))
        da.rio.to_raster(out_tif)


if __name__ == '__main__':
    yr_ = 2016

    root = '/media/research/IrrigationGIS/Montana/water_rights/landsat/etf'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/Montana/water_rights/landsat/etf'

    tif = os.path.join(root, 'masked')
    tif_o = os.path.join(root, 'merged', 'etf_{}_test_mask_.tif'.format(yr_))
    t = time.process_time()
    get_landsat_etf_time_series(tif, tif_o, chunk='auto', year=yr_)
    elapsed_time = time.process_time() - t
    print(elapsed_time)
# ========================= EOF ====================================================================
