import os
import time
import multiprocessing
import psutil

import numpy as np
import numpy.ma as ma
import xarray as xr
from pandas import date_range

from dask import config
from numba import jit
from pandas import to_datetime
import rasterio
from rasterio.merge import merge
from rasterio.windows import Window
from affine import Affine

# MEM = psutil.virtual_memory().total / 1e9
# CORES = multiprocessing.cpu_count()
# print(CORES, 'cores', MEM, 'memory')
config.set({'array.slicing.split_large_chunks': True})


def read_raster(t, c, return_win=False):
    win = Window(row_off=c[0], col_off=c[1], height=c[2], width=c[3])
    with rasterio.open(t, 'r') as src:
        a = src.read(1, window=win)
        window = src.window_transform(win)
    a = a[np.newaxis, :, :]
    if return_win:
        return a, window
    return a


def interp(a):
    def pad(data):
        if np.count_nonzero(data) == 0:
            return data
        good = np.nonzero(data)[0]
        x = np.arange(data.shape[0])
        fp = data[good]
        _interp = np.interp(x, good, fp).astype(np.uint8)
        return _interp

    a = np.apply_along_axis(pad, 0, a)
    return a


def get_landsat_etf_time_series(tif_dir, out_tif, chunk=1000, year=2017):
    l = [os.path.join(tif_dir, x) for x in os.listdir(tif_dir) if x.endswith('.tif') and '_{}'.format(year) in x]
    srt = sorted([x for x in l], key=lambda x: int(x.split('.')[0][-4:]))
    d = [x.split('.')[0][-8:] for x in srt]
    d_numeric = [int(x) for x in d]
    dstr = ['{}-{}-{}'.format(x[:4], x[4:6], x[-2:]) for x in d]

    def read_rasters(file_list, idx):
        da = None
        first = True
        for tif in file_list:
            if first:
                da, w = read_raster(tif, idx, return_win=True)
                first = False
            else:
                d = read_raster(tif, idx)
                da = np.append(da, d, axis=0)
        return da, w

    def resample_time_sereies(a):
        count_2d = np.count_nonzero(a, axis=0)
        count_stack = np.broadcast_to(count_2d, a.shape)
        a = np.where(a > 0, a, 0)
        a = np.where(count_stack > 1, a, 0)
        dt_range = date_range(dstr[0], dstr[-1])
        d_ints = [int('{}{}{}'.format(d.year, str(d.month).rjust(2, '0'), str(d.day).rjust(2, '0'))) for d in dt_range]
        resamp = np.zeros((len(d_ints), a.shape[1], a.shape[2]), dtype=float)
        for i, x in enumerate(d_ints):
            if x in d_numeric:
                unsamp_idx = d_numeric.index(x)
                resamp[i, :, :] = a[unsamp_idx, :, :]
        resamp = interp(resamp)
        return resamp, dt_range

    def get_doy_index(d):
        dates_ = [to_datetime(x) for x in dstr]
        dt_idx = date_range(dates_[0], dates_[-1])
        idx = [x.dayofyear for x in dt_idx]
        broadcast = np.broadcast_to(np.array(idx), np.ones_like(d).T.shape).T
        seq = broadcast * np.ones_like(d)
        doy_idx = np.where(d > 50, seq, 0)
        doy_idx = doy_idx.max(axis=0)
        return doy_idx

    with rasterio.open(l[0], 'r') as src:
        meta = src.meta

    h, w = meta['height'], meta['width']

    t = [((x, y), (chunk, chunk)) for x in range(0, h, chunk) for y in range(0, w, chunk)]
    tiffs, doy_tiffs = [], []

    for elem, c in enumerate(t[32:]):

        ovr = (3000, 5000, 300, 300), (1800, 4000, 300, 300)
        c = ovr[elem]
        print('raster part {}'.format(c))

        da, w = read_rasters(l, c)
        da, dt = resample_time_sereies(da)
        doy = get_doy_index(da)

        if np.count_nonzero(da) == 0:
            print('all zero')
            continue

        meta['height'] = c[2]
        meta['width'] = c[3]
        meta['count'] = len(dt)
        meta['transform'] = w
        meta['dtype'] = 'uint8'
        meta['nodata'] = 0

        file_name = os.path.join(out_tif, '{}_{}.tif'.format(year, elem))
        with rasterio.open(file_name, 'w', **meta) as dst:
            dst.descrtiptions = tuple(dstr)
            for b in range(da.shape[0]):
                dst.write(da[b, :, :], indexes=b + 1)

        meta['count'] = 1
        meta['dtype'] = 'uint16'

        doy_pth = file_name.replace('.tif', '_doy.tif')
        with rasterio.open(doy_pth, 'w', **meta) as dst:
            dst.write(doy, 1)

        tiffs.append(file_name)
        doy_tiffs.append(doy_pth)

        if len(tiffs) == 2:
            break

    t_src, d_src = [], []
    for t, d in zip(tiffs, doy_tiffs):
        ts = rasterio.open(t)
        t_src.append(ts)
        td = rasterio.open(d)
        d_src.append(td)

    mosaic_t, out_trans_t = merge(t_src)
    mosaic_d, out_trans_d = merge(d_src)

    out_meta_t = ts.meta.copy()
    out_meta_d = td.meta.copy()

    out_meta_t.update({'height': mosaic_t.shape[1],
                       'width': mosaic_t.shape[2],
                       'transform': out_trans_t})

    out_meta_d.update({'height': mosaic_d.shape[1],
                       'width': mosaic_d.shape[2],
                       'count': 1,
                       'transform': out_trans_d})

    out_t = os.path.join(out_tif, '{}.tif'.format(year))
    with rasterio.open(out_t, 'w', **out_meta_t) as dst:
        dst.write(mosaic_t)

    out_d = os.path.join(out_tif, '{}_doy.tif'.format(year))
    with rasterio.open(out_d, 'w', **out_meta_d) as dst:
        dst.write(mosaic_d)

    delete = tiffs + doy_tiffs
    [os.remove(x) for x in delete]


if __name__ == '__main__':
    yr_ = 2016

    root = '/media/research/IrrigationGIS/Montana/water_rights/landsat/etf'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/Montana/water_rights/landsat/etf'

    tif = os.path.join(root, 'masked')
    tif_o = os.path.join(root, 'merged')
    t = time.process_time()
    get_landsat_etf_time_series(tif, tif_o, chunk=1000, year=yr_)
    elapsed_time = time.process_time() - t
    print(elapsed_time)
# ========================= EOF ====================================================================
