import os
from subprocess import check_call
import json
from tqdm import tqdm

import numpy as np
import pandas as pd
import rasterio
from rasterio.merge import merge
from rasterio.windows import Window

conda = '/home/dgketchum/miniconda3/envs/opnt/bin/'
if not os.path.isdir(conda):
    conda = '/home/dgketchum/miniconda/envs/opnt/bin/'
OVR = os.path.join(conda, 'gdaladdo')
TRANSLATE = os.path.join(conda, 'gdal_translate')
PROJ = '+proj=aea +lat_0=23 +lon_0=-96 +lat_1=29.5 +lat_2=45.5 +x_0=0 +y_0=0 ' \
       '+ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs'

BASINS = {'12334550': '040028',
          '12340000': '040027',
          '06052500': '039028',
          '06195600': '039028',
          '06076690': '039028'}


def get_landsat_ndvi_time_series(tif_dir, out_tif, chunk=1000, year=2017,
                                 meta_data=None, overwrite=False, threshold=600):
    file_list, int_dates, doy = get_list_info(tif_dir, year, list_=False)

    file_name = os.path.join(out_tif, '{}.tif'.format(year))
    if os.path.exists(file_name) and not overwrite:
        print('{} exists already'.format(file_name))
        return None

    if meta_data:
        with open(meta_data, 'r') as f:
            meta_d = json.load(f)
            hydro = meta_d[str(year)]

    with rasterio.open(file_list[0], 'r') as src:
        meta = src.meta

    h, w = meta['height'], meta['width']

    t = [(x, y, chunk, chunk) for x in range(0, h, chunk) for y in range(0, w, chunk)]
    tiffs, doy_tiffs = [], []

    dtstr_keys = {i: k for i, k in enumerate(int_dates)}
    doy_keys = {i: k for i, k in enumerate(doy)}

    for elem, c in tqdm(enumerate(t), total=len(t)):

        # if elem != 12:
        #     continue

        file_name = os.path.join(out_tif, '{}_{}.tif'.format(year, elem))
        doy_pth = file_name.replace('.tif', '_doy.tif')
        file_check = [os.path.exists(file_name), os.path.exists(doy_pth), not overwrite]
        if all(file_check):
            print('{} and {} exist, skipping'.format(os.path.basename(file_name),
                                                     os.path.basename(doy_pth)))
            continue

        # peak-finding algorithm
        arr, w = read_rasters(file_list, c)
        arr = arr.astype(float)
        arr[arr == 0] = np.nan
        max_, mean_ = np.nanmax(arr, axis=0), np.nanmean(arr, axis=0)
        ct = np.count_nonzero(arr, axis=0)
        arr[0] = np.where(np.isnan(arr[0]), np.nanmin(arr, axis=0), arr[0])
        arr[np.isnan(arr)] = 0
        arr = interp(arr, over='zero')
        da = np.diff(arr, axis=0)
        zero = np.zeros((da.shape[0] + 1, da.shape[1], da.shape[2]))
        thresh = np.ones_like(arr) * threshold
        forward, back = np.append(da, zero[0:1], axis=0), np.append(zero[0:1], da, axis=0)
        peaks = ((back > zero) & (forward < zero) & (arr > thresh))
        peaks = peaks.flatten()
        idx = np.broadcast_to(np.arange(arr.shape[0]), np.ones_like(arr).T.shape).T.flatten()
        idx[~peaks] = 0
        idx = idx.reshape(arr.shape).max(axis=0)
        dtstr = np.vectorize(dtstr_keys.get)(idx)
        dtstr[idx == 0] = 0
        doy = np.vectorize(doy_keys.get)(idx)
        doy[idx == 0] = 0

        if np.count_nonzero(arr) == 0:
            continue

        meta['height'] = c[2]
        meta['width'] = c[3]
        meta['count'] = 5
        meta['transform'] = w
        meta['dtype'] = 'uint32'
        meta['nodata'] = 0

        description = ['date', 'doy', 'count', 'max', 'mean']
        data = [dtstr, doy, ct, max_, mean_]
        with rasterio.open(file_name, 'w', **meta) as dst:
            for i, (dat, desc) in enumerate(zip(data, description)):
                dst.write(dat, indexes=i + 1)
                dst.set_band_description(i + 1, desc)

        tiffs.append(file_name)

    t_src = []
    for t in tiffs:
        ts = rasterio.open(t)
        t_src.append(ts)

    mosaic_t, out_trans_t = merge(t_src)

    out_meta_t = ts.meta.copy()

    out_meta_t.update({'height': mosaic_t.shape[1],
                       'width': mosaic_t.shape[2],
                       'transform': out_trans_t})

    out_t = os.path.join(out_tif, '{}.tif'.format(year))
    with rasterio.open(out_t, 'w', **out_meta_t) as dst:
        dst.write(mosaic_t)

    [os.remove(x) for x in tiffs]


def get_list_info(tif_dir, year, list_=False):
    """ Pass list in place of tif_dir optionally """
    if list_:
        l = tif_dir
    else:
        l = [os.path.join(tif_dir, x) for x in os.listdir(tif_dir) if
             x.endswith('.tif') and '_{}'.format(year) in x]
    srt = sorted([x for x in l], key=lambda x: int(x.split('.')[0][-4:]))
    d = [x.split('.')[0][-8:] for x in srt]
    d_numeric = [int(x) for x in d]
    dstr = ['{}-{}-{}'.format(x[:4], x[4:6], x[-2:]) for x in d]
    dates_ = [pd.to_datetime(x) for x in dstr]
    doy = [int(dt.strftime('%j')) for dt in dates_]
    # TODO: rewrite sorting, this might fail with multiple path/rows
    return l, d_numeric, doy


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


def get_doy_index(d, idx):
    broadcast = np.broadcast_to(np.array(idx), np.ones_like(d).T.shape).T
    seq = broadcast * np.ones_like(d)
    doy_idx = np.where(d > 50, seq, 0)
    doy_idx = doy_idx.max(axis=0).astype(np.uint16)
    return doy_idx


def read_raster(t, c, return_win=False):
    win = Window(row_off=c[0], col_off=c[1], height=c[2], width=c[3])
    with rasterio.open(t, 'r') as src:
        a = src.read(1, window=win)
        window = src.window_transform(win)
    a = a[np.newaxis, :, :]
    if return_win:
        return a, window
    return a


def interp(a, over='nan'):
    def pad(data):
        if np.count_nonzero(data) == 0:
            return data
        if over == 'nan':
            good = np.isfinite(data)[0]
        else:
            good = np.nonzero(data)[0]
        x = np.arange(data.shape[0])
        fp = data[good]
        _interp = np.interp(x, good, fp)
        return _interp

    a = np.apply_along_axis(pad, 0, a)
    return a


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/Montana/water_rights'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/Montana/water_rights'

    basin_ = '12334550'
    tif = os.path.join(root, 'landsat', 'ndvi', basin_, 'input')
    otif = os.path.join(root, 'landsat', 'ndvi', basin_, 'merged')
    yr = 2016
    get_landsat_ndvi_time_series(tif, otif, year=yr, meta_data=None, overwrite=True)
# ========================= EOF ====================================================================
