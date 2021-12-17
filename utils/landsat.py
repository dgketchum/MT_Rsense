import os
from subprocess import check_call
import json
from tqdm import tqdm

import numpy as np
from pandas import date_range
from pandas import to_datetime
import rasterio
from rasterio.merge import merge
from rasterio.windows import Window

conda = '/home/dgketchum/miniconda3/envs/metric/bin/'
if not os.path.isdir(conda):
    conda = '/home/dgketchum/miniconda/envs/metric/bin/'
OVR = os.path.join(conda, 'gdaladdo')
TRANSLATE = os.path.join(conda, 'gdal_translate')
PROJ = '+proj=utm +zone=12 +datum=WGS84 +units=m +no_defs'

BASINS = {'12334550': '040028',
          '12340000': '040027',
          '06052500': '039028',
          '06195600': '039028',
          '06076690': '039028'}


def get_landsat_etf_time_series(tif_dir, out_tif, chunk=1000, year=2017, meta_data=None, pr='039028'):
    file_list, int_dates, str_dates, index_ = get_list_info(tif_dir, year, pr=pr, list_=False)

    file_name = os.path.join(out_tif, '{}.tif'.format(year))
    if os.path.exists(file_name):
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

    first = True
    for elem, c in tqdm(enumerate(t), total=len(t)):

        da, w = read_rasters(file_list, c)
        da, dt = resample_time_sereies(da, str_dates, int_dates)
        doy = get_doy_index(da, index_)

        if first and meta_data:
            write_metadata_file(hydro, str_dates, index_, year, out_tif)
            first = False

        if np.count_nonzero(da) == 0:
            continue

        meta['height'] = c[2]
        meta['width'] = c[3]
        meta['count'] = len(dt)
        meta['transform'] = w
        meta['dtype'] = 'uint8'
        meta['nodata'] = 0

        file_name = os.path.join(out_tif, '{}_{}.tif'.format(year, elem))
        with rasterio.open(file_name, 'w', **meta) as dst:
            dst.descrtiptions = tuple(str_dates)
            for b in range(da.shape[0]):
                dst.write(da[b, :, :], indexes=b + 1)

        meta['count'] = 1
        meta['dtype'] = 'uint16'

        doy_pth = file_name.replace('.tif', '_doy.tif')
        with rasterio.open(doy_pth, 'w', **meta) as dst:
            dst.write(doy, 1)

        tiffs.append(file_name)
        doy_tiffs.append(doy_pth)

    merge_rasters(tiffs, doy_tiffs, out_tif, year)


def get_list_info(tif_dir, year, list_=False, pr='039028'):
    """ Pass list in place of tif_dir optionally """
    if list_:
        l = tif_dir
    else:
        l = [os.path.join(tif_dir, x) for x in os.listdir(tif_dir) if
             x.endswith('.tif') and '_{}'.format(year) in x and pr in x]
    srt = sorted([x for x in l], key=lambda x: int(x.split('.')[0][-4:]))
    d = [x.split('.')[0][-8:] for x in srt]
    d_numeric = [int(x) for x in d]
    dstr = ['{}-{}-{}'.format(x[:4], x[4:6], x[-2:]) for x in d]
    dates_ = [to_datetime(x) for x in dstr]
    dt_idx = date_range(dates_[0], dates_[-1])
    idx = [x.dayofyear for x in dt_idx]
    return l, d_numeric, dstr, idx


def merge_rasters(tiffs, doy_tiffs, out_tif, year):
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


def write_metadata_file(hydro, dstr, idx, year, out_tif):
    hydro['sensor_date_range'] = '{} to {}'.format(dstr[0], dstr[-1])
    hydro['doy_range'] = '{} to {}'.format(idx[0], idx[-1])
    dt_idx = [str(x)[:10] for x in date_range(dstr[0], dstr[-1])]
    lf_idx = list(dt_idx).index(hydro['date'])
    hydro['lf_start_on_band'] = lf_idx
    hydro['year'] = str(year)
    meta_out = os.path.join(out_tif, '{}_metadata.txt'.format(year))
    lines = ['{} {}'.format(k, v) for k, v in hydro.items()]
    [print(x) for x in lines]
    with open(meta_out, 'w') as f:
        f.write('\n'.join(lines))
    return None


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


def resample_time_sereies(a, dstr, d_numeric):
    count_2d = np.count_nonzero(a, axis=0)
    count_stack = np.broadcast_to(count_2d, a.shape)
    a = np.where(a > 0, a, 0)
    a = np.where(count_stack > 1, a, 0)
    dt_range = date_range(dstr[0], dstr[-1])
    d_ints = [int('{}{}{}'.format(d.year, str(d.month).rjust(2, '0'), str(d.day).rjust(2, '0'))) for d in dt_range]
    resamp = np.zeros((len(d_ints), a.shape[1], a.shape[2]))
    for i, x in enumerate(d_ints):
        if x in d_numeric:
            unsamp_idx = d_numeric.index(x)
            resamp[i, :, :] = a[unsamp_idx, :, :]
    resamp = interp(resamp).astype(np.uint8)
    return resamp, dt_range


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


def add_raster_overview(t_dir, years, overwrite=False):
    l = [os.path.join(t_dir, '{}.tif'.format(y)) for y in years]
    for tiff in l:
        btiff = tiff.replace('.tif', '_bt.tif')
        if os.path.exists(btiff) and not overwrite:
            continue
        cmd = [TRANSLATE, '-a_srs', PROJ, '-co', 'TILED=YES', '-co', 'BIGTIFF=YES', tiff, btiff]
        print('writing', btiff)
        check_call(cmd)
        print('writing overviews')
        cmd = [OVR, '-r', 'average', btiff]
        check_call(cmd)


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/Montana/water_rights'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/Montana/water_rights'
    basins_ = os.path.join(root, 'gage_basins', 'merged_basins.shp')
    wrs_ = os.path.join(root, 'landsat', 'wrs2', 'mt_wrs2.shp')
    root = os.path.join(root, 'landsat', 'etf')
    tif = os.path.join(root, 'masked')
    for p_r in ['039028', '040027']:
        tif_o = os.path.join(root, 'merged', p_r)
        get_landsat_etf_time_series(tif, tif_o, year=2015, pr=p_r, meta_data=None)
        add_raster_overview(tif_o, [2015])
# ========================= EOF ====================================================================
