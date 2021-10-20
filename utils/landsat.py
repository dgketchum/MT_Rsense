import os
import numpy as np

import rasterio
import requests
import xarray as xr
import rioxarray

import dask
from dask.array import map_blocks
from pandas import DataFrame, to_datetime

QA_VALUES = [322.0, 324.0, 328.0, 336.0, 352.0, 368.0, 386.0, 388.0,
             392.0, 400.0, 416.0, 432.0, 480.0, 834.0, 836.0, 840.0,
             848.0, 864.0, 880.0, 898.0, 904.0, 912.0, 928.0, 944.0,
             992.0]


def read_raster(t, d):
    da = rioxarray.open_rasterio(t, parse_coordinates=True,
                                 chunks={'x': 1000, 'y': 1000})
    da = da.squeeze()
    da = da.drop_vars('band')
    da['time'] = to_datetime(d)
    return da


def get_landsat_etf_time_series(tif_dir, out_tif):
    l = [os.path.join(tif_dir, x) for x in os.listdir(tif_dir) if x.endswith('.tif')]
    d = [x.split('.')[0][-8:] for x in l]
    names = [os.path.basename(x.split('.')[0]) for x in l]
    dstr = ['{}-{}-{}'.format(x[:4], x[4:6], x[-2:]) for x in d]
    srt = sorted([(x, y) for x, y in zip(l, dstr)], key=lambda x: x[1])
    ds = xr.Dataset()
    for name, (tif, date) in zip(names, srt):
        da = read_raster(tif, date)
        ds[name] = da
    ds.rio.to_raster(out_tif)


if __name__ == '__main__':
    tif = '/media/research/IrrigationGIS/Montana/water_rights/landsat/etf'
    tif_o = os.path.join(tif, 'merged', 'concat_xa.tif')
    get_landsat_etf_time_series(tif, tif_o)
# ========================= EOF ====================================================================
