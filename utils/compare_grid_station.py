import os

import numpy as np
import fiona
from shapely.geometry import shape
from pandas import concat, read_csv, to_datetime
import matplotlib.pyplot as plt

from thredds import GridMet


def compare_gridmet_ghcn(stations, station_data, out_dir):
    sta_names = []
    with fiona.open(stations, 'r') as src:
        for f in src:
            props = f['properties']
            geo = shape(f['geometry'])
            sta_names.append((props['STAID'], geo.y, geo.x))

    for i, sta in enumerate(sta_names):

        lat, lon, staid = sta[1], sta[2], sta[0]

        _file = os.path.join(station_data, '{}.csv'.format(staid))
        sta_df = read_csv(_file, parse_dates=True, infer_datetime_format=True,
                          index_col='DATE')

        if sta_df.index[-1] < to_datetime('1981-01-01'):
            continue

        if sta_df.index[0] < to_datetime('1981-01-01'):
            sta_df = sta_df.loc['1981-01-01':]

        try:
            sta_df = sta_df['TMIN']
            sta_df /= 10.

            if sta_df.empty or sta_df.shape[0] < 1000:
                print(staid, 'insuf records in date range')
                continue

        except KeyError as e:
            print(staid, 'missing', e)
            continue

        grd = GridMet(variable='tmmn', start='1981-01-01', end='2020-12-31',
                      lat=lat, lon=lon)
        grd_df = grd.get_point_timeseries()
        grd_df -= 273.15
        grd_df = grd_df.loc[sta_df.index]
        df = concat([sta_df, grd_df], axis=1)
        df = df.loc['2019-4-01': '2019-5-31']
        df.dropna(how='any', axis=0, inplace=True)
        pass


if __name__ == '__main__':
    d = '/media/research/IrrigationGIS/climate/'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS/climate/'

    stations_ = os.path.join(d, 'ghcn', 'ghcn_stations_bitterroot.shp')
    station_data_ = os.path.join(d, 'ghcn', 'ghcn_daily_summaries_4FEB2022')
    out_dir_ = os.path.join(d, 'ghcn', 'ghcn_gridmet_comp')
    compare_gridmet_ghcn(stations_, station_data_, out_dir_)
# ========================= EOF ====================================================================
