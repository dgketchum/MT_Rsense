import os

import numpy as np
import fiona
from shapely.geometry import shape
from pandas import concat, read_csv, to_datetime, date_range
import matplotlib.pyplot as plt

from thredds import GridMet
from agrimet import Agrimet, load_stations


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


def compare_gridmet_agrimet(station, station_data, out_dir):
    stations = load_stations()
    s_data = stations[station]
    lon, lat = tuple(s_data['geometry']['coordinates'])
    _file = os.path.join(station_data, '{}.csv'.format(station))
    sta_df = read_csv(_file, parse_dates=True, infer_datetime_format=True, header=0,
                      index_col=[0], skiprows=[1, 2])
    sta_df = sta_df.interpolate('linear', limit=5, limit_direction='forward', axis=0)
    sta_df = sta_df.dropna(subset=['ET'])
    sta_df.index = [to_datetime(dt) for dt in sta_df.index]

    sta_df['mday'] = ['{}-{}'.format(x.month, x.day) for x in sta_df.index]
    target_range = date_range('2000-{}-{}'.format(4, 15),
                              '2000-{}-{}'.format(10, 15))
    accept = ['{}-{}'.format(x.month, x.day) for x in target_range]
    sta_df.dropna(subset=['ETOS'], inplace=True)
    sta_df['mask'] = [1 if d in accept else 0 for d in sta_df['mday']]
    sta_df = sta_df[sta_df['mask'] == 1]

    size = sta_df.groupby(sta_df.index.year).size()
    for i, s in size.iteritems():
        if s < len(target_range) - 1:
            sta_df = sta_df.loc[[x for x in sta_df.index if x.year != i]]

    years = list(set([d.year for d in sta_df.index]))
    sta_df = sta_df.loc[[x for x in sta_df.index if x.year in years]]

    grd = GridMet(variable='etr', start='{}-01-01'.format(min(years)),
                  end='{}-12-31'.format(max(years)),
                  lat=lat, lon=lon)
    grd = grd.get_point_timeseries()
    grd = grd.loc[sta_df.index]
    df = concat([sta_df, grd], axis=1)
    df['ratio'] = df['etr'] / df['ETRS']
    months = list(set([x.month for x in df.index]))
    dct = {}
    for m in months:
        mdata = np.array([r['ratio'] for i, r in df.iterrows() if i.month == m]).mean()
        dct[m] = mdata
    print(dct)


if __name__ == '__main__':
    d = '/media/research/IrrigationGIS/climate/'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS/climate/'

    stations_ = os.path.join(d, 'ghcn', 'ghcn_stations_bitterroot.shp')
    station_data_ = os.path.join(d, 'ghcn', 'ghcn_daily_summaries_4FEB2022')
    out_dir_ = os.path.join(d, 'ghcn', 'ghcn_gridmet_comp')
    # compare_gridmet_ghcn(stations_, station_data_, out_dir_)

    station_data_ = os.path.join(d, 'agrimet', 'mt_stations')
    out_dir_ = os.path.join(d, 'ghcn', 'ghcn_gridmet_comp')
    compare_gridmet_agrimet('covm', station_data_, out_dir_)
# ========================= EOF ====================================================================
