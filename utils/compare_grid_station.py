import os

import fiona
from shapely.geometry import shape
import pandas as pd

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
        sta_df = pd.read_csv(_file, parse_dates=True, infer_datetime_format=True,
                          index_col='DATE')

        if sta_df.index[-1] < pd.to_datetime('1981-01-01'):
            continue

        if sta_df.index[0] < pd.to_datetime('1981-01-01'):
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
        df = pd.concat([sta_df, grd_df], axis=1)
        df = df.loc['2019-4-01': '2019-5-31']
        df.dropna(how='any', axis=0, inplace=True)
        pass


def compare_gridmet_agrimet(station, station_data, out_dir):

    s, e = '2000-01-01', '2022-12-31'
    stations = load_stations()
    s_data = stations[station]
    lon, lat = tuple(s_data['geometry']['coordinates'])
    _file = os.path.join(station_data, '{}.csv'.format(station))
    ag = Agrimet(start_date=s, end_date=e, station='lmmm')
    ag.region = 'great_plains'
    sta_df = ag.fetch_met_data()
    sta_df = sta_df[[sta_df.columns[0]]]
    sta_df.columns = ['ETr']
    sta_df.dropna(inplace=True, axis=0)
    size = sta_df.groupby(sta_df.index.year).size()
    for yr, sz in size.iteritems():
        yr_len = len(pd.date_range('{}-01-01'.format(yr), '{}-12-31'.format(yr), freq='D'))
        if sz < yr_len:
            sta_df = sta_df.loc[[x for x in sta_df.index if x.year != yr]]

    years = list(set([d.year for d in sta_df.index]))
    sta_df = sta_df.loc[[x for x in sta_df.index if x.year in years]]

    grd = GridMet(variable='etr', start='{}-01-01'.format(min(years)),
                  end='{}-12-31'.format(max(years)),
                  lat=lat, lon=lon)
    grd = grd.get_point_timeseries()
    grd = grd.loc[sta_df.index]
    df = pd.concat([sta_df, grd], axis=1)
    dfm = df.groupby([df.index.year, df.index.month]).sum()
    dfm.index = [pd.to_datetime('{}-{}-01'.format(*i)) for i in dfm.index]
    dfa = dfm.groupby([dfm.index.month]).mean()

    dfa['ratio'] = dfa['ETr'] / dfa['etr']
    dfa.to_csv(os.path.join(out_dir, '{}_bias.csv'.format(station)))


if __name__ == '__main__':
    d = '/media/research/IrrigationGIS/climate/'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS/climate/'

    grd = GridMet(variable='etr', start='2000-01-01',
                  end='2023-05-17',
                  lat=46.15, lon=-105.8083333)
    grd = grd.get_point_timeseries()

    stations_ = os.path.join(d, 'ghcn', 'ghcn_stations_bitterroot.shp')
    # station_data_ = os.path.join(d, 'ghcn', 'ghcn_daily_summaries_4FEB2022')
    # out_dir_ = os.path.join(d, 'ghcn', 'ghcn_gridmet_comp')
    # compare_gridmet_ghcn(stations_, station_data_, out_dir_)

    station_data_ = os.path.join(d, 'agrimet', 'mt_stations')
    out_dir_ = '/home/dgketchum/PycharmProjects/et-demands/et-demands/static'
    compare_gridmet_agrimet('lmmm', station_data_, out_dir_)
# ========================= EOF ====================================================================
