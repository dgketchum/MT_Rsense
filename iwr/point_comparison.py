import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from utils.agrimet import load_stations
from reference_et.combination import pm_fao56, get_rn
from reference_et.rad_utils import extraterrestrial_r, calc_rso
from reference_et.modified_bcriddle import modified_blaney_criddle
from utils.elevation import elevation_from_coordinate


def point_comparison_iwr_stations(_dir, meta_csv):
    meta_df = pd.read_csv(meta_csv)
    for i, row in meta_df.iterrows():
        if 'DILLON' not in row['NAME']:
            continue
        _file = os.path.join(_dir, '{}.csv'.format(row['STAID']))
        df = pd.read_csv(_file)
        lat = df.iloc[0]['LATITUDE']

        start, end = '1971-01-01', '2000-12-31'
        dt_index = pd.date_range(start, end)

        df.index = pd.to_datetime(df['DATE'])

        df = df.reindex(dt_index)

        df = df[['TMAX', 'TMIN', 'PRCP']]

        df['MX'] = df['TMAX'] / 10.
        df['MN'] = df['TMIN'] / 10.
        df['PP'] = df['PRCP'] / 10.
        df = df[['MX', 'MN', 'PP']]
        df['MM'] = (df['MX'] + df['MN']) / 2

        df['ETBC'] = modified_blaney_criddle(df, lat)


def point_comparison_agrimet(station_dir, out_figs, out_shp):
    stations = load_stations()
    station_files = [os.path.join(station_dir, x) for x in os.listdir(station_dir)]
    etbc, etos = [], []
    for f in station_files:
        sid = os.path.basename(f).split('.')[0]
        meta = stations[sid]
        coords = meta['geometry']['coordinates']
        coord_rads = np.array(coords) * np.pi / 180
        elev = elevation_from_coordinate(coords[1], coords[0])
        df = pd.read_csv(f, index_col=0, parse_dates=True,
                         infer_datetime_format=True, header=0,
                         skiprows=[1, 2, 3])
        tmean, tmax, tmin, wind, rs, rh = df['MM'], df['MX'], df['MN'], df['UA'], df['SR'], df['TA']
        ra = extraterrestrial_r(df.index, lat=coord_rads[1], shape=[df.shape[0]])
        rso = calc_rso(ra, elev)
        rn = get_rn(tmean, rs=rs, lat=coord_rads[1], tmax=tmax, tmin=tmin, rh=rh, elevation=elev, rso=rso)
        df['ETOS'] = pm_fao56(tmean, wind, rs=rs, tmax=tmax, tmin=tmin, rh=rh, elevation=elev, rn=rn)
        df['ETRS'] = df['ETOS'] * 1.

        try:
            bc, start, end = modified_blaney_criddle(df, coords[1])
        except IndexError:
            print(sid, 'failed')
            continue

        etbc_ = bc['ref_u'].sum()
        df['mday'] = ['{}-{}'.format(x.month, x.day) for x in df.index]
        target_range = pd.date_range('2000-{}-{}'.format(start.month, start.day),
                                     '2000-{}-{}'.format(end.month, end.day))
        accept = ['{}-{}'.format(x.month, x.day) for x in target_range]
        df.dropna(subset=['ETOS'], inplace=True)
        df['mask'] = [1 if d in accept else 0 for d in df['mday']]
        df = df[df['mask'] == 1]
        size = df.groupby(df.index.year).size()
        for i, s in size.iteritems():
            if s < len(target_range) - 1:
                df = df.loc[[x for x in df.index if x.year != i]]

        etos_ = df['ETOS'].resample('A').sum().mean() / 25.4
        etbc.append(etbc_)
        etos.append(etos_)
        print(sid)

    xmin, xmax = min([min(etos), min(etbc)]) - 5, max([max(etos), max(etbc)]) + 5
    line = np.linspace(xmin, xmax)
    plt.scatter(etos, etbc)
    plt.plot(line, line)
    plt.xlim([xmin, xmax])
    plt.ylim([xmin, xmax])
    plt.xlabel('FAO 56')
    plt.ylabel('Modified Blaney Criddle (SCS TR 21)')
    plt.show()


if __name__ == '__main__':
    d = '/media/research/IrrigationGIS'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS'

    iwr_data_dir = os.path.join(d, 'climate', 'montana_iwr_station_data', 'from_ghcn')
    stations = os.path.join(d, 'climate', 'stations', 'mt_arm_iwr_stations.csv')
    # point_comparison_iwr_stations(iwr_data_dir, stations)

    _dir = os.path.join(d, 'climate/agrimet/mt_stations')
    fig_dir = os.path.join(d, 'climate/agrimet/comparison_figures')
    out_shp = os.path.join(d, 'climate/agrimet/shapefiles/comparison.shp')

    point_comparison_agrimet(station_dir=_dir, out_figs=fig_dir, out_shp=out_shp)

# ========================= EOF ====================================================================
