import os
from copy import deepcopy

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, MultiPolygon
import matplotlib.pyplot as plt
import seaborn as sns

from utils.agrimet import load_stations
from reference_et.combination import pm_fao56, get_rn
from reference_et.rad_utils import extraterrestrial_r, calc_rso
from reference_et.modified_bcriddle import modified_blaney_criddle
from utils.elevation import elevation_from_coordinate
from utils.thredds import GridMet
from utils.elevation import elevation_from_coordinate
import warnings

warnings.filterwarnings(action='once')

large = 22
med = 16
small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (16, 10),
          'axes.labelsize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large,
          'xtick.color': 'black',
          'ytick.color': 'black',
          'xtick.direction': 'out',
          'ytick.direction': 'out',
          'xtick.bottom': True,
          'xtick.top': False,
          'ytick.left': True,
          'ytick.right': False,
          }

plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')
sns.set_style("white", {'axes.linewidth': 0.5})


def field_comparison(shp, etof, out):
    gdf = gpd.read_file(shp)
    start, end = '2016-01-01', '2021-12-31'

    summary = deepcopy(gdf)
    summary['etos'] = [-99.99 for _ in summary['FID']]
    summary['etbc'] = [-99.99 for _ in summary['FID']]
    summary['geo'] = [None for _ in summary['FID']]
    idx = max(summary.index.values) + 1

    ct = 0
    for i, row in gdf.iterrows():
        file_ = os.path.join(etof, str(row['FID']).rjust(3, '0'))
        lon, lat = row.geometry.centroid.x, row.geometry.centroid.y
        elev = elevation_from_coordinate(lat, lon)
        gridmet = GridMet('pet', start=start, end=end,
                          lat=lat, lon=lon)
        grd = gridmet.get_point_timeseries()
        grd = grd.rename(columns={'pet': 'ETOS'})
        for var, _name in zip(['tmmn', 'tmmx', 'pr'], ['MN', 'MX', 'PP']):
            ts = GridMet(var, start=start, end=end,
                         lat=lat, lon=lon).get_point_timeseries()
            if 'tm' in var:
                ts -= 273.15

            grd[_name] = ts

        grd['MM'] = (grd['MN'] + grd['MX']) / 2
        bc, start, end, kc = modified_blaney_criddle(grd, lat, elev)

        bc_pet = bc['u'].sum()
        summary.loc[i, 'etbc'] = bc_pet

        grd['mday'] = ['{}-{}'.format(x.month, x.day) for x in grd.index]
        target_range = pd.date_range('2000-{}-{}'.format(start.month, start.day),
                                     '2000-{}-{}'.format(end.month, end.day))

        accept = ['{}-{}'.format(x.month, x.day) for x in target_range]
        grd['mask'] = [1 if d in accept else 0 for d in grd['mday']]
        grd = grd[grd['mask'] == 1]

        df = pd.read_csv(file_, index_col=0, infer_datetime_format=True, parse_dates=True)
        df = df.rename(columns={list(df.columns)[0]: 'etof'})
        r_index = pd.date_range('2016-01-01', '2021-12-31', freq='D')
        df = df.reindex(r_index)
        df = df.interpolate()
        df['mday'] = ['{}-{}'.format(x.month, x.day) for x in df.index]
        target_range = pd.date_range('2016-05-09', '2016-09-19')

        accept = ['{}-{}'.format(x.month, x.day) for x in target_range]
        df['mask'] = [1 if d in accept else 0 for d in df['mday']]
        df = df[df['mask'] == 1]
        if isinstance(row['geometry'], MultiPolygon):
            first = True
            for g in row['geometry'].geoms:
                if first:
                    summary.loc[i, 'geo'] = g
                    summary.loc[i, 'etof'] = df['etof'].mean()

                    first = False
                else:
                    summary.loc[idx] = row
                    summary.loc[idx, 'geo'] = g
                    summary.loc[idx, 'etof'] = df['etof'].mean()
                    idx += 1
        else:
            summary.loc[i, 'geo'] = row['geometry']
            summary.loc[i, 'etof'] = df['etof'].mean()

        pass


def point_comparison_iwr_stations(_dir, meta_csv, out_summary):
    meta_df = pd.read_csv(meta_csv)
    summary = deepcopy(meta_df)

    summary['etos'] = [-99.99 for _ in summary['LON']]
    summary['etbc'] = [-99.99 for _ in summary['LON']]
    summary['geo'] = [None for _ in summary['LON']]

    for i, row in meta_df.iterrows():
        _file = os.path.join(_dir, '{}.csv'.format(row['STAID']))
        df = pd.read_csv(_file)
        lat = df.iloc[0]['LATITUDE']
        lon = df.iloc[0]['LONGITUDE']
        elev = row['ELEV']
        geo = Point(lon, lat)
        summary.loc[i, 'geo'] = geo
        start, end = '1997-01-01', '2006-12-31'
        gridmet = GridMet('pet', start=start, end=end,
                          lat=lat, lon=lon)
        grd = gridmet.get_point_timeseries() / 25.4

        dt_index = pd.date_range(start, end)

        df.index = pd.to_datetime(df['DATE'])

        df = df.reindex(dt_index)

        try:
            df = df[['TMAX', 'TMIN', 'PRCP']]

            df['MX'] = df['TMAX'] / 10.
            df['MN'] = df['TMIN'] / 10.
            df['PP'] = df['PRCP'] / 10.
            df = df[['MX', 'MN', 'PP']]
            df['MM'] = (df['MX'] + df['MN']) / 2

            bc, start, end, kc = modified_blaney_criddle(df, lat, elev)

            bc_pet = bc['ref_u'].sum()
            summary.loc[i, 'etbc'] = bc_pet

            grd['mday'] = ['{}-{}'.format(x.month, x.day) for x in grd.index]
            target_range = pd.date_range('2000-{}-{}'.format(start.month, start.day),
                                         '2000-{}-{}'.format(end.month, end.day))
            accept = ['{}-{}'.format(x.month, x.day) for x in target_range]
            grd['mask'] = [1 if d in accept else 0 for d in grd['mday']]
            grd = grd[grd['mask'] == 1]

            grd_etos = grd['pet'].resample('A').sum().mean()
            summary.loc[i, 'etos'] = grd_etos
        except Exception as e:
            print(row, e)

        print('{}: bc {:.3f} grd {:.3f}'.format(row['NAME'], bc_pet, grd_etos))

    gdf = gpd.GeoDataFrame(summary)
    gdf.geometry = summary['geo']
    gdf.drop(columns=['geo'], inplace=True)
    gdf = gdf.set_crs('epsg:4326')
    gdf.to_file(out_summary)
    gdf.to_csv(out_summary.replace('.shp', '.csv'))


def point_comparison_agrimet(station_dir, out_figs, out_shp):
    stations = load_stations()
    station_files = [os.path.join(station_dir, x) for x in os.listdir(station_dir)]
    etbc, etos = [], []
    station_comp = {}
    for f in station_files:
        sid = os.path.basename(f).split('.')[0]
        meta = stations[sid]
        coords = meta['geometry']['coordinates']
        geo = Point(coords)
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
        df['ETRS'] = df['ETOS'] * 1.2

        try:
            bc, start, end, kc = modified_blaney_criddle(df, coords[1], elev)
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

        years = len(list(set([d.year for d in df.index])))

        etos_ = df['ETOS'].resample('A').sum().mean() / 25.4
        etbc.append(etbc_)
        etos.append(etos_)
        diff_ = (etbc_ - etos_) / etos_
        station_comp[sid] = {'STAID': sid, 'etbc': etbc_, 'etos': etos_,
                             'no_yrs': years, 'diff': diff_ * 100., 'geo': geo}
        print(sid)

    df = pd.DataFrame(station_comp, columns=station_comp.keys()).T
    df = df.astype({'etbc': float,
                    'etos': float,
                    'diff': float})
    df = df.astype({'no_yrs': int})

    gdf = gpd.GeoDataFrame(df)

    gdf.geometry = gdf['geo']
    gdf.drop(columns=['geo'], inplace=True)
    gdf = gdf.set_crs('epsg:4326')
    gdf.to_file(out_shp)
    gdf.to_csv(out_shp.replace('.shp', '.csv'))

    xmin, xmax = min([min(etos), min(etbc)]) - 2, max([max(etos), max(etbc)]) + 2
    line = np.linspace(xmin, xmax)
    sns.scatterplot(etos, etbc)
    sns.lineplot(line, line, alpha=0.8, dashes=[(2, 2), (2, 2)])
    plt.xlim([xmin, xmax])
    plt.ylim([xmin, xmax])
    plt.xlabel('Penman-Monteith (FAO 56)')
    plt.ylabel('Blaney Criddle (SCS TR 21)')
    plt.suptitle('Seasonal Crop Reference Evapotranspiration Comparison [inches/season]')
    _filename = os.path.join(out_figs, 'FAO_56_SCSTR21_MBC_comparison.png')
    # plt.show()
    plt.savefig(_filename)


def check_implementation(station='USC00242409', data_dir=None, iwr_table=None,
                         start='1970-01-01', end='2000-12-31'):

    _file = os.path.join(data_dir, '{}.csv'.format(station))
    df = pd.read_csv(_file)
    dt_index = pd.date_range(start, end)
    df.index = pd.to_datetime(df['DATE'])
    df = df.reindex(dt_index)

    lat = df.iloc[0]['LATITUDE']
    lon = df.iloc[0]['LONGITUDE']
    elev = elevation_from_coordinate(lat, lon)

    df = df[['TMAX', 'TMIN', 'PRCP']]

    df['MX'] = df['TMAX'] / 10.
    df['MN'] = df['TMIN'] / 10.
    df['PP'] = df['PRCP'] / 10.
    df = df[['MX', 'MN', 'PP']]
    df['MM'] = (df['MX'] + df['MN']) / 2

    bc, start, end, kc = modified_blaney_criddle(df, lat, elev,
                                                 season_start='2000-05-09',
                                                 season_end='2000-09-19',
                                                 mid_month=False)

    pass


if __name__ == '__main__':
    d = '/media/research/IrrigationGIS'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS'

    r = os.path.join(d, 'Montana', 'water_rights', 'hua', 'comparison_data')
    shp_ = os.path.join(r, 'sweetgrass_fields_sample.shp')
    etof_ = os.path.join(r, 'sweetgrass_fields_etof')
    out_summary = os.path.join(r, 'sweetgrass_fields_comparison.shp')
    # field_comparison(shp_, etof_, out_summary)

    iwr_data_dir = os.path.join(d, 'climate', 'montana_iwr_station_data', 'from_ghcn')
    stations = os.path.join(d, 'climate', 'stations', 'mt_arm_iwr_stations.csv')
    comp = os.path.join(d, 'climate', 'stations', 'iwr_gridmet_comparison.shp')
    # point_comparison_iwr_stations(iwr_data_dir, stations, comp)

    _dir = os.path.join(d, 'climate/agrimet/mt_stations')
    fig_dir = os.path.join(d, 'climate/agrimet/comparison_figures')
    out_shp = os.path.join(d, 'climate/agrimet/shapefiles/comparison.shp')
    # point_comparison_agrimet(station_dir=_dir, out_figs=fig_dir, out_shp=out_shp)

    iwr_table = None
    start, end = '1971-01-01', '2000-12-31'
    # start, end = '1997-01-01', '2006-12-31'
    # check_implementation('USC00242409', iwr_data_dir, iwr_table, start=start, end=end)
# ========================= EOF ====================================================================
