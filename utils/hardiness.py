import os
import json
import pytz
from datetime import datetime

import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from utils.thredds import GridMet

import warnings


def suppress_numpy_warning():
    warnings.filterwarnings('ignore', category=RuntimeWarning, message='All-NaN (slice|axis) encountered')


suppress_numpy_warning()

PARAMS = {'tmin': 'tmmn'}

COL_ORDER = ['STANAME', 'LAT', 'LON', 'ELEV', 'STATION_START', 'STATION_END', 'analysis_start', 'analysis_end',
             'missing_records', 'valid_records', 'total_records', 'state', 'tmin_f', 'tmin_date']


def gridmet_subset_stack(start, end, variable, out_dir, template_raster):
    annual_arrs, failures = [], []
    raster = os.path.join(out_dir, 'tmmn_{}_{}.tif'.format(start, end))

    if not os.path.isdir(out_dir):
        raise IOError('{} not a directory'.format(out_dir))

    with rasterio.open(template_raster, 'r') as ras:
        profile = ras.profile

    def get_it(start_, end_):
        gridmet = GridMet(variable=variable, start=s, end=e)
        arr_ = gridmet.full_array(start_, end_)
        tmin_ = np.nanmin(arr_, axis=0) - 273.15
        return tmin_

    for yr in range(start, end):
        s, e = '{}-01-01'.format(yr), '{}-12-31'.format(yr)
        try:
            tmin = get_it(s, e)
        except Exception as e:
            print(yr, e)
            failures.append(yr)
        annual_arrs.append(tmin)
        print(yr)

    if failures:
        for yr in failures:
            s, e = '{}-01-01'.format(yr), '{}-12-31'.format(yr)
            try:
                tmin = get_it(s, e)
            except Exception as e:
                print('failed again', yr, e)
            annual_arrs.append(tmin)
            print(yr)

    arr = np.array(annual_arrs).min(axis=0)

    with rasterio.open(raster, 'w', **profile) as dst:
        dst.write(arr, 1)


def download_ghcn(station_id, file_dst, start):
    url = 'https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station/{}.csv.gz'.format(station_id)

    df = pd.read_csv(url, header=None, usecols=[1, 2, 3, 6])
    df.columns = ['DATE', 'PARAM', 'VALUE', 'FLAG']
    params = list(np.unique(df['PARAM']))
    target_cols = ['TMIN', 'TMAX', 'PRCP']

    if not all([p in params for p in target_cols]):
        print('records missing some parameters')
        return None

    df = df.pivot_table(index='DATE', columns=['PARAM'],
                        values='VALUE', aggfunc='first').reindex()

    df = df[target_cols]

    # pd.to_datetime returns all 1970-01-01 on 'DATE'
    dt_ind = pd.DatetimeIndex([datetime.strptime(str(d), '%Y%m%d') for d in df.index], tz='UTC')
    df.index = dt_ind

    if df.index[-1] < pd.to_datetime(start, utc=True):
        print('records end before study period')
        return None

    df.to_csv(file_dst)

    return df


def gridmet_infill(df, *variables, lat, lon, units):
    df = df.tz_convert(None)
    for v in list(variables):
        p = PARAMS[v]
        g = GridMet(p, lat=lat, lon=lon, start=df.index[0], end=df.index[-1], )
        s = g.get_point_timeseries()
        if p.startswith('t'):
            if units == 'metric':
                s[p] -= 273.15
            else:
                s[p] = ((s[p] - 273.15) * 9 / 5) + 32.

        else:
            if units == 'standard':
                s[p] *= 0.03937008

        df[v].fillna(s[p], inplace=True)

    df.index = pd.DatetimeIndex(df.index, tz='UTC')
    return df


def ghcn_station_minimum_temp(stations_file, ghcn_data, out_str, start='1994-01-01', end='2023-12-31',
                              units='metric', nodata_value=-999, overwrite=False, bound_shp=None):
    invalid_stations = 0
    dropped = []

    dct = {}

    dt_index = pd.date_range(start=start, end=end, tz='UTC')
    start_dt = datetime.strptime(start_ds, '%Y-%m-%d')
    start_dt = start_dt.replace(tzinfo=pytz.UTC)

    bounds_shp = gpd.read_file(bound_shp)
    bounds_shp = bounds_shp.set_index('STUSPS')

    for state, r in bounds_shp.iterrows():

        if state != 'AK':
            continue

        print('\n\n\n', state, '\n\n\n')

        outfile = out_str.format(state, start[:4], end[:4])

        bounds = bounds_shp.loc[state, 'geometry']
        stations = gpd.read_file(stations_file, bbox=bounds)

        stations = stations.set_index('STAID')
        stations = stations.to_dict(orient='index')

        initial_ct = len(stations.keys())

        for k, v in stations.items():

            if k != 'USC00509641':
                continue

            _file = os.path.join(ghcn_data, '{}.csv'.format(k))

            if not os.path.exists(_file) or overwrite:
                df = download_ghcn(k, _file, start)
                if not isinstance(df, pd.DataFrame):
                    continue
            else:
                try:
                    df = pd.read_csv(_file, parse_dates=True, index_col=1, low_memory=False)
                    df.index = pd.DatetimeIndex(df.index, tz='UTC')
                    cols = sorted(list(df.columns))
                except Exception as e:
                    invalid_stations += 1
                    dropped.append(k)
                    continue

                if 'TMIN' not in cols:
                    invalid_stations += 1
                    dropped.append(k)
                    continue

            s = v['START']

            if pd.to_datetime(start) > pd.to_datetime(s):
                df = df.loc[start:]
                if df.empty or (df.shape[0] / len(dt_index)) < 0.7:
                    # print(k, 'insuf records in date range')
                    invalid_stations += 1
                    dropped.append(k)
                    continue

            else:
                print(k, '{} records'.format(df.shape[0]))

            if df.index[-1] < start_dt:
                dfs, dfe = df.index[0], df.index[-1]
                df_start = '{}-{}-{}'.format(dfs.year, dfs.month, dfs.day)
                df_end = '{}-{}-{}'.format(dfe.year, dfe.month, dfe.day)
                # print(k, 'record entirely before target date range: {} to {}'.format(df_start, df_end))
                invalid_stations += 1
                dropped.append(k)
                continue

            df = df.reindex(dt_index)

            try:
                df = df[['TMIN']]

                df['tmin'] = df['TMIN'] / 10.
                df[df['tmin'] > 43.0] = np.nan
                df[df['tmin'] < -40.0] = np.nan

                df = df[['tmin']]

                if units != 'metric':
                    df['tmin'] = (df['tmin'] * 9 / 5) + 32.

                if df.empty:
                    # print(k, 'insuf records in date range')
                    invalid_stations += 1

                nan_ct, size = np.count_nonzero(np.isnan(df.values)), df.values.size
                valid_ct = np.count_nonzero(~np.isnan(df.values))

                if nan_ct / size > 0.2:
                    # print('{} missing {} of {} values'.format(k, nan_ct, size))
                    invalid_stations += 1
                    dropped.append(k)
                    continue

            except KeyError as e:
                # print(k, e)
                invalid_stations += 1
                dropped.append(k)
                continue

            _ = v.pop('geometry')
            _ = v.pop('FID')
            v['STANAME'] = v['STANAME'].replace(' ', '_')
            v['STATION_START'] = v.pop('START')
            v['STATION_END'] = v.pop('END')
            v['analysis_start'] = start
            v['analysis_end'] = end
            v['missing_records'] = nan_ct
            v['valid_records'] = valid_ct
            v['total_records'] = df.shape[0]
            v['state'] = state

            tmin = df['tmin'].min(axis=0)
            tmin_idx = df.index[df['tmin'].argmin()]
            tmin_ds = '{}-{}-{}'.format(tmin_idx.year, str(tmin_idx.month).rjust(2, '0'),
                                        str(tmin_idx.day).rjust(2, '0'))
            v['tmin_f'] = tmin
            v['tmin_date'] = tmin_ds
            dct[k] = v

        df = pd.DataFrame.from_dict(dct).T
        df.to_csv(outfile, columns=COL_ORDER, index='STAID', index_label='STAID')

        print('wrote {} with {} station records, of {} stations'.format(os.path.basename(outfile),
                                                                    len(dct.keys()),
                                                                    initial_ct))


def concatenate_state_data(bound_shp, in_str, out_str, start='1994-01-01', end='2023-12-31'):

    bounds_shp = gpd.read_file(bound_shp)
    bounds_shp = bounds_shp.set_index('STUSPS')

    first = True

    for state, r in bounds_shp.iterrows():
        _file = in_str.format(state, start[:4], end[:4])
        if first:
            df = pd.read_csv(_file)
            df['geometry'] = df.apply(lambda rr: Point(rr['LON'], rr['LAT']), axis=1)
            df = gpd.GeoDataFrame(df)
            df = df[df.intersects(r['geometry'])]
            first = False
        else:
            c = pd.read_csv(_file)
            c['geometry'] = c.apply(lambda rr: Point(rr['LON'], rr['LAT']), axis=1)
            c = gpd.GeoDataFrame(c)
            c = c[c.intersects(r['geometry'])]
            df = pd.concat([df, c], axis=0, ignore_index=True)

    df = df.sort_values(by='state')
    out_file = out_str.format(start[:4], end[:4])
    df.to_csv(out_file)
    print(os.path.basename(out_file), df.shape[0])


if __name__ == '__main__':
    d = '/media/research/IrrigationGIS'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS'

    shapes = os.path.join(d, 'boundaries', 'tl_2017_us_state.shp')
    rasters = os.path.join(d, 'climate', 'hardiness')
    template = os.path.join(rasters, 'gridmet_elev.tif')

    # gridmet_subset_stack(1994, 2024, 'tmmn', rasters, template_raster=template)
    # gridmet_subset_stack(1979, 2009, 'tmmn', rasters, template_raster=template)

    ghcn_dir = os.path.join(d, 'climate', 'ghcn')
    min_tmp_d = os.path.join(ghcn_dir, 'ghcn_min_temps')
    stations_ = os.path.join(ghcn_dir, 'ghcn_stations_us.shp')
    daily_data_dir = os.path.join(ghcn_dir, 'ghcn_daily_summaries_4FEB2022')

    bounds = os.path.join(d, 'boundaries', 'states', 'us_state_20m', 'cb_2016_us_state_20m_4326.shp')

    min_tmp_out = os.path.join(min_tmp_d, 'ghcn_stations_us_{}_{}_{}.csv')
    min_tmp_csv = os.path.join(min_tmp_d, 'ghcn_stations_{}_{}.csv')

    start_ds = '1994-01-01'
    end_ds = '2023-12-31'
    # ghcn_station_minimum_temp(stations_, daily_data_dir, min_tmp_out, start_ds, end_ds, units='standard',
    #                           overwrite=False, bound_shp=bounds)

    concatenate_state_data(bounds, min_tmp_out, min_tmp_csv, start_ds, end_ds)

    start_ds = '1980-01-01'
    end_ds = '2009-12-31'
    # ghcn_station_minimum_temp(stations_, daily_data_dir, min_tmp_out, start_ds, end_ds, units='standard',
    #                           overwrite=False, bound_shp=bounds)

    # concatenate_state_data(bounds, min_tmp_out, min_tmp_csv, start_ds, end_ds)
# ========================= EOF ====================================================================
