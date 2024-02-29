import os
import pytz
from copy import deepcopy
from operator import itemgetter
from calendar import monthrange
from datetime import datetime, date
from itertools import groupby, islice, tee

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import dataretrieval.nwis as nwis

from hydrograph import read_hydrograph
from state_county_names_codes import state_county_code

PARAMS = {'station_nm': 'STANAME',
          'dec_lat_va': 'LAT',
          'dec_long_va': 'LON',
          'county_cd': 'CO_CODE',
          'alt_va': 'ELEV',
          'huc_cd': 'HUC',
          'construction_dt': 'CONSTRU',
          'drain_area_va': 'BASINAREA'}

ORDER = ['STAID', 'STANAME', 'start', 'end', 'last_rec', 'rec_len', 'CO_CODE', 'LAT', 'LON',
         'ELEV', 'HUC', 'CONSTRU', 'BASINAREA', 'COUNTY', 'geometry']


def get_usgs_station_metadata(in_shp, out_shp):
    df = gpd.read_file(in_shp)
    df.index = df['SOURCE_FEA']
    df = df[['geometry']]
    df['CO_CODE'] = [0 for _ in df.iterrows()]

    co_codes = state_county_code()['MT']

    today = datetime.now(pytz.utc)

    for sid, feat in df.iterrows():
        data = nwis.get_info(sites=[sid])

        try:
            for param, name in PARAMS.items():
                df.loc[sid, name] = data[0][param][0]

            df.loc[sid, 'COUNTY'] = co_codes[str(df.loc[sid, 'CO_CODE']).rjust(3, '0')]['NAME']

        except KeyError as e:
            print('\n Error on {}: {}'.format(sid, e))

        recs = nwis.get_record(sites=[sid], start='1800-01-01', end='2023-08-31', service='dv')
        if not isinstance(recs.index, pd.DatetimeIndex):
            df.loc[sid, 'start'] = 'None'
            df.loc[sid, 'end'] = 'None'
        else:
            s, e = recs.index[0], recs.index[-1]
            rec_start = '{}-{}-{}'.format(s.year, str(s.month).rjust(2, '0'), str(s.day).rjust(2, '0'))
            df.loc[sid, 'start'] = rec_start
            rec_end = '{}-{}-{}'.format(e.year, str(e.month).rjust(2, '0'), str(e.day).rjust(2, '0'))
            df.loc[sid, 'end'] = rec_end
            df.loc[sid, 'last_rec'] = np.round((today - e).days / 365.25, 2)
            df.loc[sid, 'rec_len'] = df.shape[0]
            print(sid, df.loc[sid, 'STANAME'], rec_start, rec_end)

    df['STAID'] = df.index
    df = df[ORDER]
    df.to_file(out_shp)


def get_station_daily_data(start, end, stations, out_dir, plot_dir=None, overwrite=False):
    dt_range = pd.date_range(start, end, freq='D')
    ct_df = pd.DataFrame(index=pd.DatetimeIndex(dt_range), data=np.arange(len(dt_range)))
    ct_df = ct_df.groupby([ct_df.index.year, ct_df.index.month]).agg('count')
    counts = [r[0] for i, r in ct_df.iterrows()]

    stations = gpd.read_file(stations)
    stations.index = stations['SOURCE_FEA']

    for sid, data in stations.iterrows():

        # if sid != '06065500':
        #     continue

        out_file = os.path.join(out_dir, '{}.csv'.format(sid))
        if os.path.exists(out_file) and not overwrite:
            print(sid, 'exists, skipping')
            continue

        df = nwis.get_record(sites=sid, service='dv', start=start, end=end, parameterCd='00060')
        df = df.tz_convert(None)

        if df.empty:
            print(sid, ' is empty')
            continue

        q_col = '00060_Mean'
        df = df.rename(columns={q_col: 'q'})
        df = df.reindex(pd.DatetimeIndex(dt_range), axis=0)

        df['q'] = np.where(df['q'] < 0, np.zeros_like(df['q']) * np.nan, df['q'])
        nan_count = np.count_nonzero(np.isnan(df['q']))

        # exclude months without complete data
        if nan_count > 0:
            df['q'] = df['q'].dropna(axis=0)
            record_ct = df['q'].groupby([df.index.year, df.index.month]).agg('count')
            records = [r for i, r in record_ct.items()]
            mask = [0] + [int(a == b) for a, b in zip(records, counts)]
            missing_mo = len(counts) - sum(mask)
            resamp_start = pd.to_datetime(start) - pd.DateOffset(months=1)
            resamp_end = pd.to_datetime(end) + pd.DateOffset(months=1)
            mask = pd.Series(index=pd.DatetimeIndex(pd.date_range(resamp_start, resamp_end, freq='M')),
                             data=mask).resample('D').bfill()
            mask = mask[1:]
            df = df.loc[mask[mask == 1].index, 'q']
            print('write {:.1f}'.format(data['BASINAREA']), sid, 'missing {} months'.format(missing_mo),
                  data['STANAME'])
        else:
            df = df['q']
            print('write {:.1f}'.format(data['BASINAREA']), sid, data['STANAME'])

        df.to_csv(out_file)

        if plot_dir:
            plt.plot(df.index, df)
            plt.savefig(os.path.join(plot_dir, '{}.png'.format(sid)))
            plt.close()


def get_station_daterange_data(daily_q_dir, aggregate_q_dir, resample_freq='A',
                               convert_to_mcube=True, plot_dir=None):
    q_files = [os.path.join(daily_q_dir, x) for x in os.listdir(daily_q_dir)]
    sids = [os.path.basename(c).split('.')[0] for c in q_files]
    out_records, short_records = [], []
    for sid, c in zip(sids, q_files):

        # if sid != '06065500':
        #     continue

        df = pd.read_csv(c, index_col=0, infer_datetime_format=True, parse_dates=True)

        # cfs to m ^3 d ^-1
        df = df['q']
        if convert_to_mcube:
            df = df * 2446.58
        df = df.resample(resample_freq).agg(pd.DataFrame.sum, skipna=False)
        dates = deepcopy(df.index)

        out_file = os.path.join(aggregate_q_dir, '{}.csv'.format(sid))
        df.to_csv(out_file)
        out_records.append(sid)
        print(sid)

        if plot_dir:
            pdf = pd.DataFrame(data={'Date': dates, 'q': df.values})
            pdf.plot('Date', 'q')
            plt.savefig(os.path.join(plot_dir, '{}.png'.format(sid)))
            plt.close()

    print('{} processed'.format(len(out_records)))
    print(out_records)


def find_lowest_flows(shapes, monthly_q, out_shape):
    gdf = gpd.read_file(shapes)
    gdf.index = gdf['STAID']

    years = range(1987, 2022)
    months = list(range(4, 11))

    periods = [(m,) for m in months] + [list(consecutive_subseq(months, x)) for x in range(2, 7)]
    periods = [item for sublist in periods for item in sublist]
    periods = [i if isinstance(i, tuple) else (i, i) for i in periods]
    str_pers = ['{}-{}'.format(q_win[0], q_win[-1]) for q_win in periods]
    nan_data = [np.nan for i, r in gdf.iterrows()]
    for p in str_pers:
        gdf[p] = nan_data

    for sid, data in gdf.iterrows():
        _file = os.path.join(monthly_q, '{}.csv'.format(sid))

        try:
            df = read_hydrograph(_file)
        except FileNotFoundError:
            print(sid, 'not found')
            continue

        df['ppt'], df['etr'] = df['gm_ppt'], df['gm_etr']
        df = df[['q', 'etr', 'ppt', 'cc']]
        df[df == 0.0] = np.nan
        for q_win in periods:
            key_ = '{}-{}'.format(q_win[0], q_win[-1])
            q_dates = [(date(y, q_win[0], 1), date(y, q_win[-1], monthrange(y, q_win[-1])[1])) for y in years]
            q = np.array([df['q'][d[0]: d[1]].sum() for d in q_dates])
            finite = np.isfinite(q)

            if sum(finite) != len(q):
                df.loc[sid, key_] = int(0)

            q = q[finite]
            y = np.array(years)[finite]
            _min_year = y[np.argmin(q)].item()
            gdf.loc[sid, key_] = int(_min_year)
            print(_min_year, sid, key_, 'of {} years'.format(len(y)))

    vals = gdf[str_pers]
    vals[np.isnan(vals)] = 0.0
    vals = vals.astype(int)
    gdf[str_pers] = vals
    gdf.drop(columns=['STAID'], inplace=True)
    gdf.to_file(out_shape)
    df = pd.DataFrame(gdf)
    df.drop(columns=['geometry'], inplace=True)
    df.to_csv(out_shape.replace('.shp', '.csv'))


def consecutive_subseq(iterable, length):
    for _, consec_run in groupby(enumerate(iterable), lambda x: x[0] - x[1]):
        k_wise = tee(map(itemgetter(1), consec_run), length)
        for n, it in enumerate(k_wise):
            next(islice(it, n, n), None)  # consume n items from it
        yield from zip(*k_wise)


if __name__ == '__main__':
    gages_in = '/media/research/IrrigationGIS/usgs_gages/mt_usgs_gages.shp'
    gages_out = '/media/research/IrrigationGIS/usgs_gages/mt_usgs_gages_por.shp'
    # get_usgs_station_metadata(gages_in, gages_out)

    daily_q_ = '/media/research/IrrigationGIS/usgs_gages/mt_hydrographs_daily'
    get_station_daily_data('1900-01-01', '2023-11-01', gages_out, daily_q_)
    monthly_q_ = '/media/research/IrrigationGIS/usgs_gages/mt_hydrographs_monthly'
    get_station_daterange_data(daily_q_,  monthly_q_, resample_freq='M')

# ========================= EOF ====================================================================
