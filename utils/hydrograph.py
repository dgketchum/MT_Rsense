import os
import json

import pandas as pd
import numpy as np
import dataretrieval.nwis as nwis


def get_station_daily_data(start, end, sid, out_dir=None, overwrite=False, full_months=False):
    dt_range = pd.date_range(start, end, freq='D')
    ct_df = pd.DataFrame(index=pd.DatetimeIndex(dt_range), data=np.arange(len(dt_range)))
    ct_df = ct_df.groupby([ct_df.index.year, ct_df.index.month]).agg('count')
    counts = [r[0] for i, r in ct_df.iterrows()]

    if out_dir:
        out_file = os.path.join(out_dir, '{}.csv'.format(sid))
        if os.path.exists(out_file) and not overwrite:
            print(sid, 'exists, skipping')
            return None

    df = nwis.get_record(sites=sid, service='dv', start=start, end=end, parameterCd='00060')
    df = df.tz_convert(None)

    if df.empty:
        print(sid, ' is empty')
        return None

    q_col = '00060_Mean'
    df = df.rename(columns={q_col: 'q'})
    df = df.reindex(pd.DatetimeIndex(dt_range), axis=0)

    df['q'] = np.where(df['q'] < 0, np.zeros_like(df['q']) * np.nan, df['q'])
    nan_count = np.count_nonzero(np.isnan(df['q']))

    # exclude months without complete data
    if nan_count > 0 and full_months:
        df['q'] = df['q'].interpolate(limit=7, method='linear')
        df['q'] = df['q'].dropna(axis=0)
        record_ct = df['q'].groupby([df.index.year, df.index.month]).agg('count')
        records = [r for i, r in record_ct.items()]
        mask = [int(a == b) for a, b in zip(records, counts)]
        missing_mo = len(counts) - sum(mask)
        mask = pd.Series(index=pd.DatetimeIndex(pd.date_range(start, end, freq='M')),
                         data=mask).resample('D').bfill()
        df = df.loc[mask[mask == 1].index, 'q']
        print('write {}, missing {} months'.format(sid, missing_mo))
    else:
        df = df['q']
    if out_dir:
        df.to_csv(out_file)
    else:
        return df


def read_hydrograph(c):
    df = pd.read_csv(c)
    if 'Unnamed: 0' in list(df.columns):
        df = df.rename(columns={'Unnamed: 0': 'dt'})
    try:
        df['dt'] = pd.to_datetime(df['dt'])
    except KeyError:
        df['dt'] = pd.to_datetime(df['datetimeUTC'])
    df = df.set_index('dt')
    try:
        df.drop(columns='datetimeUTC', inplace=True)
    except KeyError:
        pass
    try:
        df = df.tz_convert(None)
    except:
        pass
    return df


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS/Montana/upper_yellowstone/hydrographs'
    dirs_ = [os.path.join(d, x) for x in os.listdir(d)]

    for d in dirs_:
        daily = os.path.join(d, 'daily')
        monthly = os.path.join(d, 'monthly')
        sid = os.path.basename(d)
        get_station_daily_data('2016-01-01', '2021-12-31', sid, d)
# ========================= EOF ====================================================================
