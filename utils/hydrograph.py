import os
import json

import pandas as pd
import numpy as np
import dataretrieval.nwis as nwis


def get_station_flows(start, end, sid, freq='dv', out_dir=None, overwrite=False):

    if out_dir:
        out_file = os.path.join(out_dir, '{}.csv'.format(sid))
        if os.path.exists(out_file) and not overwrite:
            print(sid, 'exists, skipping')
            return None

    df = nwis.get_record(sites=[sid], service=freq, start=start, end=end, parameterCd='00060')
    df.index = pd.DatetimeIndex([pd.to_datetime(i, utc=True) for i in df.index])

    if df.empty:
        print(sid, start, end, ' is empty')
        return None

    freq_map = {'dv': '00060_Mean', 'iv': '00060'}
    q_col = freq_map[freq]
    df = df.rename(columns={q_col: 'q'})

    df['q'] = np.where(df['q'] < 0, np.zeros_like(df['q']) * np.nan, df['q'])

    df = df[['q']]
    if out_dir:
        df.to_csv(out_file)
    else:
        return pd.DataFrame(df)


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

# ========================= EOF ====================================================================
