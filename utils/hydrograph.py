from pandas import read_csv, to_datetime

import os
import fiona
import hydrofunctions as hf
from pandas import date_range, DatetimeIndex, DataFrame


def get_station_daily_data(param, start, end, sid, freq='dv', out_dir=None):
    try:
        nwis = hf.NWIS(sid, freq, start_date=start, end_date=end)
        df = nwis.df(param)

        if freq == 'iv':
            out_file = os.path.join(out_dir, '{}_{}.csv'.format(sid, start[:4]))
            df.to_csv(out_file)

        elif out_dir:
            out_file = os.path.join(out_dir, '{}.csv'.format(sid))
            df.to_csv(out_file)

        else:
            return df

    except ValueError as e:
        print(e)
    except hf.exceptions.HydroNoDataError:
        print('no data for {} to {}'.format(start, end))
        pass


def get_station_daterange_data(year_start, daily_q_dir, aggregate_q_dir, start_month=None, end_month=None,
                               resample_freq='A', convert_to_mcube=True):
    q_files = [os.path.join(daily_q_dir, x) for x in os.listdir(daily_q_dir)]

    s, e = '{}-01-01'.format(year_start), '2020-12-31'
    daterange = date_range(s, e, freq='D')
    idx = DatetimeIndex(daterange, tz=None)

    out_records, short_records = [], []
    for c in q_files:
        sid = os.path.basename(c).split('.')[0]
        df = read_hydrograph(c)

        if start_month or end_month:
            idx_window = idx[idx.month.isin([x for x in range(start_month, end_month + 1)])]
            df = df[df.index.month.isin([x for x in range(start_month, end_month + 1)])]
            df = df[df.index.year.isin([x for x in range(year_start, 2021)])]
            idx = idx_window

        dflen, idxlen = df.shape[0], idx.shape[0]
        if dflen < idxlen:
            short_records.append(sid)
            if float(dflen) / idxlen < 0.8:
                print(sid, 'df: {}, idx: {}, q skipped'.format(df.shape[0], int(idx.shape[0])))
                continue
            df = df.reindex(idx)

        # cfs to m ^3 d ^-1
        if convert_to_mcube:
            df = df * 2446.58
        df = df.resample(resample_freq).agg(DataFrame.sum, skipna=False)

        out_file = os.path.join(aggregate_q_dir, '{}.csv'.format(sid))
        df.to_csv(out_file)
        out_records.append(sid)
        print(sid)

    print('{} processed'.format(len(out_records)))


def read_hydrograph(c):
    df = read_csv(c)
    if 'Unnamed: 0' in list(df.columns):
        df = df.rename(columns={'Unnamed: 0': 'dt'})
    try:
        df['dt'] = to_datetime(df['dt'])
    except KeyError:
        df['dt'] = to_datetime(df['datetimeUTC'])
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

    d = '/media/research/IrrigationGIS/Montana/water_rights/hydrographs'
    dirs_ = [os.path.join(d, x) for x in os.listdir(d)]

    for d in dirs_:
        dst = os.path.join(d, 'insta_q')
        sid = os.path.basename(d)
        for year in [x for x in range(1991, 2021)]:
            get_station_daily_data('discharge', '{}-01-01'.format(year), '{}-12-31'.format(year), sid, freq='iv',
                                   out_dir=dst)
# ========================= EOF ====================================================================
