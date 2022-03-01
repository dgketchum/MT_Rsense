import os
from copy import deepcopy
from datetime import datetime as dt

import numpy as np
from pandas import read_csv, DataFrame, date_range
import requests
from io import StringIO

from state_county_names_codes import state_name_abbreviation

URL_STRINGS = ['WTEQ::value', 'TMIN::value', 'TMAX::value', 'TAVG::value', 'PREC::value']
URL_LEAD = 'https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/customGroupByMonthReport'


def snotel_meteorology(csv, out_dir, overwrite=False):

    df = read_csv(csv)
    df = df[df['Network'] == 'SNOTEL']
    df['ID'] = df['ID'].apply(lambda x: x.strip())

    abbrev = {v: k for k, v in state_name_abbreviation().items()}
    station_dct, start_dates = {}, []
    vars_ = ['swe', 'tmin', 'tmax', 'tavg', 'prec']
    header, data_start = None, None

    for _id, row in df.iterrows():
        incomplete = False
        st = abbrev[row['State']].lower()

        desc = '{}_{}_{}.csv'.format(_id, row['Name'].replace(' ', '_'), st.upper())
        print(desc)
        file_ = os.path.join(out_dir, desc)

        if os.path.exists(file_) and not overwrite:
            continue

        for url_str, var in zip(URL_STRINGS, vars_):

            url = '{}/daily/{}:{}:SNTL%7Cid=%22%22%7Cname/POR_BEGIN,POR_END/{}'.format(URL_LEAD, row['ID'],
                                                                                       st, url_str)
            resp = requests.get(url).text.splitlines()
            if len(resp) < 60:
                continue
            for i, l in enumerate(resp):
                if l.startswith('Water Year'):
                    header = i
                    data_start = i + 2

            lines = '\n'.join([resp[header]] + resp[data_start:])
            sttrr = StringIO(lines)
            wy_sta_df = read_csv(sttrr)
            wy_sta_df = wy_sta_df.rename(columns={'Water Year': 'wy', 'Day': 'd'})
            station_dct[var] = deepcopy(wy_sta_df)

        if incomplete:
            continue
        date_index = date_range('1990-01-01', '2021-12-31')
        empty = np.ones((len(date_index), len(vars_))) * np.nan
        sta_df = DataFrame(data=empty, columns=vars_, index=date_index)

        months = [(m, dt.strftime(dt(1991, m, 1), '%b')) for m in range(1, 13)]
        for var, c in station_dct.items():
            for m, mon_ in months:
                mdf = deepcopy(c[['wy', 'd', mon_]])
                if m >= 10:
                    mdf.loc[:, 'wy'] = mdf['wy'] - 1
                for _, m_row in mdf.iterrows():
                    if m_row['wy'] == 1990 and m < 10 or m_row['wy'] < 1990:
                        continue
                    loc_ = '{}-{}-{}'.format(int(m_row['wy']), m, int(m_row['d']))
                    val = m_row[mon_]

                    # convert to mm, deg C
                    if var in ['swe', 'prec']:
                        val *= 25.4
                    if var in ['tmin', 'tmax', 'tavg']:
                        val = (val - 32) * 5 / 9
                    sta_df.loc[loc_, var] = val

        prec = list(sta_df.loc[:, 'prec'].values)
        sta_df['prec'] = [np.nan] + [prec[i] - prec[i - 1] for i in range(1, len(prec))]

        sta_df.to_csv(file_, float_format='%.3f')


if __name__ == '__main__':
    d = '/media/research/IrrigationGIS/climate/snotel'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS/climate/snotel'
    sno_list = os.path.join(d, 'snotel_list.csv')
    rec_dir = os.path.join(d, 'snotel_records')
    snotel_meteorology(sno_list, rec_dir)
# ========================= EOF ====================================================================
