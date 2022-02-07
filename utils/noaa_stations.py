import os
import json
from datetime import date
from calendar import monthrange

import numpy as np
from pandas import DataFrame, to_datetime, read_csv, date_range, concat
import requests


def generate_precip_records(climate_dir, co_fip='30001'):
    try:
        with open('/home/dgketchum/ncdc_noaa_token.json', 'r') as j:
            API_KEY = json.load(j)['auth']
    except FileNotFoundError:
        with open('/home/dgketchum/data/IrrigationGIS/ncdc_noaa_token.json', 'r') as j:
            API_KEY = json.load(j)['auth']

    header = {"token": API_KEY}
    params = {'locationid': 'FIPS:{}'.format(co_fip),
              'limit': 52,
              'datasetid': 'GHCND'}

    server = 'https://www.ncdc.noaa.gov/cdo-web/api/v2/'
    url = server + 'stations'

    resp = requests.get(url=url, headers=header, params=params)
    jsn = resp.json()
    max_dates = [int(x['maxdate'][:4]) for x in jsn['results']]
    min_dates = [int(x['mindate'][:4]) for x in jsn['results']]
    per_records = [(x, y) for x, y in zip(min_dates, max_dates)]
    stations = [x['id'] for x in jsn['results']]

    for i, (sid, p_rec) in enumerate(zip(stations, per_records), start=1):
        if p_rec[0] > 1997 or p_rec[1] < 2021:
            continue
        dates, values = [], []
        for s in [1997, 2007, 2017]:
            params = {'stationid': sid,
                      'limit': 1000,
                      'units': 'metric',
                      'datasetid': 'GSOM',
                      'datatypeid': 'PRCP',
                      'startdate': '{}-01-01'.format(s),
                      'enddate': '{}-12-31'.format(s + 9)}
            url = server + 'data'
            resp = requests.get(url=url, headers=header, params=params)
            try:
                r_jsn = resp.json()
                res = r_jsn['results']
            except KeyError:
                print('{} attempt {} failed'.format(co_fip, i))
                continue
            dates = dates + [x['date'] for x in res]
            values = values + [x['value'] for x in res]

        df = DataFrame(data={'prec': values, 'date': dates})
        df.index = to_datetime(dates)
        params = {'stationid': sid,
                  'limit': 1000,
                  'startdate': '2010-01-01',
                  'enddate': '2010-12-01',
                  'datatypeid': 'MLY-PRCP-NORMAL',
                  'units': 'metric',
                  'datasetid': 'NORMAL_MLY'}

        url = server + 'data'
        resp = requests.get(url=url, headers=header, params=params)
        try:
            res = resp.json()['results']
        except KeyError:
            print('{} attempt {} failed'.format(co_fip, i))
            continue
        dt_range = date_range('1901-01-1', '1901-12-31', freq='M')
        values = [x['value'] for x in res]
        ndf = DataFrame(data={'prec': values, 'date': dt_range}, index=dt_range)
        df = concat([df, ndf], axis=0, ignore_index=False)
        df['date'] = df.index
        out_csv = os.path.join(climate_dir, '{}.csv'.format(co_fip))
        df.to_csv(out_csv)
        print('save climate to {}'.format(out_csv))
        return None
    print('{} found no stations'.format(co_fip))


def get_prec_anomaly(csv, start_month=4, end_month=9):
    df = read_csv(csv, index_col='date', parse_dates=True)
    df.drop(columns=[x for x in df.columns if x != 'prec'], inplace=True)
    df = df.sort_index()

    dct = {}

    s = '1901-{}-01'.format(str(start_month).rjust(2, '0'))
    month_end = monthrange(1901, end_month)[1]
    e = '1901-{}-{}'.format(str(end_month).rjust(2, '0'), month_end)
    normal = df[s: e]['prec'].sum()

    s, e = df['1997-01-01':].index[0].year, df['1997-01-01':].index[-1].year
    dates = [(date(y, start_month, 1), date(y, end_month, monthrange(y, end_month)[1])) for y in range(s, e + 1)]
    deltas = [df[d[0]:d[1]].shape[0] for d in dates]
    check_range = deltas.count(deltas[0]) == len(deltas)

    # check for short records
    r_mode = False
    if not check_range:
        vals, counts = np.unique(deltas, return_counts=True)
        r_mode = vals[np.argmax(counts)].item()
    for d, delta in zip(dates, deltas):
        if delta < r_mode:
            continue
        dct[d[0].year] = df['prec'][d[0]: d[1]].sum() - normal

    prcp = list(dct.items())
    prcp.sort(key=lambda x: x[1])
    return prcp


def get_climate(geoid, climate_dir, return_dry=True, n_years=3):
    precip_record = os.path.join(climate_dir, '{}.csv'.format(geoid))
    if not os.path.exists(precip_record):
        generate_precip_records(climate_dir, geoid)
        if not return_dry:
            return None
    precip_record = get_prec_anomaly(precip_record)
    dry_years = [x[0] for x in precip_record[:n_years]]
    if return_dry:
        return dry_years
    else:
        return precip_record


if __name__ == '__main__':
    gis = os.path.join('/media/research', 'IrrigationGIS')
    if not os.path.exists(gis):
        gis = '/home/dgketchum/data/IrrigationGIS'
    _co_climate = os.path.join(gis, 'training_data', 'humid', 'county_precip_normals')
    generate_precip_records('30001', _co_climate)
# ========================= EOF ====================================================================
