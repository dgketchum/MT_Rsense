import json
import os
from time import sleep
from calendar import monthrange
from pprint import pprint
from datetime import datetime

import requests
from pandas import read_csv, concat, DataFrame, date_range, to_datetime
import geopandas as gpd

requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

with open('/home/dgketchum/PycharmProjects/MT_Rsense/data/openet_auth.json', 'r') as j:
    API_KEY = json.load(j)['auth']


def point_series(shp, out, var='et'):
    gdf = gpd.read_file(shp)
    start, end = '2016-01-01', '2021-12-31'
    ct = 0
    for i, row in gdf.iterrows():
        fid = row['FID']
        if not fid <= 20:
            continue
        file_ = os.path.join(out, str(fid).rjust(3, '0'))
        if os.path.exists(file_):
            continue
        lon, lat = row.geometry.centroid.xy[0], row.geometry.centroid.xy[1]
        header = {"Authorization": API_KEY}
        params = {'start_date': start,
                  'end_date': end,
                  'lon': lon, 'lat': lat,
                  'output_file_format': 'json',
                  'variable': var,
                  'model': 'ensemble',
                  'ref_et_source': 'gridmet',
                  'units': 'metric',
                  'interval': 'monthly'}

        server = 'https://openet.dri.edu/'
        endpoint = 'raster/timeseries/point'
        url = server + endpoint

        try:
            resp = requests.get(url=url, headers=header, params=params, verify=False)
            data = [d[var] for d in resp.json()]
            time_ = [d['time'] for d in resp.json()]
            time_ = to_datetime(time_)
            df = DataFrame(data, index=time_, columns=[row['IType']])
            df.to_csv(file_)
            ct += 1
            print(ct, file_)
            sleep(1.)
        except TypeError as e:
            print(row['FID'], 'failed', e)
            sleep(1.)


def monthly_ensemble(shp, year, month, filename_suffix='upper_yellowstone'):

    start = '{}-{}-01'.format(year, str(month).rjust(2, '0'))
    end_day = monthrange(year, month)[1]
    end = '{}-{}-{}'.format(year, str(month).rjust(2, '0'), end_day)

    header = {"Authorization": API_KEY}
    params = {'start_date': start,
              'end_date': end,
              'shapefile_asset_id': shp,
              'include_columns': 'FID',
              'variable': 'et',
              'model': 'ensemble',
              'ref_et_source': 'gridmet',
              'units': 'metric',
              'output_date_format': 'standard',
              'filename_suffix': filename_suffix}

    server = 'https://openet.dri.edu/'
    endpoint = 'raster/timeseries/multipolygon'
    url = server + endpoint

    resp = requests.get(url=url, headers=header, params=params, verify=False)
    pprint(resp.text)


def concatenate_openet_data(_dir, out_csv):
    l = [os.path.join(_dir, x) for x in os.listdir(_dir)]
    first = True
    for f in l:
        if first:
            df = read_csv(f, index_col='time', infer_datetime_format=True,
                          parse_dates=True).drop(columns=['Id']).rename(columns={'time': 'Date'})
            first = False
        else:
            c = read_csv(f, index_col='time', infer_datetime_format=True,
                         parse_dates=True).drop(columns=['Id']).rename(columns={'time': 'Date'})

            df = concat([df, c])
    df = df.sort_index()
    df.to_csv(out_csv)


if __name__ == '__main__':
    d = '/media/research/IrrigationGIS'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS'
    # r = os.path.join(d, 'Montana', 'water_rights', 'hua', 'comparison_data')
    # shp_ = os.path.join(r, 'sweetgrass_fields_sample.shp')
    r = os.path.join(d, 'Montana', 'iad')
    shp_ = os.path.join(r, 'Harpers_Itype_Centroids.shp')
    out_ = os.path.join(r, 'harper_fields_etof')
    point_series(shp_, out_, var='etof')
# ========================= EOF ====================================================================
