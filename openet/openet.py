import json
import os
from calendar import monthrange
from pprint import pprint
from datetime import datetime

import requests
from pandas import read_csv, concat

requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

with open('/home/dgketchum/PycharmProjects/MT_Rsense/data/openet_auth.json', 'r') as j:
    API_KEY = json.load(j)['auth']


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
    # FIELDS = 'projects/earthengine-legacy/assets/projects/openet/field_boundaries/Upper_Yellowstone_3DEC2021'
    # for year in range(2016, 2021):
    #     for month in range(1, 13):
    #         if year == 2016 and month == 1:
    #             continue
    #         monthly_ensemble(FIELDS, year, month, filename_suffix='uy_{}_{}'.format(year, month))
    #         print(datetime.now())
    d = '/home/dgketchum/Downloads/lost_creek_fan/'
    o = os.path.join(d, 'lost_creek_fan_openet.csv')
    r = os.path.join(d, 'openet_data')
    concatenate_openet_data(r, o)
# ========================= EOF ====================================================================
