import json
from calendar import monthrange
from pprint import pprint
from datetime import datetime

import requests

requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

with open('/home/dgketchum/PycharmProjects/MT_Rsense/data/auth.json', 'r') as j:
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


if __name__ == '__main__':
    FIELDS = 'projects/openet/field_boundaries/UpperYellowstone/UpperYellowstone_27SEPT2021'
    for year in range(2018, 2021):
        for month in range(5, 11):
            if year == 2018 and month == 5:
                continue
            monthly_ensemble(FIELDS, year, month, filename_suffix='uy_{}_{}'.format(year, month))
            print(datetime.now())
# ========================= EOF ====================================================================
