# ===============================================================================
# Copyright 2017 dgketchum
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===============================================================================
import os
import requests
import collections
import re
from datetime import datetime, timedelta
from lxml import html
from dateutil.rrule import rrule, DAILY


def get_L5_overpass(lat_lon, start_date):
    url = 'https://cloudsgate2.larc.nasa.gov/cgi-bin/predict/predict.cgi'
    # submit form > copy POST data
    payload = dict(c=compute,
                   sat=LANDSAT + 5,
                   instrument=0 - 0,
                   res=9,
                   month=05,
                   day=01,
                   year=2007,
                   numday=20,
                   viewangle,
                   solarangle=all,
                   gif=track,
                   ascii=element,
                   lat=47.45,
                   lon=-107.951,
                   sitename=Optional,
                   choice=track)

    r = requests.get(url).content
    print r
    return None


def landsat_overpass_data(path_row, start_date, satellite):
    delta = timedelta(days=16)
    end = start_date + delta
    if satellite == 'LT5':

        lat_lon = lat_lon_wrs2pr_convert(path_row, converstion_type='convert_pr_to_ll')


    else:
        base = 'https://landsat.usgs.gov/landsat/all_in_one_pending_acquisition/'
        for day in rrule(DAILY, dtstart=start_date, until=end):

            tail = '{}/Pend_Acq/y{}/{}/{}.txt'.format(satellite, day.year,
                                                      day.strftime('%b'),
                                                      day.strftime('%b-%d-%Y'))

            print 'Searching: {}'.format(day.strftime('%b-%d-%Y'))
            url = '{}{}'.format(base, tail)
            r = requests.get(url)
            for line in r.iter_lines():
                l = line.split()
                try:
                    if l[0] == str(path_row[0]):
                        if l[1] == str(path_row[1]):
                            # dtime is in GMT
                            time_str = '{}-{}'.format(start.year, l[2])
                            time = datetime.strptime(time_str, '%Y-%j-%H:%M:%S')
                            print 'datetime object: {}'.format(time)
                            dct = dict(path=l[0], row=l[1], dtime=time, station=l[3])
                            return dct

                except IndexError:
                    pass

        raise NotImplementedError('Did not find overpass data...')


def lat_lon_wrs2pr_convert(pr_latlon, converstion_type='convert_ll_to_pr'):
    base = 'https://landsat.usgs.gov/landsat/lat_long_converter/tools_latlong.php'

    if converstion_type == 'convert_ll_to_pr':
        full_url = '{}?rs=&rsargs[]={}&rsargs[]={}&rsargs[]=1&rsrnd=1490995492704'.format(base, converstion_type,
                                                                                          pr_latlon[0], pr_latlon[1])
        r = requests.get(full_url)
        tree = html.fromstring(r.text)

        # remember to view source html to build xpath
        # i.e. inspect element > network > find GET with relevant PARAMS > go to GET URL > view source HTML
        p_string = tree.xpath('//table/tr[1]/td[2]/text()')
        path = int(re.search(r'\d+', p_string[0]).group())
        r_string = tree.xpath('//table/tr[1]/td[4]/text()')
        row = int(re.search(r'\d+', r_string[0]).group())
        print 'path: {}, row: {}'.format(path, row)

        return path, row

    elif converstion_type == 'convert_pr_to_ll':
        full_url = '{}?rs={}&rsargs[]=\n' \
                   '{}&rsargs[]={}&rsargs[]=1&rsrnd=1490995492704'.format(base, converstion_type,
                                                                          pr_latlon[0], pr_latlon[1])

        r = requests.get(full_url)
        tree = html.fromstring(r.text)
        lat_string = tree.xpath('//table/tr[2]/td[2]/text()')
        lat = re.search(r'\d+\.\d+', lat_string[0]).group()
        lon_string = tree.xpath('//table/tr[2]/td[4]/text()')
        lon = re.search(r'\d+\.\d+', lon_string[0]).group()
        print 'lat: {}, lon: {}'.format(lat, lon)

        return lat, lon

    else:
        raise NotImplementedError('Must chose either convert_pr_to_ll or convert_ll_to_pr')


if __name__ == '__main__':
    home = os.path.expanduser('~')
    print 'home: {}'.format(home)
    path_row = (37, 27)
    lat_lon = 47.45, -107.951
    start = datetime(2007, 05, 10)
    get_L5_overpass(lat_lon, start)

# ===============================================================================
