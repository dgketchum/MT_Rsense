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


def landsat_7_8_overpass_data((path, row), start_date, satellite='L7'):
    delta = timedelta(days=17)
    end = start_date + delta
    for day in rrule(DAILY, dtstart=start_date, until=end):

        base = 'https://landsat.usgs.gov/landsat/all_in_one_pending_acquisition/'

        tail = '{}/Pend_Acq/y{}/{}/{}.txt'.format(satellite, day.year,
                                                  day.strftime('%b'),
                                                  day.strftime('%b-%d-%Y'))

        print 'Start: {}'.format(day.strftime('%b-%d-%Y'))
        url = '{}{}'.format(base, tail)
        r = requests.get(url)
        print r.content
        for line in r.iter_lines():
            l = line.split()
            try:
                add
                lat, lon
                to
                L7 and L5
                path
                row, make
                test
                if l[0] == str(path):
                    if l[1] == str(row):
                        # dtime is in GMT
                        dct = dict(path=l[0], row=l[1], dtime=l[2], station=l[3])
            except IndexError:
                pass

    return dct


def lat_lon_to_wrs2_path_row((lat, lon)):
    data = [('rs', 'convert_ll_to_pr'),
            ('rsargs1[]', str(lat)),
            ('rsargs2[]', str(lon)),
            ('rsargs3[]', '1'),
            ('rsrnd', '1490993174595')]

    data = collections.OrderedDict(data)

    full_url = 'https://landsat.usgs.gov/landsat/lat_long_converter/tools_latlong.php?rs=convert_ll_to_pr&rsargs[]=\n' \
               '{}&rsargs[]={}&rsargs[]=1&rsrnd=1490995492704'.format(data['rsargs1[]'], data['rsargs2[]'])

    r = requests.get(full_url)
    tree = html.fromstring(r.text)

    # remember to view source html to build xpath
    p_string = tree.xpath('//table/tr[1]/td[2]/text()')
    path = int(re.search(r'\d+', p_string[0]).group())
    r_string = tree.xpath('//table/tr[1]/td[4]/text()')
    row = int(re.search(r'\d+', r_string[0]).group())
    print 'path: {}, row: {}'.format(path, row)

    return int(path), int(row)


if __name__ == '__main__':
    home = os.path.expanduser('~')
    print 'home: {}'.format(home)
    path_row = (36, 25)
    start = datetime(2007, 05, 01)
    landsat_7_8_overpass_data(path_row, start_date=start, satellite='L5')

# ===============================================================================
