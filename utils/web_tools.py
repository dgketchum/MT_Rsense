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
from lxml import html


def lat_lon_to_path_row((lat, lon)):
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
    lat_lon_to_path_row((47.5, -107.2))

# ===============================================================================
