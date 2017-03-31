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


def lat_lon_to_path_row(lat, lon):
    # site = 'https://landsat.usgs.gov/wrs-2-pathrow-latitudelongitude-converter'
    site = 'https://landsat.usgs.gov/landsat/lat_long_converter/tools_latlong.php'

    data = [('rs', 'convert_ll_to_pr'),
            ('rsargs[]', str(lat)),
            ('rsargs[]', str(lon)),
            ('rsargs[]', '1'),
            ('rsrnd', '1490993174595')]

    data = collections.OrderedDict(data)
    r = requests.get(site, data=data)
    print r.apparent_encoding
    print r.content


if __name__ == '__main__':
    home = os.path.expanduser('~')
    print 'home: {}'.format(home)
    lat_lon_to_path_row(47.5, 107.2)

    # ===============================================================================
