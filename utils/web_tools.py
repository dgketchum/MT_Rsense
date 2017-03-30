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
from mechanize import Browser, _http


def open_browser(site):

    br = Browser()
    br.set_handle_equiv(True)
    br.set_handle_gzip(True)
    br.set_handle_redirect(True)
    br.set_handle_referer(True)
    br.set_handle_robots(False)
    br.set_handle_refresh(_http.HTTPRefreshProcessor(), max_time=1)
    br.set_debug_http(True)
    br.set_debug_redirects(True)
    br.set_debug_responses(True)

    r = br.open(site)
    html = r.read()
    print 'Check out the available forms:'
    for f in br.forms():
        print f

    return br


def lat_lon_to_path_row(lat, lon):
    site = 'https://landsat.usgs.gov/wrs-2-pathrow-latitudelongitude-converter'
    # site = 'http://google.com'
    browser = open_browser(site)
    browser.select_form(nr=0)
    browser.form['search_block_form'] = str(lat)
    # browser.form['q'] = str(lat)
    browser.submit()
    print 'response: {}'.format(browser.response().read())



if __name__ == '__main__':
    home = os.path.expanduser('~')
    print 'home: {}'.format(home)
    lat_lon_to_path_row(47.5, 107.2)

    # ===============================================================================
