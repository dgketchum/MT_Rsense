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
import numpy as np
from netCDF4 import Dataset, num2date
from datetime import datetime, timedelta
from xlrd import xldate

lat_bound = [44, 49]
lon_bound = [-117, -104]


def get_gridmet(day):
    variables = ['pr', 'rmax', 'rmin', 'sph', 'srad', 'th', 'tmmn', 'tmmx', 'pet', 'vs']

    site = 'http://thredds.northwestknowledge.net:8080/thredds/dodsC/MET/pet/{}.nc'.format(day.year)
    nc = Dataset(site)
    print nc.variables.keys()
    # convert time axis to datetime object
    time_var = nc.variables['day'][:]
    # sph = nc.variables['specific_humidity'][:]
    print 'variable of type {} has shape {}'.format(type(time_var), time_var.shape)
    for item in time_var:
        pass

    # find indices of lat lon bounds in nc file
    lats = nc.variables['lat'][:]
    lons = nc.variables['lon'][:]
    lat_lower = np.argmin(np.abs(lats - lat_bound[1]))
    lat_upper = np.argmin(np.abs(lats - lat_bound[0]))
    lon_lower = np.argmin(np.abs(lons - lon_bound[1]))
    lon_upper = np.argmin(np.abs(lons - lon_bound[0]))

    # subset = nc.variables['specific_humidity'][:, lat_lower:lat_upper, lon_lower:lon_upper]
    # print subset


if __name__ == '__main__':
    home = os.path.expanduser('~')
    date = datetime(2002, 4, 1, 12)
    get_gridmet(date)

    # ===============================================================================
