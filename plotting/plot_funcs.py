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
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap


def plot_basemap(data, bounds):
    lons, lats = bounds
    subset = np.transpose(data)
    lon_bar = lons.mean()
    lat_bar = lats.mean()

    m = Basemap(width=5e6, height=3.5e5, resolution='l', projection='stere',
                lat_ts=45, lat_0=lat_bar, lon_0=lon_bar)
    lon, lat = np.meshgrid(lons, lats)
    xi, yi = m(lon, lat)

    cs = m.pcolor(xi, yi, np.squeeze(subset))
    # Add Grid Lines
    m.drawparallels(np.arange(-80., 81., 10.), labels=[1, 0, 0, 0], fontsize=10)
    m.drawmeridians(np.arange(-180., 181., 10.), labels=[0, 0, 0, 1], fontsize=10)

    # Add Coastlines, States, and Country Boundaries
    m.drawcoastlines()
    m.drawstates()
    m.drawcountries()

    # Add Colorbar
    cbar = m.colorbar(cs, location='bottom', pad="10%")
    # cbar.set_label(units)

    # Add Title
    plt.title('DJF Maximum Temperature')

    plt.show()
    # print 'subset data in {} units'.format(nc.variables['potential_evapotranspiration'].units)


if __name__ == '__main__':
    home = os.path.expanduser('~')

# ===============================================================================
