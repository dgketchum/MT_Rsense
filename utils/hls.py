import os
import os
import json
from datetime import datetime
import requests as r
import numpy as np
import pandas as pd
import geopandas as gp
from skimage import io
import matplotlib.pyplot as plt
from osgeo import gdal
import rasterio as rio
from rasterio.mask import mask
from rasterio.enums import Resampling
from rasterio.shutil import copy
import pyproj
from pyproj import Proj
from shapely.ops import transform
import xarray as xr
import geoviews as gv
# from cartopy import crs
import hvplot.xarray
import holoviews as hv


r.packages.urllib3.disable_warnings(r.packages.urllib3.exceptions.InsecureRequestWarning)

with open('/home/dgketchum/PycharmProjects/MT_Rsense/data/edl_auth.json', 'r') as j:
    API_KEY = json.load(j)['auth']


def get_stac_hls(figs):
    header = {"Authorization": API_KEY}
    params = {}
    shp = '/media/research/IrrigationGIS/Montana/water_rights/clark_fork_12334550_4326.geojson'
    basin = gp.read_file(shp)
    bounds = basin.loc[0].geometry.bounds
    bbox = [x for x in bounds]

    lp_cloud = 'https://cmr.earthdata.nasa.gov/stac/LPCLOUD'
    lp_links = r.get(lp_cloud).json()['links']
    lp_search = [l['href'] for l in lp_links if l['rel'] == 'search'][0]
    date_time = "2013-07-01T00:00:00Z/2021-9-30T23:59:59Z"  # Define start time period / end time period
    params = {'limit': 100, 'bbox': bbox,
              'datetime': date_time,
              'collections': ['HLSS30.v1.5', 'HLSS30.v1.5']}

    resp = r.post(lp_search, json=params, headers=header).json()
    for f in resp['features']:
        href = f['assets']['browse']['href']
        i = io.imread(href)
        plt.figure()
        plt.imshow(i)
        f_file = os.path.join(figs, '{}.png'.format(f['id']))
        plt.savefig(f_file)
        print(f_file)
    pass



if __name__ == '__main__':
    f = '/media/research/IrrigationGIS/Montana/water_rights/hls'
    get_stac_hls(f)
# ========================= EOF ====================================================================
