import os
import pytz
from datetime import datetime, date

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.dates as mdates
from toolbox import EasyMap, pc
from paint.standard2 import cm_tmp, cm_pcp
import matplotlib.pyplot as plt

from herbie import Herbie
from noaa_sdk import NOAA


""""
 Get NOAA data from some different sources: gridded Global Forecast System and North American Mesoscale model
 temperature data at long and short time scales, respectively are fetched using Herbie 
 (https://github.com/blaylockbk/Herbie/tree/main/herbie).
 Precipitationn data (gridded) is from the NOAA Weather Prediction Center's Quantitative Precipitation Forecast, 
 which is an ensemble of various numerical models, using direct get request in xarray.
 Point weather data is fetched using the NOAA-SDK (https://github.com/paulokuong/noaa).
 
 
 See keys: 
 
 NAM: 'https://www.nco.ncep.noaa.gov/pmb/products/nam/nam.t00z.conusnest.hiresf06.tm00.grib2.shtml'
 GFS: 'https://www.nco.ncep.noaa.gov/pmb/products/gfs/gfs.t00z.pgrb2.0p25.f003.shtml'

"""


def get_gridded_forecast(met_dir, start_date='2023-05-04'):
    """
    Get gridded meteorological data from GFS and NAM, using NAM up to three days out, then GFS to seven days.
    """
    now = datetime.strptime(start_date, '%Y-%m-%d')
    mountain_tz = pytz.timezone('US/Mountain')
    forecast_hours = list(range(24, 264, 24))


    for fxx in forecast_hours:
        if fxx <= 10:
            key_csv = '../data/gfs_keys.csv'
            model = 'nam'
            product = None
            now = mountain_tz.localize(now)
            now = now.astimezone(pytz.utc)
            rounded = datetime(
                year=now.year,
                month=now.month,
                day=now.day,
                hour=(now.hour // 6) * 6,
                minute=0,
                second=0,
                microsecond=0)
        else:
            key_csv = '../data/gfs_keys.csv'
            model = 'gfs'
            product = 'pgrb2.0p25'
            rounded = date.today()

        dt_str = rounded.strftime('%Y-%m-%d %H')

        H = Herbie(
            dt_str,
            model=model,
            product=product,
            fxx=fxx)

        inv = H.inventory()
        keys = pd.read_csv(key_csv)

        match = [i for i in keys.index if i in inv.index]
        try:
            inv.loc[match, 'desc'] = keys.loc[match, 'Description']
        except KeyError:
            inv.loc[match, 'desc'] = keys.loc[match, 'desc']

        inv = inv[['search_this', 'desc', 'grib_message', 'start_byte', 'end_byte',
                   'range', 'reference_time', 'valid_time',
                   'variable', 'level', 'forecast_time']]

        var_search = {v['variable'].lower(): v['search_this'] for k, v in inv.iterrows()}

        ds = H.xarray(var_search['tmax'])
        slice_args = dict(latitude=slice(50, 31.5), longitude=slice(360 - 125, 360 - 100))
        ds = ds.sel(**slice_args)
        tmmx = H.xarray(var_search['tmax']).sel(**slice_args).tmax.values
        tmmn = H.xarray(var_search['tmin']).sel(**slice_args).tmin.values


def test_noaa():
    n = NOAA()
    res = n.get_forecasts('59802', 'US')
    l = [r for r in res]
    df = pd.DataFrame(l)

    df.index = [pd.to_datetime(x) for x in df['startTime']]
    df['windSpeed'] = [int(w.split(' ')[0]) for w in df['windSpeed']]
    df['precipProb'] = [v['value'] for v in df['probabilityOfPrecipitation']]
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 10))
    sns.lineplot(data=df, x=df.index, y='temperature', legend='auto', ax=ax)
    sns.lineplot(data=df, x=df.index, y='windSpeed', legend='auto', ax=ax)
    sns.scatterplot(data=df, x=df.index, y='precipProb', legend='auto', ax=ax)
    plt.xticks(rotation=45)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%a %m/%d  %H:%M'))
    plt.grid()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'

    # test_noaa()
    met_dir_ = os.path.join(root, 'climate', 'forecast')
    get_gridded_forecast(met_dir_)
# ========================= EOF ====================================================================
