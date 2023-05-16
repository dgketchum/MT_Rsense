import os
import pytz
import tempfile
import urllib
from datetime import datetime, timedelta, timezone

import numpy as np
import xarray
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from mpl_toolkits.basemap import Basemap

import pandas as pd
import seaborn as sns
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

QPF_URL = 'https://ftp.wpc.ncep.noaa.gov/2p5km_qpf/p24m_{}00f{}.grb'
GFS_URL = 'https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{}/00/atmos/gfs.t00z.pgrb2.0p25.f{}'


def get_gridded_forecast(met_dir, lat, lon, start_date='2023-04-09'):
    """
    Get gridded meteorological data from GFS and NAM, using NAM up to three days out, then GFS to seven days.
    """

    forecast_hours = list(range(3, 192, 3))

    if lon < 0:
        mod_lon = 180 - lon

    dt_start = datetime.strptime(start_date, '%Y-%m-%d')
    dt_str = dt_start.strftime('%Y-%m-%d %H')

    dt_idx = [(dt_start + timedelta(hours=fxx)) for fxx in forecast_hours]
    df = pd.DataFrame(index=dt_idx, columns=['tmin', 'tmax', 'ppt'])
    df.index = df.index.tz_localize(pytz.utc)
    df['fcst_hrs'] = forecast_hours

    for i, r in df.iterrows():
        fxx = r['fcst_hrs']
        key_csv = '../data/gfs_keys.csv'
        model = 'gfs'
        product = 'pgrb2.0p25'

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

        tmmx = H.xarray(var_search['tmax'])
        tmmx = tmmx.sel(latitude=lat, longitude=mod_lon, method='nearest').tmax.values.item()
        tmmx -= 273.15

        tmmn = H.xarray(var_search['tmin'])
        tmmn = tmmn.sel(latitude=lat, longitude=mod_lon, method='nearest').tmin.values.item()
        tmmn -= 273.15

        if fxx % 24 == 0:
            response = urllib.request.urlopen(QPF_URL.format(dt_start.strftime('%Y%m%d'), str(fxx).rjust(3, '0')))
            _file = response.read()
            with tempfile.NamedTemporaryFile(suffix=".grib2") as f:
                f.write(_file)
                qpf = xarray.open_dataset(f.name)
                qpf['longitude'] = xarray.where(qpf['longitude'] >= 180, qpf['longitude'] - 360, qpf['longitude'])
                abslat = np.abs(qpf.latitude - lat)
                abslon = np.abs(qpf.longitude - lon)
                c = np.maximum(abslon, abslat)
                ([yloc], [xloc]) = np.where(c == np.min(c))
                pt_ds = qpf.sel(x=xloc, y=yloc)
                qpf_val = pt_ds.tp.values.item()
                df.loc[i] = {'ppt': qpf_val, 'tmin': tmmn, 'tmax': tmmx}
        else:
            df.loc[i] = {'ppt': 0.0, 'tmin': tmmn, 'tmax': tmmx}

    mountain = pytz.timezone('US/Mountain')
    df.index = df.index.tz_convert(mountain)
    df_d = df.resample('D')
    agg_funcs = {'ppt': 'sum', 'tmin': 'min', 'tmax': 'max'}
    df = df_d.agg(agg_funcs)


def test_noaa(lat, lon):
    n = NOAA()
    res = n.points_forecast(lat, lon)
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
    get_gridded_forecast(met_dir_, lat=46.9, lon=-113.9, start_date='2023-05-10')
    # test_noaa(lat=46.9, lon=-113.9)
# ========================= EOF ====================================================================
