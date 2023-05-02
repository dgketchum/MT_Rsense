import os
import pytz
from datetime import datetime, date

import pandas as pd
import seaborn as sns
import matplotlib.dates as mdates
from toolbox import EasyMap, pc
from paint.standard2 import cm_tmp, cm_pcp
import matplotlib.pyplot as plt

from herbie import Herbie
from noaa_sdk import NOAA


def test_nomad(met_dir, fxx_length=72):
    now = datetime.now()
    mountain_tz = pytz.timezone('US/Mountain')

    if fxx_length <= 72:
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
        rounded = date.today()

    dt_str = rounded.strftime('%Y-%m-%d %H')

    H = Herbie(
        dt_str,
        model='gfs',
        product='pgrb2.0p25',
        fxx=240
    )

    key_url = 'https://www.nco.ncep.noaa.gov/pmb/products/nam/nam.t00z.conusnest.hiresf06.tm00.grib2.shtml'
    keys = pd.read_html(key_url)[1]
    inv = H.inventory()
    match = [i for i in keys.index if i in inv.index]
    inv.loc[match, 'desc'] = keys.loc[match, 'Description']
    inv = inv[['search_this', 'desc', 'grib_message', 'start_byte', 'end_byte', 'range', 'reference_time', 'valid_time',
               'variable', 'level', 'forecast_time']]

    ds = H.xarray('MAXVW:10 m above')
    # ds = H.xarray('TMP:2 m above')

    ax = EasyMap('50m', crs=ds.herbie.crs, figsize=(8, 8)).STATES().ax
    p = ax.pcolormesh(ds.longitude, ds.latitude, ds.unknown, transform=pc,
                      **cm_pcp(units='mm').cmap_kwargs)

    plt.colorbar(p, ax=ax, orientation='horizontal', pad=0.05, **cm_pcp(units='mm').cbar_kwargs)

    ax.set_title(
        f"{ds.model.upper()}: {H.product_description} "
        f"({H.product})\nValid: {ds.valid_time.dt.strftime('%H:%M UTC %d %b %Y').item()}",
        loc="left")
    ax.set_title(ds.unknown.GRIB_name, loc="right")
    plt.show()


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
    test_nomad(met_dir_, fxx_length=240)
# ========================= EOF ====================================================================

