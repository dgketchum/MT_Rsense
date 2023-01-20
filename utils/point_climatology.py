import os

from pandas import concat, Series, read_csv, DataFrame, date_range
from datetime import date, datetime, timedelta
from dateutil.rrule import rrule, DAILY
import numpy as np
import rasterio

import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings(action='once')

large = 22
med = 16
small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (16, 10),
          'axes.labelsize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large,
          'xtick.color': 'black',
          'ytick.color': 'black',
          'xtick.direction': 'out',
          'ytick.direction': 'out',
          'xtick.bottom': True,
          'xtick.top': False,
          'ytick.left': True,
          'ytick.right': False,
          }
plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')
sns.set_style("white", {'axes.linewidth': 0.5})

from gridmet import GridMet


def aspect_histogram(raster, out):
    with rasterio.open(raster, 'r') as src:
        a = src.read()
    a = a.ravel()
    a = a[a > -0.1]
    h = np.histogram(a, bins=100)
    x, y = [x for x in h[1][:-1]], h[0]
    df = Series(y, index=x)
    sns.lineplot(data=df)
    plt.xlabel('Cardinal Direction (degrees)')
    plt.ylabel('Count')
    plt.suptitle('Lidar Aspect Histogram (60 cm)\n14480 Harpers Bridge Rd.')
    # plt.show()
    plt.savefig(out)


def point_climatology(lat, lon, out_csv):
    vars_ = ['pr', 'tmmn', 'tmmx', 'vs', 'th']

    first, df = True, None
    for v in vars_:
        grd = GridMet(variable=v, start='1981-01-01', end='2020-12-31',
                      lat=lat, lon=lon)
        if first:
            df = grd.get_point_timeseries()
            first = False
        else:
            pt_data = grd.get_point_timeseries()
            df = concat([df, pt_data], axis=1)

    for v in ['tmmn', 'tmmx']:
        df[v] = (df[v] - 273.15) * 9 / 5 + 32

    df.to_csv(out_csv, float_format='%.2f', index_label='date')

    a = date(2021, 1, 1)
    b = date(2021, 12, 31)
    years = [x for x in range(1981, 2021)]

    dct = {v: {'mean': [], 'std': [], 'date': []} for v in ['tmmn', 'tmmx', 'vs']}
    for dt in rrule(DAILY, dtstart=a, until=b):
        for v in dct.keys():
            dates = [date(year, dt.month, dt.day).strftime('%Y-%m-%d') for year in years]
            vals = df.loc[dates, v].values

            mean_ = vals.mean()
            std_ = vals.std()

            dct[v]['mean'].append(mean_)
            dct[v]['std'].append(std_)
            dct[v]['date'].append(dt)

    for k, v in dct.items():
        _file = out_csv.replace('.csv', '_{}.csv'.format(k))
        DataFrame(data={'mean': v['mean'], 'std': v['std']},
                  index=v['date']).to_csv(_file, float_format='%.2f', index_label='date')

    _file = out_csv.replace('.csv', '_min_winter_temp.csv')
    min_temps = []
    for year in years:
        s = df.loc['{}-01-01'.format(year): '{}-12-31'.format(year), 'tmmn']
        idx = s.argmin()
        d, val = s.index[idx], s.iloc[idx]
        min_temps.append((d, val))

    s = Series(index=[x[0] for x in min_temps], data=[x[1] for x in min_temps])
    s.name = 'min_t_F'
    s.to_csv(_file, float_format='%.2f', index_label='date')


def plot_climatology(csv):
    for v, t in zip(['tmmn', 'tmmx', 'vs'], ['Mean Minimum Temperature (F)',
                                             'Mean Maximum Temperature (F)',
                                             'Mean Wind Speed (m/s)']):
        _file = csv.replace('.csv', '_{}.csv'.format(v))
        df = read_csv(_file, parse_dates=True, index_col='date', infer_datetime_format=True)
        fig_ = os.path.join(_file.replace('.csv', '.png'.format(v)))
        time_series(df, t, fig_)

    _file = csv.replace('.csv', '_min_winter_temp.csv')
    min_winter_t = read_csv(_file, parse_dates=True, infer_datetime_format=True, index_col='date')
    fig_ = _file.replace('.csv', '.png')
    histogram(min_winter_t, fig_)


def histogram(df, fig_):
    # Create Fig and gridspec
    fig = plt.figure(figsize=(16, 10), dpi=80)
    grid = plt.GridSpec(4, 4, hspace=0.5, wspace=0.2)

    ax_main = fig.add_subplot(grid[:-1, :-1])
    ax_right = fig.add_subplot(grid[:-1, -1], xticklabels=[], yticklabels=[])
    ax_bottom = fig.add_subplot(grid[-1, 0:-1], xticklabels=[], yticklabels=[])

    int_dates = []
    for i, r in df.iterrows():
        jday = int(datetime.strftime(i, '%j'))
        if jday > 180:
            jday -= 365
        int_dates.append(jday)

    df['date'] = int_dates

    ax_main.scatter(df.date.values, df.min_t_F.values, s=25, alpha=.9,
                    cmap="tab10", edgecolors='gray', linewidths=.5)

    sns.kdeplot(data=df, ax=ax_right, y='min_t_F')
    ax_right.set_xlabel(None)
    ax_right.set_ylabel(None)
    ax_bottom.invert_yaxis()

    sns.kdeplot(data=df, ax=ax_bottom, x='date')
    ax_bottom.set_xlabel(None)
    ax_bottom.set_ylabel(None)

    ax_main.set(title='14480 Harpers Bridge Rd., Missoula, MT\nDate of Annual Minimum Temperatures', xlabel='Date',
                ylabel='Minimum Temperature (F)')
    ax_main.title.set_fontsize(20)

    plt.sca(ax_main)

    dates = [dt.strftime("%b-%d") for dt in date_range('1991-09-01', '1992-05-01') if dt.day == 1]
    dummy_x = np.linspace(-121, 121, 9)
    plt.xticks(dummy_x, dates)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.tick_params(axis='both', which='minor', labelsize=12)
    plt.tick_params(width=2, length=10)
    plt.tick_params(width=3, length=10)

    sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})

    plt.savefig(fig_)


def time_series(df, title='Mean Minimum Temperature', fig=None):
    from scipy.stats import sem

    # Plot
    plt.figure(figsize=(16, 10), dpi=80)
    if 'Temperature' in title:
        plt.ylabel("Temperature (F)", fontsize=16)
    elif 'Wind Speed' in title:
        plt.ylabel("Wind Speed (m/s)", fontsize=16)

    x = [d.date().strftime('%Y-%m-%d') for d in df['mean'].index]
    plt.plot(x, df['mean'], color="white", lw=2)
    plt.fill_between(x, df['mean'] - df['std'], df['mean'] + df['std'], color="#3F5D7D")

    ax = plt.gca()
    plt.gca().spines["top"].set_alpha(0)
    plt.gca().spines["bottom"].set_alpha(1)
    plt.gca().spines["right"].set_alpha(0)
    plt.gca().spines["left"].set_alpha(1)
    dates = [dt.strftime("%b-%d") for dt in df.index if dt.day == 1]
    dummy_x = [int(dt.strftime("%j")) for dt in df.index if dt.day == 1]
    plt.xticks(dummy_x, dates, fontsize=15)
    plt.setp(ax.get_xticklabels(), rotation=0)
    plt.title("14480 Harpers Bridge Rd., Missoula, MT\nDaily {}\n "
              "Standard Deviation Band".format(title), fontsize=20)
    # Axis limits
    s, e = plt.gca().get_xlim()
    plt.xlim(s, e - 2)
    # plt.ylim(4, 10)
    plt.savefig(fig)


if __name__ == '__main__':
    d = '/media/research/IrrigationGIS/climate/'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS/climate/'
    out_ = os.path.join(d, 'harpers', 'harpers_climate.csv')
    out_min_winter_t = os.path.join(d, 'harpers', 'harpers_climate_min_winter_temp.csv')
    # point_climatology(46.933, -114.199, out_)
    plot_climatology(out_)
    aspect = '/media/research/IrrigationGIS/Montana/iad/harpers_aspect.tif'
    out_aspect = '/media/research/IrrigationGIS/Montana/iad/harpers_aspect.png'
    # aspect_histogram(aspect, out_aspect)
# ========================= EOF ====================================================================
