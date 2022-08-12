import os
from copy import deepcopy

from pandas import read_csv, DataFrame, date_range
from geopandas import read_file
from datetime import date
from dateutil.rrule import rrule, DAILY
import numpy as np

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

etr_corr = {1: 1, 2: 1, 3: 1, 4: 1.39, 5: 1.43, 6: 1.38, 7: 1.40,
            8: 1.53, 9: 1.70, 10: 1.89, 11: 1, 12: 1}


def point_water_use(inshp, et_dir, out_csv):
    gdf = read_file(inshp)
    lat, lon = gdf.iloc[0]['geometry'].y, gdf.iloc[0]['geometry'].x

    grd = GridMet(variable='etr', start='2016-01-01', end='2021-12-31',
                  lat=lat, lon=lon)
    grd = grd.get_point_timeseries()

    r_index = date_range('2016-01-01', '2021-12-31', freq='D')

    fids = []
    for i, row in gdf.iterrows():
        fid = row['FID']
        fids.append(fid)
        col = 'etof_{}'.format(fid)
        file_ = os.path.join(et_dir, str(fid).rjust(3, '0'))
        etdf = read_csv(file_, index_col=0, infer_datetime_format=True, parse_dates=True)
        etdf = etdf.rename(columns={list(etdf.columns)[0]: col})
        etdf = etdf.reindex(r_index)
        etdf = etdf.interpolate()
        grd[col] = etdf[col]

    grd['mday'] = ['{}-{}'.format(x.month, x.day) for x in grd.index]
    target_range = date_range('2000-{}-{}'.format(4, 15),
                              '2000-{}-{}'.format(10, 15))
    accept = ['{}-{}'.format(x.month, x.day) for x in target_range]
    grd['mask'] = [1 if d in accept else 0 for d in grd['mday']]
    grd = grd[grd['mask'] == 1]

    cdf = deepcopy(grd)
    cdf['etr_c'] = [np.nan for _ in cdf.index]
    for i, r in grd.iterrows():
        cfactor = etr_corr[i.month]
        cdf.loc[i, 'etr_c'] = r['etr'] / cfactor

    # TODO: bring in OpenET data and resample to daily according to etr
    fid_itpe = {4: 'property', 193: 'sprinkler', 190: 'pivot', 189: 'flood'}
    for f in fids:
        cdf['et_{}'.format(fid_itpe[f])] = cdf['etof_{}'.format(f)] * cdf['etr_c'] / 25.4
    cdf.drop(columns=['mday', 'mask'])
    cdf[['etr', 'etr_c']] /= 25.4
    first = False

    grd = deepcopy(cdf)

    a = date(2021, 4, 15)
    b = date(2021, 10, 15)
    years = [x for x in range(2016, 2022)]

    vars = ['etr_c', 'et_property', 'et_flood', 'et_sprinkler', 'et_pivot']
    names = ['Reference ET [in]',
             'ET Property Planting Area',
             'Nearby Flood Irrigation',
             'Nearby Sprinkler Irrigation',
             'Nearby Pivot Irrigation']
    dct = {v: {'mean': [], 'std': [], 'date': [], 'min': [], 'max': []} for v in vars}

    for i, dt in enumerate(rrule(DAILY, dtstart=a, until=b)):
        for i, v in enumerate(dct.keys()):
            dates = [date(year, dt.month, dt.day).strftime('%Y-%m-%d') for year in years]
            vals = grd.loc[dates, v].values

            mean_ = vals.mean()
            std_ = vals.std()
            min_ = vals.min()
            max_ = vals.max()

            dct[v]['mean'].append(mean_)
            dct[v]['std'].append(std_)
            dct[v]['min'].append(min_)
            dct[v]['max'].append(max_)
            dct[v]['date'].append(dt)

    for k, v in dct.items():
        _file = out_csv.replace('.csv', '_{}.csv'.format(k))
        DataFrame(data={'mean': v['mean'], 'std': v['std']},
                  index=v['date']).to_csv(_file, float_format='%.2f', index_label='date')

    for v, t in zip(vars, names):
        _file = out_csv.replace('.csv', '_{}.csv'.format(v))
        df = read_csv(_file, parse_dates=True, index_col='date', infer_datetime_format=True)
        fig_ = os.path.join(_file.replace('.csv', '.png'.format(v)))
        time_series(df, t, fig_)


def time_series(df, title='Mean Minimum Temperature', fig=None):
    from scipy.stats import sem

    # Plot
    plt.figure(figsize=(16, 10), dpi=80)

    x = [d.date().strftime('%Y-%m-%d') for d in df['mean'].index]
    plt.plot(x, df['mean'], color="white", lw=2)
    plt.fill_between(x, df['mean'] - df['std'], df['mean'] + df['std'], color="#3F5D7D")

    ax = plt.gca()
    plt.gca().spines["top"].set_alpha(0)
    plt.gca().spines["bottom"].set_alpha(1)
    plt.gca().spines["right"].set_alpha(0)
    plt.gca().spines["left"].set_alpha(1)
    dates = [dt.strftime("%b-%d") for dt in df.index if dt.day == 1]
    dummy_x = [i for i, dt in enumerate(df.index) if dt.day == 1]
    plt.xticks(dummy_x, dates, fontsize=15)
    plt.setp(ax.get_xticklabels(), rotation=0)
    plt.title("14480 Harpers Bridge Rd., Missoula, MT\nDaily {}\n 2016-2021 \n"
              "Min/Max Band".format(title), fontsize=20)
    # Axis limits
    # s, e = plt.gca().get_xlim()
    # plt.xlim(s, e - 2)
    # plt.ylim(4, 10)
    plt.savefig(fig)


if __name__ == '__main__':
    d = '/media/research/IrrigationGIS/'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS/'

    out_ = os.path.join(d, 'climate', 'harpers', 'harpers_climate.csv')
    out_min_winter_t = os.path.join(d, 'climate', 'harpers', 'harpers_water_use.csv')
    r = os.path.join(d, 'Montana', 'iad')
    et_dir_ = os.path.join(r, 'harper_fields_etof')
    _shp = os.path.join(r, 'Harpers_Itype_Centroids.shp')
    out_c = os.path.join(r, 'harpers_et.csv')
    point_water_use(_shp, et_dir_, out_c)

# ========================= EOF ====================================================================
