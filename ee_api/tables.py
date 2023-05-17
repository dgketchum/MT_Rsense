import os

import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from geopandas import GeoDataFrame
import matplotlib.pyplot as plt
from scipy.stats import linregress


def write_et_data(gridded_dir, summary, csv_all, csv_fields, start_year=1987,
                  end_year=2021, glob='glob',
                  join_key='FID'):
    missing, missing_ct, processed_ct = [], 0, 0

    et_data = os.path.join(gridded_dir, 'et')
    l = [os.path.join(et_data, x) for x in os.listdir(et_data) if glob in x]
    irr_data = os.path.join(gridded_dir, 'irr')
    l.reverse()

    first = True
    for csv in l:
        splt = os.path.basename(csv).split('_')
        y, m = int(splt[-2]), int(splt[-1].split('.')[0])

        if first:
            df = pd.read_csv(csv, index_col=join_key)
            o_cols = list(df.columns)
            df.columns = ['{}_{}_{}'.format(col, y, m) for col in list(df.columns)]
            first = False
        else:
            c = pd.read_csv(csv, index_col=join_key)
            cols = list(c.columns)
            c.columns = ['{}_{}_{}'.format(col, y, m) for col in cols]
            df = pd.concat([df, c], axis=1)

        if m == 4:
            f = os.path.join(irr_data, 'tongue_irr_9MAY2023_{}_4.csv'.format(y))
            c = pd.read_csv(f, index_col=join_key)
            match = [i for i in c.index if i in df.index]
            df.loc[match, 'irr_{}_{}'.format(y, m)] = c.loc[match, 'irr']

    df.to_csv(csv_all)
    df = df.copy()
    df = df.fillna(0.0)
    dfd = df.T.to_dict(orient='dict')
    s, e = '{}-01-01'.format(start_year), '{}-12-31'.format(end_year)
    idx = pd.DatetimeIndex(pd.date_range(s, e, freq='M'))

    sdf = pd.DataFrame(columns=o_cols, index=df.index)

    months = [(idx.year[x], idx.month[x]) for x in range(idx.shape[0])]

    for i, d in dfd.items():
        irr, cc, et, ietr, ept = [], [], [], [], []
        for y, m in months:
            try:
                cc_, et_, ietr_ = d['cc_{}_{}'.format(y, m)], d['et_{}_{}'.format(y, m)], d['ietr_{}_{}'.format(y, m)]
                ept_ = d['eff_ppt_{}_{}'.format(y, m)]
                cc.append(cc_)
                et.append(et_)
                ietr.append(ietr_)
                ept.append(ept_)

            except KeyError:
                cc.append(np.nan)
                et.append(np.nan)
                ietr.append(np.nan)
                ept.append(np.nan)

            try:
                irr_ = d['irr_{}_{}'.format(y, m)]
                irr.append(irr_)
            except KeyError:
                irr.append(np.nan)

        irr = irr, 'irr'
        cc = cc, 'cc'
        et = et, 'et'
        ietr = ietr, 'ietr'
        ept = ept, 'eff_ppt'

        ppt = [d['ppt_{}_{}'.format(y, m)] for y, m in months], 'ppt'
        etr = [d['etr_{}_{}'.format(y, m)] for y, m in months], 'etr'

        recs = pd.DataFrame(dict([(x[1], x[0]) for x in [irr, et, cc, ppt, etr, ietr, ept]]), index=idx)
        _file = os.path.join(csv_fields, '{}.csv'.format(i))
        # recs.to_csv(_file)
        # print(_file)
        recs = recs[['irr', 'et', 'cc', 'ppt', 'etr', 'ietr', 'eff_ppt']]
        annual = recs.resample('A')
        agg_funcs = {'irr': 'mean', 'cc': 'sum', 'eff_ppt': 'sum', 'et': 'sum',
                     'ppt': 'sum', 'etr': 'sum', 'ietr': 'sum'}
        annual = annual.agg(agg_funcs)
        sdf.loc[i] = annual.mean(axis=0)

    sdf.to_csv(summary)


def concatenate_field_water_balance(extracts, out_filename, glob='glob', join_key='FID',
                                    template_geometry=None, fig=None):
    l = [os.path.join(extracts, x) for x in os.listdir(extracts) if glob in x]
    l.reverse()

    first = True
    for csv in l:
        splt = os.path.basename(csv).split('_')
        y, m = int(splt[-2]), int(splt[-1].split('.')[0])
        if first:
            df = pd.read_csv(csv, index_col=join_key)
            # df.drop(columns=['irr'], inplace=True)
            df.columns = ['aet' if 'swb' in col else col for col in list(df.columns)]
            df.columns = ['{}_{}_{}'.format(col, y, m) for col in list(df.columns)]
            df.columns = [c.replace('ketchum', 'k') for c in df.columns]
            df.columns = [c.replace('senay', 's') for c in df.columns]

            df[join_key] = df.index
            first = False
        else:
            c = pd.read_csv(csv, index_col=join_key)
            # c.drop(columns=['irr'], inplace=True)
            c.columns = ['aet' if 'swb' in col else col for col in list(c.columns)]
            c.columns = ['{}_{}_{}'.format(col, y, m) for col in list(c.columns)]
            df.columns = [c.replace('ketchum', 'k') for c in df.columns]
            df.columns = [c.replace('senay', 's') for c in df.columns]
            df = pd.concat([df, c], axis=1)

    df.to_csv(out_filename)
    if fig:
        df.dropna(axis=0, inplace=True)
        senay = df[[c for c in df.columns if '_s_' in c]]
        ketchum = df[[c for c in df.columns if '_k_' in c]]
        slope, intercept, r_value, p_value, std_err = linregress(senay.values.ravel(),
                                                                 ketchum.values.ravel())
        fig = plt.figure()
        ax = fig.add_subplot(111)
        c = ['blue', 'green', 'yellow', 'pink', 'red']
        m = [x for x in range(5, 11)]
        for month, color in zip(m, c):
            s, k = senay['et_s_2015_{}'.format(month)], ketchum['et_k_2015_{}'.format(month)]
            ax.scatter(s.values.ravel(), k.values.ravel(), alpha=0.2, label='Month {}'.format(month),
                       facecolor=None, edgecolor=c, s=10)

        plt.plot([0, 0.25], [0, 0.25], c='k')
        txt = ('Slope: {:.3f}\nIntercept: {:.3f}\nr: {:.3f}'.format(slope, intercept, r_value))
        plt.text(0.18, 0.01, txt)
        plt.suptitle('SSEBop Comparison May - Sep 2015\nET in meters per month')
        plt.xlabel('Senay et al 2022')
        plt.ylabel('Ketchum')
        plt.legend()
        plt.savefig(fig)
        plt.close()

    if template_geometry:
        t_gdf = GeoDataFrame.from_file(template_geometry).to_crs('epsg:5071')
        t_gdf.index = t_gdf['FID_new']
        t_gdf.drop(columns=['FID_new'], inplace=True)
        geo = t_gdf['geometry']
        df = pd.concat([df, t_gdf], axis=1)
        gpd = GeoDataFrame(df, crs='epsg:5071', geometry=geo)
        gpd.drop(columns=['FID_new'], inplace=True)
        gpd.to_file(out_filename.replace('.csv', '.shp'))


def to_polygon(j):
    if not isinstance(j, list):
        return np.nan
    try:
        return Polygon(j[0])
    except ValueError:
        return np.nan
    except TypeError:
        return np.nan
    except AssertionError:
        return np.nan


if __name__ == '__main__':
    gis = '/media/research/IrrigationGIS/Montana/tongue'
    csv_ = os.path.join(gis, 'gridded_data')
    csv_fields = os.path.join(gis, 'fields')
    csv_all = os.path.join(gis, 'all_data.csv')
    csv_out = os.path.join(gis, 'summary_data.csv')
    write_et_data(csv_, csv_out, csv_all, csv_fields, glob='tongue_fields_9MAY2023', join_key='FID')

# ========================= EOF ====================================================================
