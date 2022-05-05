import os

import numpy as np
from pandas import read_csv, concat
from shapely.geometry import Polygon
from geopandas import GeoDataFrame
import matplotlib.pyplot as plt
from scipy.stats import linregress


def concatenate_field_water_balance(extracts, out_filename, glob='glob', join_key='FID',
                                    template_geometry=None, fig=None):
    l = [os.path.join(extracts, x) for x in os.listdir(extracts) if glob in x]
    l.reverse()

    first = True
    for csv in l:
        splt = os.path.basename(csv).split('_')
        y, m = int(splt[-2]), int(splt[-1].split('.')[0])
        if first:
            df = read_csv(csv, index_col=join_key)
            # df.drop(columns=['irr'], inplace=True)
            df.columns = ['aet' if 'swb' in col else col for col in list(df.columns)]
            df.columns = ['{}_{}_{}'.format(col, y, m) for col in list(df.columns)]
            df.columns = [c.replace('ketchum', 'k') for c in df.columns]
            df.columns = [c.replace('senay', 's') for c in df.columns]

            df[join_key] = df.index
            first = False
        else:
            c = read_csv(csv, index_col=join_key)
            # c.drop(columns=['irr'], inplace=True)
            c.columns = ['aet' if 'swb' in col else col for col in list(c.columns)]
            c.columns = ['{}_{}_{}'.format(col, y, m) for col in list(c.columns)]
            df.columns = [c.replace('ketchum', 'k') for c in df.columns]
            df.columns = [c.replace('senay', 's') for c in df.columns]
            df = concat([df, c], axis=1)

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
        df = concat([df, t_gdf], axis=1)
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
    _bucket = 'gs://wudr'
    south = False
    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'
    _tables = 'Montana/st_mary/ssebop_data/depth'
    tables = os.path.join(root, _tables, '3MAY2022')
    o_file = os.path.join(root, _tables, 'et_comp_3MAY2022.csv')
    _glob = '3MAY2022'
    template_shape = os.path.join(root, 'Montana', 'st_mary',
                                  'All_Milk_River_Basin_Irrigation_NoFallow_wChannel_wgridcel_update.shp')
    concatenate_field_water_balance(tables, out_filename=o_file, glob=_glob, template_geometry=template_shape,
                                    join_key='FID_new')

# ========================= EOF ====================================================================
