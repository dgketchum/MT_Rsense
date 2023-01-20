import os
import matplotlib
import numpy as np
from pandas import read_csv, DatetimeIndex, date_range
import geopandas as gpd

import matplotlib.pyplot as plt
from basin_availability import BASINS
from utils.hydrograph import read_hydrograph


def hydrograph_comparison(basin_data, figs=None, shape=None):
    for basin, (segment, obs, _) in BASINS.items():

        if not segment and not obs:
            continue

        if shape:
            names = gpd.read_file(shape)
            names = names[['BASINNUM', 'BASINNAME']]
            names = {r['BASINNUM']: r['BASINNAME'] for i, r in names.iterrows()}

        idx = DatetimeIndex(date_range('2019-01-01', '2019-12-31', freq='D'))

        basin_dir = os.path.join(basin_data, basin)
        if not os.path.exists(basin_dir):
            continue

        gs_file = os.path.join(basin_dir, 'gage_data.csv')
        if os.path.exists(gs_file):
            gs_obs = read_hydrograph(gs_file)
            gs_obs.index = idx
            plt.plot(gs_obs.index, gs_obs['median'], color='b', lw=1, label='Observed Flow')
            plt.fill_between(gs_obs.index, gs_obs['min'], gs_obs['max'], color='b', alpha=0.1)
            gs_max = gs_obs.values.max()
        else:
            gs_max = 0

        prms_file = os.path.join(basin_dir, 'prms_hydrograph.csv')
        if not os.path.exists(prms_file):
            continue
        prms_df = read_hydrograph(prms_file)
        prms_df.index = idx
        plt.plot(prms_df.index, prms_df['median'], color='r', lw=1, label='Naturalized Flow')
        plt.fill_between(prms_df.index, prms_df['min'], prms_df['max'],
                         color='r', alpha=0.1)

        legal = read_hydrograph(os.path.join(basin_dir, 'legal_demands.csv'))
        legal = legal.reindex(idx)
        plt.plot(legal.index, legal['cfs'], color='k', lw=1, label='Legal Demands')
        plt.yscale('log')

        grid_df = read_hydrograph(os.path.join(basin_dir, 'cc_hydrograph.csv'))
        grid_df[grid_df.values == 0] = np.nan
        plt.plot(grid_df.index, grid_df['median'], color='green', lw=1, label='Crop Consumption')
        plt.title('{} {}'.format(basin, names[basin]))
        plt.legend(loc='best')

        try:
            max_val = np.nanmax([gs_max, prms_df.values.max(), legal.values.max(), grid_df.values.max()])
            plt.ylim([1, max_val])
        except ValueError:
            pass

        if figs:
            f_name = os.path.join(figs, '{}_{}.png'.format(basin, names[basin]))
            plt.savefig(f_name)
            plt.close()
        else:
            plt.show()


if __name__ == '__main__':
    matplotlib.use('TkAgg')

    d = '/media/research/IrrigationGIS/Montana'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS/Montana'

    demands = os.path.join(d, 'water_availability', 'daily_demands_data')
    figs_ = os.path.join(d, 'water_availability', 'hydrograph_figures')
    figs_ = os.path.join(d, 'water_availability', 'hydrograph_figures')
    shp = os.path.join(d, 'AdminBasinsShapefile', 'BasinBoundaries_USGSALB.shp')
    hydrograph_comparison(demands, figs=figs_, shape=shp)
# ========================= EOF ====================================================================
