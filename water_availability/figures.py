import os
import matplotlib
from pandas import read_csv, DatetimeIndex, date_range

import matplotlib.pyplot as plt
from basin_availability import BASINS


def hydrograph_comparison(basin_data, figs=None):
    for basin, (segment, obs) in BASINS.items():
        if basin != '40A':
            continue
        idx = DatetimeIndex(date_range('2019-01-01', '2019-12-31', freq='D'))
        if obs:
            try:
                gs_file = os.path.join(basin_data, basin, 'usgs_hydrograph.csv')
                gs_obs = read_csv(gs_file, index_col=0, infer_datetime_format=True,
                                  parse_dates=True)
                gs_obs.index = idx
                plt.plot(gs_obs.index, gs_obs['median'], color='b', lw=1)
                plt.fill_between(gs_obs.index, gs_obs['min'], gs_obs['max'], color='b', alpha=0.1)
            except FileNotFoundError:
                continue

        prms_file = os.path.join(basin_data, basin, 'prms_hydrograph.csv')

        try:
            prms_simulated = read_csv(prms_file, index_col=0, infer_datetime_format=True,
                                      parse_dates=True)
            prms_simulated.index = idx
        except FileNotFoundError:
            continue

        plt.plot(prms_simulated.index, prms_simulated['median'], color='r', lw=1)
        plt.fill_between(prms_simulated.index, prms_simulated['min'], prms_simulated['max'],
                         color='r', alpha=0.1)

        legal_file = os.path.join(basin_data, basin, 'legal_demands.csv')
        legal = read_csv(legal_file, index_col=0, infer_datetime_format=True,
                         parse_dates=True)
        legal = legal.reindex(idx)
        plt.plot(legal.index, legal['cfs'], color='k', lw=1)
        plt.title(basin)
        plt.yscale('log')
        plt.ylim([10, prms_simulated['max'].max()])

        if figs:
            f_name = os.path.join(figs, '{}.png'.format(basin))
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
    hydrograph_comparison(demands, figs=None)
# ========================= EOF ====================================================================
