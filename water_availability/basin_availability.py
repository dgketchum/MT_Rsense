import os
import json
from copy import deepcopy

import pandas as pd
from dateutil.parser import ParserError
import fiona
import geopandas as gpd
import numpy as np
from pandas import read_csv, concat, DataFrame, to_datetime, date_range, DatetimeIndex, errors
import matplotlib

import matplotlib.pyplot as plt

from utils.hydrograph import read_hydrograph
from utils.hydrograph import get_station_flows

BASINS = {'38H': (None, None, None),
          '39E': (29756, None, None),
          '39F': (29783, None, None),
          '39G': (32036, None, None),
          '39H': (32486, None, None),
          '40A': (31100, '06126500', None),
          '40C': (31100, None, None),
          '40E': (30557, None, None),
          '40EJ': (30488, None, None),
          '40G': (47863, None, None),
          '40H': (47849, '06139500', None),
          '40I': (47913, None, None),
          '40J': (47836, None, None),
          '40M': (30946, '06167500', None),
          '40N': (47794, None, None),
          '40O': (48353, None, None),
          '40P': (48288, None, None),
          '40Q': (48304, None, None),
          '40R': (48205, '06185110', None),
          '40S': (48166, '06185500', None),
          '41B': (28826, '06023100', None),
          '41C': (28444, '06023000', None),
          '41D': (28879, '06026420', None),
          '41G': (28994, '06036650', None),
          '41F': (28592, '06042500', None),
          '41H': (28712, '06052500', None),
          '41J': (30762, '06077500', None),
          '41L': (31203, None, None),
          '41M': (30985, None, None),
          '41O': (31451, None, None),
          '41P': (31451, None, None),
          '41QJ': (30774, None, None),
          '41R': (31427, None, None),
          '41S': (30562, None, None),
          '41T': (31407, None, None),
          '42B': (31484, None, None),
          '42C': (31452, '06308500', None),
          '42I': (29685, None, None),
          '42J': (29845, '06326500', None),
          '42KJ': (32022, '06309000', 'subtract 06308500'),
          '42M': (33557, '06329590', None),
          '43A': (28753, '06195600', None),
          '43BJ': (29049, '06200000', None),
          '43BV': (29056, None, None),
          '43C': (29011, None, None),
          '43D': (29018, None, None),
          '43E': (29556, None, None),
          '43O': (31494, '06294000', None),
          '43QJ': (29019, None, None),
          # '76B': (51554, '12304500', None),
          '76C': (51625, '12302055', None),
          '76D': (51267, '12305000', None),
          '76E': (51480, '12334510', None),
          '76F': (51481, '12340000', None),
          '76G': (52800, '12334550', None),
          '76GJ': (51656, '12331590', None),
          '76H': (52817, '12352500', None),
          # '76HF': (51513, None, None),
          '76K': (51544, '12370000', None),
          '76L': (51335, '12388700', None),
          '76LJ': (51544, '12372000', None),
          '76M': (52810, '12354500', None),
          '76N': (51319, '12391950', None)}

POD_COLS = ['PER_USE', 'PER_DIV']
POU_COLS = ['MAXACRES', 'FLWRTGPM', 'FLWRTCFS', 'VOL', 'ACREAGE']


def _to_datetime(p, start_end='start'):
    if not p:
        return None
    else:
        if start_end == 'start':
            idx = 0
        else:
            idx = 1
        try:
            return to_datetime('{}/2019'.format(p.split(' to ')[idx][:5]))
        except ParserError:
            return None
        except ValueError:
            return None


def _time_delta(s, e):
    return abs((e - s).days) + 1


def write_features(shp, out_dir):
    with fiona.open(shp, 'r') as src:
        meta = src.meta
        for f in src:
            name = f['properties']['BASINNUM']
            file_ = os.path.join(out_dir, '{}.shp'.format(name))
            with fiona.open(file_, 'w', **meta) as dst:
                dst.write(f)
                print(file_)


def merge_gridded(extracts, out_dir, glob='glob', join_key='BASINNUM'):
    missing, missing_ct, processed_ct = [], 0, 0

    l = [os.path.join(extracts, x) for x in os.listdir(extracts) if glob in x]
    l.reverse()

    first = True
    for csv in l:
        splt = os.path.basename(csv).split('_')
        y, m = int(splt[-2]), int(splt[-1].split('.')[0])

        try:
            if first:
                df = read_csv(csv, index_col=join_key)
                df.drop(columns=['BASINNAME'], inplace=True)
                df.columns = ['{}_{}_{}'.format(col, y, m) for col in list(df.columns)]
                first = False
            else:
                c = read_csv(csv, index_col=join_key)
                c.drop(columns=['BASINNAME'], inplace=True)
                cols = list(c.columns)
                c.columns = ['{}_{}_{}'.format(col, y, m) for col in cols]
                df = concat([df, c], axis=1)

        except errors.EmptyDataError:
            print('{} is empty'.format(csv))
            pass
    yr_ = [int(x.split('_')[1]) for x in list(df.columns) if 'cc' in x]
    year_start = min(yr_)
    year_end = max(yr_)
    df['BASINNUM'] = df.index

    dfd = df.to_dict(orient='records')
    s, e = '{}-01-01'.format(year_start), '{}-12-31'.format(year_end)
    idx = DatetimeIndex(date_range(s, e, freq='M'))
    idx = DatetimeIndex([i for i in idx if i.month in [x for x in range(4, 11)]])
    months = [(idx.year[x], idx.month[x]) for x in range(idx.shape[0])]
    for d in dfd:
        try:
            sta = d['BASINNUM']

            # handle pre-1991 and off-growing season KeyError
            irr, cc, et = [], [], []
            for y, m in months:
                try:
                    cc_, et_ = d['cc_{}_{}'.format(y, m)], d['et_{}_{}'.format(y, m)]
                    cc.append(cc_)
                    et.append(et_)
                except KeyError:
                    cc.append(np.nan)
                    et.append(np.nan)

                try:
                    irr_ = d['irr_{}_{}'.format(y, m)]
                    irr.append(irr_)
                except KeyError:
                    irr.append(np.nan)

            irr = irr, 'irr'
            cc = cc, 'cc'
            et = et, 'et'

            if not np.any(irr[0]):
                print(sta, 'no irrigation')
                continue

            ppt = [d['ppt_{}_{}'.format(y, m)] for y, m in months], 'ppt'
            etr = [d['etr_{}_{}'.format(y, m)] for y, m in months], 'etr'
            sm = [d['swb_aet_{}_{}'.format(y, m)] for y, m in months], 'swb'
            ppt_irr = [d['ppt_irr_{}_{}'.format(y, m)] for y, m in months], 'ppt_irr'
            recs = DataFrame(dict([(x[1], x[0]) for x in [irr, et, cc, ppt, etr, sm, ppt_irr]]), index=idx)

            h = recs

            file_name = os.path.join(out_dir, '{}.csv'.format(sta))
            h.to_csv(file_name)
            processed_ct += 1

            print(file_name)

        except FileNotFoundError:
            missing_ct += 1
            print(sta, 'not found')
            missing.append(sta)

    print(processed_ct, 'processed')
    print(missing_ct, 'missing')


def compile_segment_hydrographs(hydrograph_dir, out_csv):
    target_segments = [str(v[0]) for k, v in BASINS.items()]
    dirs = [os.path.join(hydrograph_dir, x) for x in os.listdir(hydrograph_dir)]
    dirs = [x for x in dirs if os.path.isdir(x)]
    files = [os.path.join(d, 'model_output', 'seg_outflow.csv') for d in dirs]
    first = True
    for f in files:
        if first:
            df = read_csv(f, index_col='time', infer_datetime_format=True,
                          parse_dates=True).rename(columns={'time': 'Date'})
            keep = [c for c in df.columns if c in target_segments]
            df = df[keep]
            first = False
        else:
            csv = read_csv(f, index_col='time', infer_datetime_format=True,
                           parse_dates=True).rename(columns={'time': 'Date'})
            keep = [c for c in csv.columns if c in target_segments and c not in df.columns]
            csv = csv[keep]
            df = concat([df, csv], axis=1)

    df = df.loc['1989-01-01': '2018-12-31']
    df.to_csv(out_csv)


def compile_availability_data(wr_dir, model_hydrograph, demands_data, crop_cons_data):
    hydro = read_csv(model_hydrograph, index_col='time', infer_datetime_format=True,
                     parse_dates=True)
    dtr = date_range('2019-01-01', '2019-12-31', freq='D')

    for basin, (segment, gage, _) in BASINS.items():

        if not segment:
            continue

        print('basin {}, gage {}, segment {}'.format(basin, gage, segment))

        basin_dir = os.path.join(demands_data, basin)
        if not os.path.isdir(basin_dir):
            os.mkdir(basin_dir)

        gs_file = os.path.join(basin_dir, 'gage_data.csv')
        demands_file = os.path.join(basin_dir, 'legal_demands.csv')
        prms_file = os.path.join(basin_dir, 'prms_hydrograph.csv')
        grid_file = os.path.join(basin_dir, 'cc_hydrograph.csv')

        if all([os.path.exists(x) for x in [gs_file, demands_file, prms_file, grid_file]]):
            continue

        if not os.path.exists(grid_file):
            years = range(1987, 2022)
            gdf_idx = DatetimeIndex(date_range('2019-04-01', '2019-10-31', freq='M'))
            days_per_month = np.array([30, 31, 30, 31, 31, 30, 31])
            gdf = DataFrame(columns=years, index=gdf_idx)

            cc_input = os.path.join(crop_cons_data, '{}.csv'.format(basin))

            try:
                edf = read_hydrograph(cc_input)
            except FileNotFoundError:
                continue

            irr_areas = edf['irr'].copy(deep=True)
            edf = edf.divide(edf['irr'], axis=0)
            edf['dnrc_cc'] = edf['et'] - edf['ppt_irr'] * 0.93
            april_mod_idx = [i for i in edf.index if i.month == 4]
            edf.loc[april_mod_idx, 'dnrc_cc'] -= 0.0254

            for idx, r in edf['dnrc_cc'].iteritems():
                if r < 0:
                    next_ = idx + pd.offsets.MonthEnd(n=1)
                    try:
                        edf.at[next_, 'dnrc_cc'] += r
                    except KeyError:
                        next_ = idx + pd.offsets.MonthEnd(n=6)
                        if next_ > edf.index[-1]:
                            edf.at[idx, 'dnrc_cc'] = 0.
                            break
                        else:
                            edf.at[next_, 'dnrc_cc'] += r
                    edf.at[idx, 'dnrc_cc'] = 0.

            edf = edf.multiply(irr_areas, axis=0)
            for yr in years:
                grid_data = [r['dnrc_cc'] for i, r in edf.iterrows() if i.year == yr]
                try:
                    gdf.loc[:, yr] = grid_data
                except ValueError:
                    continue
            gdf = gdf.divide((days_per_month * 2446.58), axis=0)
            gdf['min'] = gdf.min(axis=1)
            gdf['max'] = gdf.max(axis=1)
            gdf['median'] = gdf.median(axis=1)
            gdf.to_csv(grid_file)

        if not os.path.exists(demands_file):
            pou_file = os.path.join(wr_dir, '{}_pou.shp'.format(basin))
            pod_file = os.path.join(wr_dir, '{}_pod.shp'.format(basin))

            pou = gpd.read_file(pou_file)
            pou = pou[(pou['PURPOSE'] == 'IRRIGATION') & (pou['WRSTATUS'] == 'ACTIVE')]
            pou.drop(columns=['POUID'], inplace=True)
            pou = pou.drop_duplicates(subset=['WRKEY'])

            pod = gpd.read_file(pod_file)
            pod.drop(columns=['DIVNUMBER'], inplace=True)
            pod = pod.drop_duplicates(subset=['WRKEY'])
            pod = pod[pod['WR_STATUS'] == 'ACTV']

            pou_keys = [x for x in pou['WRKEY']]
            pod['has_pou'] = [1 if x in pou_keys else 0 for x in pod['WRKEY']]
            pod = pod[pod['has_pou'] == 1]

            pod_keys = [x for x in pod['WRKEY']]
            pou['has_pod'] = [1 if x in pod_keys else 0 for x in pou['WRKEY']]
            pou = pou[pou['has_pod'] == 1]

            df = pou.set_index('WRKEY').join(pod.set_index('WRKEY'), lsuffix='_pou', rsuffix='_pod')
            df = df[POD_COLS + POU_COLS + ['ABSTRACT_pod']]

            df['start'] = df.apply(lambda row: _to_datetime(row['PER_USE'], 'start'), axis=1)
            df['end'] = df.apply(lambda row: _to_datetime(row['PER_USE'], 'end'), axis=1)
            df['days'] = df.apply(lambda row: _time_delta(row['end'], row['start']), axis=1)

            df['VOL_to_CFS'] = df['VOL'] / (df['days'] * 1.98)
            fill_idx = df.loc[df['VOL_to_CFS'] == 0.0, 'VOL_to_CFS']
            df.loc[fill_idx.index, 'VOL_to_CFS'] = df.loc[fill_idx.index, 'FLWRTCFS']

            df = df.drop(index=df[np.isnan(df['VOL_to_CFS'])].index)

            demand_hydrograph = []
            for dt in dtr:
                daily_demand = 0.0
                for i, r in df.iterrows():
                    if r['start'] <= dt <= r['end']:
                        daily_demand += r['VOL_to_CFS']
                demand_hydrograph.append(daily_demand)

            demand_df = DataFrame(columns=['cfs'], index=dtr, data=demand_hydrograph)
            demand_df.to_csv(demands_file)

        doy_idx = [x for x in range(365)]

        if not os.path.exists(prms_file):
            years = range(1991, 2018)
            prms_simulated = DataFrame(index=doy_idx, columns=years)
            try:
                prms_hydro = hydro[str(segment)]
            except KeyError:
                continue

            for yr in years:
                prms_data = [r for i, r in prms_hydro.iteritems() if
                             i.year == yr and not (i.day == 29 and i.month == 2)]
                prms_simulated.loc[:, yr] = prms_data

            prms_simulated['min'] = prms_simulated.min(axis=1)
            prms_simulated['max'] = prms_simulated.max(axis=1)
            prms_simulated['median'] = prms_simulated.median(axis=1)
            prms_simulated.to_csv(prms_file)

        years = range(1980, 2022)
        gs_obs = DataFrame(index=doy_idx, columns=years)
        if not os.path.exists(gs_file) and gage:
            gs_hydro = get_station_instantaneous_flows('1980-01-01', '2021-12-31', gage)
            if isinstance(gs_hydro, type(None)):
                continue
            for yr in years:
                gs_record = [r for i, r in gs_hydro.iteritems() if
                             i.year == yr and not (i.day == 29 and i.month == 2)]
                gs_obs.loc[:, yr] = gs_record

            gs_obs['min'] = gs_obs.min(axis=1)
            gs_obs['max'] = gs_obs.max(axis=1)
            gs_obs['median'] = gs_obs.median(axis=1)
            gs_obs.to_csv(gs_file)


def get_hydrograph_comparison(basin_data, shape, out_shape):
    over = []
    under = []
    for basin, (segment, obs) in BASINS.items():

        if not segment and not obs:
            continue

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
        prms_file = os.path.join(basin_dir, 'prms_hydrograph.csv')
        if not os.path.exists(prms_file):
            continue
        prms_df = read_hydrograph(prms_file)
        prms_m = prms_df.groupby([prms_df.index.month]).mean()
        legal = read_hydrograph(os.path.join(basin_dir, 'legal_demands.csv'))
        legal_m = legal.groupby([legal.index.month]).mean()
        grid_df = read_hydrograph(os.path.join(basin_dir, 'cc_hydrograph.csv'))
        grid_m = grid_df.mean(axis=1)
        grid_df[grid_df.values == 0] = np.nan


def compile_to_single_file(basin_data, shape, out_csv):
    ct = 0
    cols = ['date', 'basin', 'name', 'legal', 'prms_min', 'prms_med',
            'prms_max', 'cc_med', 'obs_min', 'obs_med', 'obs_max']
    idx = DatetimeIndex(date_range('2019-01-01', '2019-12-31', freq='D'))
    df = pd.DataFrame(columns=cols)

    names = gpd.read_file(shape)
    names = names[['BASINNUM', 'BASINNAME']]
    names = {r['BASINNUM']: r['BASINNAME'] for i, r in names.iterrows()}

    for basin, (segment, obs, _) in BASINS.items():

        if not segment and not obs:
            continue

        basin_col = [basin for _ in range(365)]
        name = [names[basin] for _ in range(365)]

        basin_dir = os.path.join(basin_data, basin)
        if not os.path.exists(basin_dir):
            continue

        gs_file = os.path.join(basin_dir, 'gage_data.csv')
        if os.path.exists(gs_file):
            gs_obs = read_hydrograph(gs_file)
            gs_obs.index = idx
        else:
            gs_obs = {k: [np.nan for _ in range(365)] for k in ['min', 'median', 'max']}

        prms_file = os.path.join(basin_dir, 'prms_hydrograph.csv')
        if not os.path.exists(prms_file):
            continue

        prms_df = read_hydrograph(prms_file)
        prms_df.index = idx
        legal = read_hydrograph(os.path.join(basin_dir, 'legal_demands.csv'))
        grid_df = read_hydrograph(os.path.join(basin_dir, 'cc_hydrograph.csv'))
        grid_df = grid_df['median'].reindex(idx)
        s, e = ('2019-04-01', '2019-11-01')
        grid_df.loc[s: e] = grid_df.loc[s: e].ffill().bfill()

        if prms_df['median'].max() < legal['cfs'].max():
            print(basin, True)
            ct += 1
        else:
            print(basin, False)

        appnd = {'basin': basin_col,
                 'name': name,
                 'date': idx,
                 'legal': legal['cfs'],
                 'prms_min': prms_df['min'],
                 'prms_med': prms_df['median'],
                 'prms_max': prms_df['max'],
                 'cc_med': grid_df,
                 'obs_min': gs_obs['min'],
                 'obs_med': gs_obs['median'],
                 'obs_max': gs_obs['max']}

        d = pd.DataFrame(columns=cols, index=idx, data=appnd)

        df = df.append(d, ignore_index=True)

    df.to_csv(out_csv)
    print(ct, 'over-allocated')


if __name__ == '__main__':
    matplotlib.use('TkAgg')

    d = '/media/research/IrrigationGIS/Montana'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS/Montana'

    cc_dir = os.path.join(d, 'water_availability', 'consumptive_use_data')
    ee_data = os.path.join(cc_dir, 'ee_extracts', '30OCT2022')
    cc_data = os.path.join(cc_dir, 'basin_cc_series', '30OCT2022')
    # merge_gridded(ee_data, cc_data, glob='basins_30OCT2022')

    segment_dir_ = os.path.join(d, 'water_availability', 'segment_hydrographs')
    master_hydro = os.path.join(d, 'water_availability', 'segment_hydrographs', 'compiled_hydrographs.csv')
    # compile_segment_hydrographs(segment_dir_, master_hydro)

    basin_wr_pod = os.path.join(d, 'water_availability', 'basin_wr_gdb')
    demands = os.path.join(d, 'water_availability', 'daily_demands_data')
    cc_data = os.path.join(cc_dir, 'basin_cc_series', '30OCT2022')
    # compile_availability_data(basin_wr_pod, master_hydro, demands, cc_data)

    shp = os.path.join(d, 'AdminBasinsShapefile', 'BasinBoundaries_USGSALB.shp')
    out_ = os.path.join(d, 'water_availability', 'master_hydrograph.csv')
    basin_data_ = os.path.join(d, 'water_availability', 'daily_demands_data')
    # compile_to_single_file(basin_data_, shp, out_)

    with open('/home/dgketchum/Downloads/gages_assessment/admin_basin_gaged_outlets.json', 'w') as fp:
        fp.write(json.dumps(BASINS))

# ========================= EOF ====================================================================
