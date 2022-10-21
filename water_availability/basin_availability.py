import os
from copy import deepcopy
from dateutil.parser import ParserError
import fiona
import geopandas as gpd
import numpy as np
from pandas import read_csv, concat, DataFrame, to_datetime, date_range, DatetimeIndex, errors
import matplotlib

import matplotlib.pyplot as plt

from utils.hydrograph import read_hydrograph
from utils.hydrograph import get_station_daily_data

# DNRC BASIN: (PRMS_SEGMENT, USGS GAGE (if applicable))
BASINS = {'38H': (None, None),
          '39E': (29756, None),
          '39F': (29783, None),
          '39G': (32036, None),
          '39H': (32486, None),
          '40A': (31100, '06126500'),
          '40C': (31100, None),
          '40E': (30557, None),
          '40EJ': (30488, None),
          '40G': (47863, None),
          '40H': (47849, None),
          '40I': (47913, None),
          '40J': (47836, None),
          '40N': (47794, None),
          '40O': (48353, None),
          '40P': (48288, None),
          '40Q': (48304, None),
          '40R': (48205, None),
          '40S': (48166, None),
          '41B': (28826, '06023100'),
          '41C': (28444, '06023000'),
          '41D': (28879, '06026420'),
          '41G': (28994, '06036650'),
          '41F': (28592, '06042500'),
          '41H': (28712, '06052500'),
          '41J': (30762, '06078000'),
          '41L': (31203, None),
          '41M': (30985, None),
          '41P': (31451, None),
          '41QJ': (30774, None),
          '41R': (31427, None),
          '41S': (30562, None),
          '41T': (31407, None),
          '42B': (31484, None),
          '42I': (29685, None),
          '42KJ': (32022, None),
          '42M': (33557, None),
          '43A': (28753, '06195600'),
          '43BJ': (29049, '06200000'),
          '43BV': (29056, None),
          '43C': (29011, None),
          '43D': (29018, None),
          '43E': (29556, None),
          '43O': (31494, '06294000'),
          '43QJ': (29019, None),
          '76B': (51554, '12304500'),
          '76C': (51625, '12302055'),
          '76D': (51267, '12305000'),
          '76E': (51480, '12334510'),
          '76F': (51481, '12340000'),
          '76G': (52800, '12334550'),
          '76GJ': (51656, '12331590'),
          '76H': (52817, '12352500'),
          # '76HF': (51513, None),
          '76K': (51544, '12370000'),
          '76L': (51335, '12388700'),
          '76LJ': (51544, '12372000'),
          '76M': (52810, '12354500'),
          '76N': (51319, '12391950')}

BASIN_AREA_SQMT = {'38H': 278550695.389283,
                   '39E': 3042729109.6021,
                   '39F': 3258744547.50484,
                   '39FJ': 1610111064.67012,
                   '39G': 1482509737.92465,
                   '39H': 229106036.51465,
                   '40A': 10417929301.1,
                   '40B': 4795054068.85033,
                   '40C': 9353401519.83034,
                   '40D': 7189468855.05369,
                   '40E': 8212667300.99136,
                   '40EJ': 5395086030.69088,
                   '40F': 4604864789.86496,
                   '40G': 2574330741.8903,
                   '40H': 2080967085.91507,
                   '40I': 1843824289.5143,
                   '40J': 13379209339.6,
                   '40K': 1376987524.30757,
                   '40L': 709503500.996461,
                   '40M': 4618648822.45336,
                   '40N': 2238456024.6417,
                   '40O': 5835374781.564279,
                   '40P': 5468364673.13323,
                   '40Q': 5601872814.80121,
                   '40R': 7444613894.15584,
                   '40S': 8120243996.5229,
                   '40T': 1761668033.53433,
                   '41A': 5993063628.62584,
                   '41B': 3770864867.7636094,
                   '41C': 2543417242.02906,
                   '41D': 7257029885.38008,
                   '41E': 1973828566.85478,
                   '41F': 5248228759.15476,
                   '41G': 3389566636.5886,
                   '41H': 4577117681.49215,
                   '41I': 7640484262.64705,
                   '41J': 5191124259.59254,
                   '41K': 4868329063.10839,
                   '41L': 3106927707.45646,
                   '41M': 3332167549.4096894,
                   '41N': 2575440844.22995,
                   '41O': 5245990343.27717,
                   '41P': 9178325826.01638,
                   '41Q': 4997400930.49162,
                   '41QJ': 3553635849.3109393,
                   '41R': 3156860855.66642,
                   '41S': 7146403998.74224,
                   '41T': 5014820127.51165,
                   '41U': 1414701977.07612,
                   '42A': 3385782411.42072,
                   '42B': 2355864379.13646,
                   '42C': 7422651889.05271,
                   '42I': 1685161555.9987,
                   '42J': 8809145870.21472,
                   '42K': 3760284371.51538,
                   '42KJ': 10927364245.5,
                   '42L': 4100623333.15109,
                   '42M': 12268292741.9,
                   '43A': 2204223521.52459,
                   '43B': 6545567015.69694,
                   '43BJ': 1385672012.70831,
                   '43BV': 975879851.79402,
                   '43C': 2790529516.36088,
                   '43D': 4016921908.90411,
                   '43E': 1558251583.0696104,
                   '43N': 677727702.498748,
                   '43O': 2593076989.63791,
                   '43P': 6518447798.38467,
                   '43Q': 7300655000.538091,
                   '43QJ': 1889671560.73067,
                   '76B': 1673353265.65035,
                   '76C': 2135261946.7454,
                   '76D': 5811772790.00205,
                   '76E': 2303615888.9324,
                   '76F': 5978372940.11868,
                   '76G': 5972249686.20062,
                   '76GJ': 1302470961.31509,
                   '76H': 7404994956.23921,
                   '76HA': 1712190709.76527,
                   '76HB': 945146090.714725,
                   '76HE': 3117141462.98134,
                   '76HF': 1270111784.82358,
                   '76I': 2930763795.7011003,
                   '76J': 4346474108.23291,
                   '76K': 1893369671.9957,
                   '76L': 5208593202.0236,
                   '76LJ': 7548010275.05377,
                   '76M': 5147145091.49565,
                   '76N': 5500295439.49127}

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

    year_start = int(sorted([x.split('_')[1] for x in list(df.columns)])[0])
    df['BASINNUM'] = df.index

    dfd = df.to_dict(orient='records')
    s, e = '{}-01-01'.format(year_start), '2020-12-31'
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
            recs = DataFrame(dict([(x[1], x[0]) for x in [irr, et, cc, ppt, etr, sm]]), index=idx)

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
    target_segments = [str(v) for k, v in BASINS.items()]
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


def compile_availability_data(wr_dir, model_hydrograph, gs_hydro_dir, demands_data, crop_cons_data):
    hydro = read_csv(model_hydrograph, index_col='time', infer_datetime_format=True,
                     parse_dates=True)
    years = [yr for yr in range(1991, 2019)]
    dtr = date_range('2019-01-01', '2019-12-31', freq='D')

    for basin, (segment, gage) in BASINS.items():

        if not gage:
            continue

        basin_dir = os.path.join(demands_data, basin)
        if not os.path.isdir(basin_dir):
            os.mkdir(basin_dir)

        area = BASIN_AREA_SQMT[basin]
        gdf_idx = DatetimeIndex(date_range('2019-04-01', '2019-10-31', freq='M'))
        gdf = DataFrame(columns=years, index=gdf_idx, data=np.zeros((len(gdf_idx), len(years))))

        grid_file = os.path.join(crop_cons_data, '{}.csv'.format(basin))
        grid_df = read_hydrograph(grid_file)

        grid_df['ppt'] = grid_df['ppt'] / area * grid_df['irr']
        grid_df['et_ratio'] = grid_df['et'] / grid_df['ppt']
        grid_df['dnrc_cc'] = grid_df['et'] - grid_df['ppt'] * 0.93
        for yr in years:
            grid_data = [r['dnrc_cc'] for i, r in grid_df.iterrows() if i.year == yr]
            gdf.loc[:, yr] = grid_data

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
        demand_df.to_csv(os.path.join(basin_dir, 'legal_demands.csv'))

        try:
            prms_hydro = hydro[str(segment)]
            gs_data = os.path.join(gs_hydro_dir, '{}.csv'.format(gage))
            if not os.path.exists(gs_data):
                gs_hydro = get_station_daily_data('q', '1989-01-01', '2018-12-31', gage)
                gs_hydro.to_csv(gs_data)
            else:
                gs_hydro = read_csv(gs_data, parse_dates=True, index_col=0,
                                    infer_datetime_format=True)

            doy_idx = [x for x in range(365)]
            empty = np.zeros((len(doy_idx), len(years)))
            prms_simulated = DataFrame(index=doy_idx, columns=years, data=empty)
            if gage:
                gs_obs = deepcopy(prms_simulated)

        except Exception as e:
            print(gage, e)
            continue

        for yr in years:
            try:
                prms_data = [r for i, r in prms_hydro.iteritems() if
                             i.year == yr and not (i.day == 29 and i.month == 2)]
                prms_simulated.loc[:, yr] = prms_data

                if gage:
                    gs_data = [r.values[0] for i, r in gs_hydro.iterrows() if
                               i.year == yr and not (i.day == 29 and i.month == 2)]
                    gs_obs.loc[:, yr] = gs_data

            except ValueError:
                pass
        prms_simulated['min'] = prms_simulated.min(axis=1)
        prms_simulated['max'] = prms_simulated.max(axis=1)
        prms_simulated['median'] = prms_simulated.median(axis=1)
        prms_simulated.to_csv(os.path.join(basin_dir, 'prms_hydrograph.csv'))

        if gage:
            gs_obs['min'] = gs_obs.min(axis=1)
            gs_obs['max'] = gs_obs.max(axis=1)
            gs_obs['median'] = gs_obs.median(axis=1)
            gs_obs.to_csv(os.path.join(basin_dir, 'usgs_hydrograph.csv'))

        break


if __name__ == '__main__':
    matplotlib.use('TkAgg')

    d = '/media/research/IrrigationGIS/Montana'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS/Montana'

    cc_dir = os.path.join(d, 'water_availability', 'consumptive_use_data')
    ee_data = os.path.join(cc_dir, 'ee_extracts')
    cc_data = os.path.join(cc_dir, 'basin_cc_series')
    # merge_gridded(ee_data, cc_dir, glob='DNRC_Basins_3OCT2022')

    basin_wr_pod = os.path.join(d, 'water_availability', 'basin_wr_gdb')
    gs_hydrographs = os.path.join(d, 'water_availability', 'gs_hydrographs')
    demands = os.path.join(d, 'water_availability', 'daily_demands_data')
    master_hydro = os.path.join(d, 'water_availability', 'segment_hydrographs', 'compiled_hydrographs.csv')

    compile_availability_data(basin_wr_pod, master_hydro, gs_hydrographs, demands, cc_data)
# ========================= EOF ====================================================================
