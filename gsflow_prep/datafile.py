import os
import json
from datetime import datetime
from collections import OrderedDict

import numpy as np
from pandas import read_csv, date_range, to_datetime, isna, DataFrame

from utils.hydrograph import get_station_daily_data


def write_basin_datafile(station_json, gage_json, ghcn_data, data_file, out_csv, start='1990-01-01'):
    with open(station_json, 'r') as fp:
        stations = json.load(fp)
    with open(gage_json, 'r') as fp:
        gages = json.load(fp)

    dt_index = date_range(start, '2021-12-31')

    invalid_stations = 0

    for k, v in stations.items():
        # if k not in ['USC00243378', 'USC00245080']:
        #     continue
        _file = os.path.join(ghcn_data, '{}.csv'.format(k))
        df = read_csv(_file, parse_dates=True, infer_datetime_format=True)
        df.index = to_datetime(df['DATE'])

        s = v['start']
        if to_datetime(start) > to_datetime(s):
            df = df.loc[start:]
            if df.empty or df.shape[0] < 1000:
                print(k, 'insuf records in date range')
                invalid_stations += 1
                continue

        df = df.reindex(dt_index)

        try:
            df = df[['TMAX', 'TMIN', 'PRCP']]
            df['tmax'] = df['TMAX'] / 10.
            df['tmin'] = df['TMIN'] / 10.
            df['precip'] = df['PRCP'] / 10.
            df = df[['tmax', 'tmin', 'precip']]

            if df.empty or df.shape[0] < 1000:
                print(k, 'insuf records in date range')
                invalid_stations += 1

        except KeyError as e:
            print(k, 'incomplete', e)
            invalid_stations += 1
            continue

        df[isna(df)] = -999

        print(k, df.shape[0])
        stations[k]['data'] = df

    for k, v in gages.items():
        if k != '06192500':
            continue
        s, e = v['start'], v['end']
        df = get_station_daily_data('discharge', s, e, k, freq='dv')
        df.columns = ['runoff']

        if to_datetime(start) > to_datetime(s):
            df = df.loc[start:]

        df = df.tz_convert(None)
        df = df.reindex(dt_index)
        df[isna(df)] = -999

        v['data'] = df

    # sort by zone for met stations
    input_dct = OrderedDict(sorted(stations.items(), key=lambda item: item[1]['zone']))
    [input_dct.update({k: v}) for k, v in gages.items()]
    dt_now = datetime.now().strftime('%Y-%m-%d %H:%M')

    with open(data_file, 'w') as f:

        df = DataFrame(index=dt_index)

        time_div = ['Year', 'Month', 'day', 'hr', 'min', 'sec']
        df['Year'] = [i.year for i in df.index]
        df['Month'] = [i.month for i in df.index]
        df['day'] = [i.day for i in df.index]
        for t_ in time_div[3:]:
            df[t_] = [0 for _ in df.index]

        [f.write('{}\n'.format(item)) for item in ['PRMS Datafile: {}\n'.format(dt_now),
                                                   '// ',
                                                   '    '.join(
                                                       ['// ID  ',
                                                        '    Site Name{}'.format(' '.join([' ' for _ in range(14)])),
                                                        'Type',
                                                        'Lat',
                                                        'Lon',
                                                        'Elev (m)',
                                                        'Units'])]]

        counts = {'tmax': 0, 'tmin': 0, 'precip': 0, 'runoff': 0}
        for var, unit in zip(['tmax', 'tmin', 'precip', 'runoff'], ['C', 'C', 'mm', 'cfs']):
            for k, v in input_dct.items():
                try:
                    d = v['data']
                except KeyError:
                    continue
                if var in d.columns:
                    line_ = ' '.join(['// ', k.rjust(11, '0'),
                                      v['name'].ljust(40, ' ')[:40],
                                      var,
                                      '{:.3f}'.format(v['lat']),
                                      '{:.3f}'.format(v['lon']),
                                      '{:.1f}'.format(v['elev']),
                                      unit]) + '\n'
                    f.write(line_)
                    df['{}_{}'.format(k, var)] = d[var]
                    counts[var] += 1

        f.write('// \n')
        for k, v in counts.items():
            f.write('{} {}\n'.format(k, v))
        f.write('######################## \n')
        df.to_csv(f, sep=' ', header=False, index=False, float_format='%.1f')

        #  save dataframe to normal csv for use elsewhere
        df = df[[c for c in df.columns if c not in time_div]]
        df['date'] = df.index
        df[df.values == -999] = np.nan
        df.to_csv(out_csv, sep=' ', float_format='%.2f')


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
