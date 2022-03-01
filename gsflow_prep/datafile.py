import os
from datetime import datetime

from pandas import read_csv, date_range, to_datetime, isna, DataFrame
import fiona
from shapely.geometry import shape

from utils.hydrograph import get_station_daily_data
from utils.elevation import elevation_from_coordinate


def write_basin_datafile(basin_shp, ghcn_shp, snotel_shp, ghcn_data, gages_shp, data_file, start='1990-01-01'):
    _crs = [fiona.open(shp).meta['crs'] for shp in [basin_shp, ghcn_shp, snotel_shp]]
    assert all(x == _crs[0] for x in _crs)
    with fiona.open(basin_shp, 'r') as basn:
        basin_geo = shape([f['geometry'] for f in basn][0])

    dt_index = date_range(start, '2021-12-31')

    snotel_stations, snotel_names = [], []
    with fiona.open(snotel_shp, 'r') as sntstn:
        for f in sntstn:
            geo = shape(f['geometry'])
            if not geo.intersects(basin_geo):
                continue
            snotel_stations.append(f['properties']['ID'])
            snotel_names.append(f['properties']['Name'])

    ghcn_dct = {}
    with fiona.open(ghcn_shp, 'r') as ghcnstn:
        for f in ghcnstn:

            geo = shape(f['geometry'])

            if not geo.intersects(basin_geo):
                continue

            name = f['properties']['STANAME']
            if name in snotel_names:
                snotel = True
            else:
                snotel = False

            ghcn_dct[f['properties']['STAID']] = {'start': f['properties']['START'],
                                                  'end': f['properties']['END'],
                                                  'length': len(date_range(f['properties']['START'],
                                                                           f['properties']['END'])),
                                                  'lat': f['properties']['LAT'],
                                                  'lon': f['properties']['LON'],
                                                  'elev': f['properties']['ELEV'],
                                                  'snotel': snotel,
                                                  'name': name}

    ghcn_srt = [k for k, v in sorted(ghcn_dct.items(), key=lambda x: x[1]['length'], reverse=True)]

    met_dct, invalid_stations = {}, 0
    for sta in ghcn_srt:
        _file = os.path.join(ghcn_data, '{}.csv'.format(sta))
        df = read_csv(_file, parse_dates=True, infer_datetime_format=True)
        df.index = to_datetime(df['DATE'])

        s = ghcn_dct[sta]['start']
        if to_datetime(start) > to_datetime(s):
            df = df.loc[start:]
            if df.empty or df.shape[0] < 1000:
                print(sta, 'insuf records in date range')
                invalid_stations += 1
                continue

        df = df.reindex(dt_index)

        try:
            df = df[['TMAX', 'TMIN', 'PRCP']]
            df['tmax'] = (df['TMAX'] / 10. * 9 / 5) + 32
            df['tmin'] = (df['TMIN'] / 10. * 9 / 5) + 32
            df['prcp'] = df['PRCP'] / 10.
            df = df[['tmax', 'tmin', 'prcp']]

            if df.empty or df.shape[0] < 1000:
                print(sta, 'insuf records in date range')
                invalid_stations += 1

        except KeyError as e:
            print(sta, 'incomplete', e)
            invalid_stations += 1
            continue

        df[isna(df)] = -999

        print(sta, df.shape[0])
        met_dct[sta] = {'data': df, 'lat': ghcn_dct[sta]['lat'], 'lon': ghcn_dct[sta]['lon'],
                        'snotel': ghcn_dct[sta]['snotel'], 'elev': ghcn_dct[sta]['elev'],
                        'name': ghcn_dct[sta]['name']}

    gage_dct = {}
    with fiona.open(gages_shp, 'r') as src:
        for f in src:
            geo = shape(f['geometry'])
            if not geo.intersects(basin_geo):
                continue
            sta = f['properties']['STAID']
            s, e = f['properties']['start'], f['properties']['end']
            elev = elevation_from_coordinate(f['properties']['LAT'], f['properties']['LON'])
            df = get_station_daily_data('discharge', s, e, sta, freq='dv')
            df.columns = ['flow']

            if to_datetime(start) > to_datetime(s):
                df = df.loc[start:]

            df = df.tz_convert(None)
            df = df.reindex(dt_index)
            df[isna(df)] = -999

            gage_dct[sta] = {'data': df,
                             'lat': f['properties']['LAT'],
                             'lon': f['properties']['LON'],
                             'elev': elev,
                             'name': f['properties']['STANAME'],
                             'snotel': False}

    input_dct = {**met_dct, **gage_dct}
    dt_now = datetime.now().strftime('%Y-%m-%d %H:%M')

    with open(data_file, 'w') as f:

        df = DataFrame(index=dt_index)
        df['Year'], df['Month'] = [i.year for i in df.index], [i.month for i in df.index]
        df['day'], df['hr'], df['min'] = [i.day for i in df.index], [0 for _ in df.index], [0 for _ in df.index]

        [f.write('{}\n'.format(item)) for item in ['PRMS Datafile',
                                                   dt_now,
                                                   ''.join(['/' for _ in range(95)]),
                                                   '    '.join(
                                                       ['# ID  ',
                                                        '    Site Name{}'.format(' '.join([' ' for _ in range(14)])),
                                                        'Type',
                                                        'Lat',
                                                        'Lon',
                                                        'Elev (m)',
                                                        'Units'])]]

        counts = {'tmax': 0, 'tmin': 0, 'prcp': 0, 'flow': 0}
        for var, unit in zip(['tmax', 'tmin', 'prcp', 'flow'], ['C', 'C', 'mm', 'cfs']):
            for k, v in input_dct.items():
                d = v['data']
                if var in d.columns:
                    line_ = ' '.join(['#', k.rjust(11, '0'),
                                      v['name'].ljust(40, ' ')[:40],
                                      var,
                                      '{:.3f}'.format(v['lat']),
                                      '{:.3f}'.format(v['lon']),
                                      '{:.1f}'.format(v['elev']),
                                      unit]) + '\n'
                    f.write(line_)
                    df['{}_{}'.format(k, var)] = d[var]
                    counts[var] += 1
        f.write(''.join(['/' for _ in range(95)]) + '\n')
        for k, v in counts.items():
            f.write('{} {}\n'.format(k, v))
        f.write(''.join(['#' for _ in range(95)]) + '\n')
        df.to_csv(f, sep=' ', header=False, index=False, float_format='%.2f')


if __name__ == '__main__':
    d = '/media/research/IrrigationGIS'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS'

    uy = os.path.join(d, 'Montana', 'upper_yellowstone', 'gsflow_prep')
    clim = os.path.join(d, 'climate')
    stations_ = os.path.join(clim, 'stations')

    ghcn_shp_aea = os.path.join(stations_, 'ghcn_us_aea.shp')
    snotel_shp_ = os.path.join(stations_, 'snotel_stations.shp')
    basin_ = os.path.join(uy, 'uyws_basin.shp')
    _ghcn_data = os.path.join(clim, 'ghcn', 'ghcn_daily_summaries_4FEB2022')
    _snotel_data = os.path.join(clim, 'snotel', 'snotel_records')
    gage_shp_ = os.path.join(d, 'gages', 'gage_loc_usgs', 'selected_gages_aea.shp')
    datafile = os.path.join(uy, 'uy.data')

    write_basin_datafile(basin_, ghcn_shp_aea, snotel_shp_, _ghcn_data, gage_shp_, datafile)
# ========================= EOF ====================================================================
