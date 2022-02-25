import os
from ftplib import FTP
from collections import OrderedDict

import numpy as np
from pandas import DataFrame, read_csv, date_range, concat
import fiona
from shapely.geometry import Point, mapping, shape

from utils.hydrograph import get_station_daterange_data
from utils.elevation import elevation_from_coordinate


def write_basin_datafile(basin_shp, ghcn_shp, snotel_shp, ghcn_data, gages_shp, n_ghcn=None):
    _crs = [fiona.open(shp).meta['crs'] for shp in [basin_shp, ghcn_shp, snotel_shp]]
    assert all(x == _crs[0] for x in _crs)
    with fiona.open(basin_shp, 'r') as basn:
        basin_geo = shape([f['geometry'] for f in basn][0])

    snotel_stations, snotel_names = [], []
    with fiona.open(snotel_shp, 'r') as sntstn:
        for f in sntstn:
            geo = shape(f['geometry'])
            if not geo.intersects(basin_geo):
                continue
            snotel_stations.append(f['properties']['ID'])
            snotel_names.append(f['properties']['Name'])

    gage_dct = {}
    with fiona.open(gages_shp, 'r') as src:
        for f in src:
            geo = shape(f['geometry'])
            if not geo.intersects(basin_geo):
                continue
            sta = f['properties']['STAID']
            elev = elevation_from_coordinate(f['properties']['LAT'], f['properties']['LON'])
            gage_dct[sta] = {'length': len(date_range(f['properties']['start'],
                                                      f['properties']['end'])),
                             'lat': f['properties']['LAT'],
                             'lon': f['properties']['LON'],
                             'elev': elev}

    ghcn_dct = {}
    with fiona.open(ghcn_shp, 'r') as ghcnstn:
        for f in ghcnstn:

            geo = shape(f['geometry'])

            if not geo.intersects(basin_geo):
                continue

            if f['properties']['STANAME'] in snotel_names:
                snotel = True
            else:
                snotel = False

            ghcn_dct[f['properties']['STAID']] = {'length': len(date_range(f['properties']['START'],
                                                                           f['properties']['END'])),
                                                  'lat': f['properties']['LAT'],
                                                  'lon': f['properties']['LON'],
                                                  'elev': f['properties']['ELEV'],
                                                  'snotel': snotel}

    ghcn_srt = [k for k, v in sorted(ghcn_dct.items(), key=lambda x: x[1]['length'], reverse=True)]

    if n_ghcn:
        ghcn_srt = ghcn_srt[:n_ghcn]

    met_dct, invalid_stations = {}, 0
    for sta in ghcn_srt:
        _file = os.path.join(ghcn_data, '{}.csv'.format(sta))
        df = read_csv(_file, parse_dates=True, infer_datetime_format=True)
        df.index = df['DATE']
        try:
            df['TMAX'] = (df['TMAX'] / 10. * 9 / 5) + 32
            df['TMIN'] = (df['TMIN'] / 10. * 9 / 5) + 32
            df['PRCP'] = df['PRCP'] / 10.
            df = df[['TMAX', 'TMIN', 'PRCP']]
        except KeyError:
            invalid_stations += 1
            continue
        print(sta, df.shape[0])
        met_dct[sta] = {'data': df, 'lat': ghcn_dct[sta]['lat'], 'lon': ghcn_dct[sta]['lon'],
                        'snotel': ghcn_dct[sta]['snotel'], 'elev': ghcn_dct[sta]['elev']}


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

    write_basin_datafile(basin_, ghcn_shp_aea, snotel_shp_, _ghcn_data, gage_shp_)
# ========================= EOF ====================================================================
