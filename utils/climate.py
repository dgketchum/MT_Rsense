import os
from ftplib import FTP
from collections import OrderedDict

import numpy as np
from pandas import DataFrame, read_csv, date_range, concat
import fiona
from shapely.geometry import Point, mapping, shape


def get_ghcnd_station_metadata(out_dir):
    temp_txt = 'ghcnd-stations.txt'
    ftp = FTP('ftp.ncdc.noaa.gov')
    ftp.login()
    ftp.cwd('pub/data/ghcn/daily')
    ftp.retrbinary('RETR {}'.format(temp_txt), open('{}'.format(temp_txt), 'wb').write)
    ftp.quit()

    ghcnd_stations = np.genfromtxt(temp_txt, delimiter=(11, 9, 10, 7, 4, 30), dtype=str)
    os.remove(temp_txt)
    df = DataFrame(ghcnd_stations, columns=['STAID', 'LATITUDE', 'LONGITUDE', 'ELEV', '-', 'STANAME'])
    df.to_csv(os.path.join(out_dir, 'ghcnd-stations.csv'))


def write_ghcn_station_shapefile(stations_file, out_shp, csv_dir):
    meta = {'driver': 'ESRI Shapefile',
            'schema': {'geometry': 'Point',
                       'properties': OrderedDict([('FID', 'int:10'),
                                                  ('STAID', 'str:254'),
                                                  ('STANAME', 'str:254'),
                                                  ('START', 'str:254'),
                                                  ('END', 'str:254'),
                                                  ('LAT', 'float:11.3'),
                                                  ('LON', 'float:11.3'),
                                                  ('ELEV', 'float:11.3')])},
            'crs': {'init': 'epsg:4326'}}

    with fiona.open(out_shp, 'w', **meta) as dst:
        df = read_csv(stations_file)
        lat, lon = df['LATITUDE'], df['LONGITUDE']
        geos = [Point(lon_, lat_) for lon_, lat_ in zip(lon, lat)]
        df = df[['STAID', 'ELEV', 'STANAME', 'LATITUDE', 'LONGITUDE']]
        df['geometry'] = geos
        ct = 1
        for i, r in df.iterrows():
            _file = os.path.join(csv_dir, '{}.csv'.format(r['STAID']))

            try:
                sdf = read_csv(_file)
            except FileNotFoundError:
                print('missing ', )
                continue

            start, end = sdf.iloc[0]['DATE'], sdf.iloc[-1]['DATE']
            if not start.startswith('20'):
                print(start)
            record = [('FID', ct),
                      ('STAID', r['STAID']),
                      ('START', start),
                      ('END', end),
                      ('LAT', r['LATITUDE']),
                      ('LON', r['LONGITUDE']),
                      ('STANAME', ' '.join(r['STANAME'].split())),
                      ('ELEV', r['ELEV'])]

            feat = {'type': 'Feature', 'properties': OrderedDict(
                record), 'geometry': mapping(r['geometry'])}

            dst.write(feat)
            print(record)
            ct += 1






if __name__ == '__main__':
    d = '/media/research/IrrigationGIS'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS'

    uy = os.path.join(d, 'Montana', 'upper_yellowstone', 'gsflow_prep')
    clim = os.path.join(d, 'climate')
    stations_ = os.path.join(clim, 'stations')
    gages_ = os.path.join(clim, 'gages')
    # get_ghcnd_station_metadata(clim)

    ghcn_data_dir = os.path.join(d, 'climate', 'ghcn', 'ghcn_daily_summaries_4FEB2022')
    _txt = os.path.join(stations_, 'ghcnd-stations.csv')
    ghcn_shp_ = os.path.join(stations_, 'ghcn_stations.shp')
    # write_ghcn_station_shapefile(_txt, ghcn_shp_, ghcn_data_dir)

    ghcn_shp_aea = os.path.join(stations_, 'ghcn_us_aea.shp')
    snotel_shp_ = os.path.join(stations_, 'snotel_stations.shp')
    basin_ = os.path.join(uy, 'uyws_basin.shp')
    _ghcn_data = os.path.join(clim, 'ghcn', 'ghcn_daily_summaries_4FEB2022')
    _snotel_data = os.path.join(clim, 'snotel', 'snotel_records')
    gage_shp_ = os.path.join(gages_, 'uy_usgs_gages.shp')
    write_basin_datafile(basin_, ghcn_shp_aea, snotel_shp_, _ghcn_data, gage_shp_)
# ========================= EOF ====================================================================
