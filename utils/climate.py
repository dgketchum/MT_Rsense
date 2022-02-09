import os
from ftplib import FTP
from collections import OrderedDict

import numpy as np
from pandas import DataFrame, read_csv
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


def write_ghcn_station_shapefile(stations_file, out_shp):
    meta = {'driver': 'ESRI Shapefile',
            'schema': {'geometry': 'Point',
                       'properties': OrderedDict([('FID', 'int:10'),
                                                  ('STAID', 'str:254'),
                                                  ('STANAME', 'str:254'),
                                                  ('ELEV', 'float:11.3')])},
            'crs': {'init': 'epsg:4326'}}

    with fiona.open(out_shp, 'w', **meta) as dst:
        df = read_csv(stations_file)
        lat, lon = df['LATITUDE'], df['LONGITUDE']
        geos = [Point(lon_, lat_) for lon_, lat_ in zip(lon, lat)]
        df = df[['STAID', 'ELEV', 'STANAME']]
        df['geometry'] = geos
        ct = 1
        for i, r in df.iterrows():
            record = [('FID', ct),
                      ('STAID', r['STAID']),
                      ('STANAME', ' '.join(r['STANAME'].split())),
                      ('ELEV', r['ELEV'])]

            feat = {'type': 'Feature', 'properties': OrderedDict(
                record), 'geometry': mapping(r['geometry'])}

            dst.write(feat)
            print(record)
            ct += 1


def select_stations(basin_shp, ghcn_shp, snotel_shp):
    _crs = [fiona.open(shp).meta['crs'] for shp in [basin_shp, ghcn_shp, snotel_shp]]
    assert all(x == _crs[0] for x in _crs)
    with fiona.open(basin_shp, 'r') as basn:
        basin_geo = shape([f['geometry'] for f in basn][0])
    ghcn_stations = []
    with fiona.open(ghcn_shp, 'r') as ghcnstn:
        for f in ghcnstn:
            geo = shape(f['geometry'])
            if not geo.intersects(basin_geo):
                continue
            ghcn_stations.append(f['properties']['STAID'])
    snotel_stations = []
    with fiona.open(snotel_shp, 'r') as sntstn:
        for f in sntstn:
            geo = shape(f['geometry'])
            if not geo.intersects(basin_geo):
                continue
            snotel_stations.append(f['properties']['ID'])


if __name__ == '__main__':
    d = '/media/research/IrrigationGIS/climate/stations'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS/climate/stations'

    uy = os.path.join(d, 'Montana', 'upper_yellowstone', 'gsflow_prep')
    clim = os.path.join(d, 'climate', 'stations')
    # get_ghcnd_station_metadata(clim)

    _txt = os.path.join(clim, 'ghcnd-stations.csv')
    ghcn_shp_ = os.path.join(clim, 'ghcn_stations_regional.shp')
    # write_ghcn_station_shapefile(_txt, ghcn_shp)

    snotel_shp_ = os.path.join(clim, 'snotel_stations.shp')
    basin_ = os.path.join(uy, 'uyws_basin.shp')
    select_stations(basin_, ghcn_shp_, snotel_shp_)
# ========================= EOF ====================================================================
