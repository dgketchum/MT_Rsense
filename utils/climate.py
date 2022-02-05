import os
from collections import OrderedDict

from tqdm import tqdm
import fiona
from pandas import read_csv
from shapely.geometry import Point, mapping


def write_ghcn_station_shapefile(stations_file, out_shp):
    meta = {'driver': 'ESRI Shapefile',
            'schema': {'geometry': 'Point',
                       'properties': OrderedDict([('FID', 'int:10'),
                                                  ('STAID', 'int:10'),
                                                  ('STANAME', 'str:254'),
                                                  ('ELEV', 'float:11.3')])},
            'crs': {'init': 'epsg:4326'}}

    with fiona.open(out_shp, 'w', **meta) as dst:
        ct = 1
        with open(stations_file, 'r') as fp:
            for line in fp.readlines():
                staid = line[0:12]
                lat, lon = float(line[13:21]), float(line[21:30])
                elev, sta_name = float(' '.join(line[32:38].split())), ' '.join(line[41: 72].split())
                if -117. < lon < -103. and 44. < lat < 49.2:
                    geo = Point(lon, lat)

                    record = [('FID', ct),
                              ('STAID', staid),
                              ('STANAME', sta_name),
                              ('ELEV', elev)]
                    feat = {'type': 'Feature', 'properties': OrderedDict(
                        record), 'geometry': mapping(geo)}
                    dst.write(feat)
                    print(record)
                    ct += 1

    # TODO: combine GHCN and SNOTEL, write code to fetch records for each at once


if __name__ == '__main__':
    d = '/media/research/IrrigationGIS/climate'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS/climate'
    _txt = os.path.join(d, 'ghcnd-stations.txt')
    _shp = os.path.join(d, 'ghcn_stations.shp')
    write_ghcn_station_shapefile(_txt, _shp)
# ========================= EOF ====================================================================
