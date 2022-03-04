import os
import json
from time import sleep
from collections import OrderedDict

import fiona
import numpy as np
from sklearn.cluster import KMeans
from shapely.geometry import shape, Point, mapping
from shapely.ops import cascaded_union
from pandas import read_csv, DataFrame, date_range
from rtree import index

from utils.elevation import elevation_from_coordinate
from datafile import write_basin_datafile


def get_gage_stations(basin_shp, gages_shp, out_json):
    """gather GHCN and USGS station/gages, write to .json"""

    _crs = [fiona.open(shp).meta['crs'] for shp in [basin_shp, gages_shp]]
    assert all(x == _crs[0] for x in _crs)
    with fiona.open(basin_shp, 'r') as basn:
        basin_geo = shape([f['geometry'] for f in basn][0])

    stations = {}

    with fiona.open(gages_shp, 'r') as src:
        for f in src:
            geo = shape(f['geometry'])
            if not geo.intersects(basin_geo):
                continue
            sta = f['properties']['STAID']
            s, e = f['properties']['start'], f['properties']['end']
            elev = elevation_from_coordinate(f['properties']['LAT'], f['properties']['LON'])
            print(sta, elev)
            sleep(2)

            stations[sta] = {'start': s, 'end': e,
                             'lat': f['properties']['LAT'],
                             'lon': f['properties']['LON'],
                             'elev': elev,
                             'type': 'usgs',
                             'name': f['properties']['STANAME'],
                             'proj_coords': (geo.y, geo.x),
                             'snotel': False}

    with open(out_json, 'w') as dst:
        dst.write(json.dumps(stations))


def get_ghcn_stations(basin_shp, ghcn_shp, snotel_shp, out_json, buffer=10):
    """gather GHCN and USGS station/gages, write to .json"""

    _crs = [fiona.open(shp).meta['crs'] for shp in [basin_shp, ghcn_shp]]
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

    stations = {}
    with fiona.open(ghcn_shp, 'r') as ghcnstn:
        buf_basin = basin_geo.buffer(buffer * 1000.)
        for f in ghcnstn:
            geo = shape(f['geometry'])

            if not geo.intersects(buf_basin):
                continue

            name = f['properties']['STANAME']
            if name in snotel_names:
                snotel = True
            else:
                snotel = False
            stations[f['properties']['STAID']] = {'start': f['properties']['START'],
                                                  'end': f['properties']['END'],
                                                  'length': len(date_range(f['properties']['START'],
                                                                           f['properties']['END'])),
                                                  'lat': f['properties']['LAT'],
                                                  'lon': f['properties']['LON'],
                                                  'elev': f['properties']['ELEV'],
                                                  'type': 'ghcn',
                                                  'snotel': snotel,
                                                  'proj_coords': (geo.y, geo.x),
                                                  'name': name}

    with open(out_json, 'w') as dst:
        dst.write(json.dumps(stations))


def precip_zone_geometry(basin_shp, huc_shp, station_meta, hru, zones_out, out_stations, n_stations=12):
    """use all snotel stations, then gather n most 'spread out' met stations"""

    with open(station_meta, 'r') as js:
        stations = json.load(js)

    met = {k: v for k, v in stations.items() if int(v['start'][:4]) < 1991 and int(v['end'][:4]) > 2020}

    pj = 'proj_coords'
    xx, yy, zz, s_ids = [], [], [], []
    [(xx.append(v[pj][1]), yy.append(v[pj][0]), zz.append(v['elev']), s_ids.append(k)) for k, v in met.items()]
    x_min, x_max, y_min, y_max, z_min, z_max = min(xx), max(xx), min(yy), max(yy), min(zz), max(zz)

    def normx(x):
        return (x - x_min) / (x_max - x_min)

    def normy(y):
        return (y - y_min) / (y_max - y_min)

    def normz(z):
        return (z - z_min) / (z_max - z_min)

    norm_cords = [(s, (normx(x), normy(y), normz(z))) for s, x, y, z in zip(s_ids, xx, yy, zz)]
    kmeans = KMeans(n_clusters=n_stations).fit([c[1] for c in norm_cords])
    centers = kmeans.cluster_centers_
    pick_stations = []

    for c in centers:
        dist = [Point(c).distance(Point(nc[1])) for nc in norm_cords]
        _id = s_ids[dist.index(min(dist))]
        pick_stations.append(_id)

    met = {k: v for k, v in met.items() if k in pick_stations}
    [met[k].update({'zone': i}) for i, k in enumerate(met.keys())]

    with fiona.open(basin_shp, 'r') as basn:
        basin_geo = shape([f['geometry'] for f in basn][0])

    huc_intersect = []
    with fiona.open(huc_shp, 'r') as src:
        meta = src.meta
        for f in src:
            geo = shape(f['geometry'])
            if geo.intersects(basin_geo) and geo.intersection(basin_geo).area / geo.area > 0.1:
                huc_intersect.append((f, shape(f['geometry']).centroid))

    for f, cent in huc_intersect:
        dist_tup = [(k, np.floor(cent.distance(Point(v[pj][1], v[pj][0])))) for k, v in met.items()]
        sid, dist = [x[0] for x in dist_tup], [x[1] for x in dist_tup]
        f['properties']['zone'] = met[sid[dist.index(min(dist))]]['zone']

    idx = index.Index()
    with fiona.open(hru, 'r') as src:
        for f in src:
            idx.insert(f['properties']['HRU_ID'], shape(f['geometry']).bounds)

    ppt_zones = {}
    for k, v in met.items():
        buf_pt = Point(v[pj][1], v[pj][0]).buffer(1).bounds
        hru = [x for x in idx.intersection(buf_pt)][0]

        zone = v['zone']
        huc_geo = [shape(h[0]['geometry']) for h in huc_intersect if h[0]['properties']['zone'] == zone]
        ppt_zone_geo = cascaded_union(huc_geo)
        ppt_zones[v['zone']] = {'type': 'Feature',
                                'geometry': mapping(ppt_zone_geo),
                                'properties': OrderedDict([('FID', zone),
                                                           ('PPT_ZONE', zone),
                                                           ('PPT_HRU_ID', hru),
                                                           ('STAID', k)])}

    meta['schema'] = {'type': 'Feature', 'properties': OrderedDict([('FID', 'int:9'),
                                                                    ('PPT_ZONE', 'int:9'),
                                                                    ('PPT_HRU_ID', 'int:9'),
                                                                    ('STAID', 'str:254')]),
                      'geometry': 'Polygon'}

    with fiona.open(zones_out, 'w', **meta) as dst:
        for k, v in ppt_zones.items():
            dst.write(v)

    with open(out_stations, 'w') as dst:
        dst.write(json.dumps(met))


def attribute_precip_zones(ppt_zones_shp, csv, out_shp):
    mdf = read_csv(csv, sep=' ', infer_datetime_format=True, index_col=0, parse_dates=True)

    with fiona.open(ppt_zones_shp, 'r') as src:
        features = [f for f in src]
        meta = src.meta

    meta['schema']['properties'].update({'HRU_PSTA': 'int:9'})
    meta['schema']['properties'].update({'STAID': 'str:254'})
    add_ = dict(('PPT_{}'.format(str(x).rjust(2, '0')), 'float:11.3') for x in range(1, 13))
    meta['schema']['properties'].update(add_)

    with fiona.open(out_shp, 'w', **meta) as dst:
        ct = 0
        for f in features:
            ct += 1
            _id = int(f['properties']['FID'])
            staname = f['properties']['STAID']

            record = OrderedDict([('FID', _id),
                                  ('PPT_ZONE', _id),
                                  ('STAID', staname),
                                  ('PPT_HRU_ID', f['properties']['PPT_HRU_ID']),
                                  ('HRU_PSTA', _id)])

            df = mdf[[c for c in mdf.columns if staname in c]]
            df = df.resample('M').agg(DataFrame.sum, skipna=False)
            for m in range(1, 13):
                data = [r['{}_prcp'.format(staname)] for i, r in df.iterrows() if i.month == m]
                record.update({'PPT_{}'.format(str(m).rjust(2, '0')): np.nanmean(data)})

            props = OrderedDict([(k, record[k]) for k in meta['schema']['properties'].keys()])
            feat = {'type': 'Feature', 'properties': props, 'geometry': f['geometry']}

            dst.write(feat)


if __name__ == '__main__':
    d = '/media/research/IrrigationGIS'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS'

    uy = os.path.join(d, 'Montana', 'upper_yellowstone', 'gsflow_prep')
    clim = os.path.join(d, 'climate')
    stations_ = os.path.join(clim, 'stations')

    basin_ = os.path.join(uy, 'uyws_basin.shp')
    gage_shp_ = os.path.join(d, 'gages', 'gage_loc_usgs', 'selected_gages_aea.shp')
    gage_json_ = os.path.join(uy, 'gages.json')
    # get_gage_stations(basin_, gage_shp_, out_json=gage_json_)

    ghcn_shp_aea = os.path.join(stations_, 'ghcn_us_aea.shp')
    snotel_shp_ = os.path.join(stations_, 'snotel_stations.shp')
    sta_json = os.path.join(uy, 'stations.json')
    # get_ghcn_stations(basin_, ghcn_shp_aea, snotel_shp_, out_json=sta_json, buffer=0)

    huc_ = os.path.join(d, 'boundaries', 'hydrography', 'HUC_Boundaries', 'merge_huc12_aea.shp')
    ppt_zones_geo = os.path.join(uy, 'met', 'ppt_zone_geometries.shp')
    selected_stations_json = os.path.join(uy, 'selected_stations.json')
    hru_shp_ = os.path.join(d, 'software', 'gsflow-arcpy-master', 'uyws_multibasin', 'hru_params', 'hru_params.shp')
    # precip_zone_geometry(basin_, huc_, sta_json, hru_shp_, ppt_zones_geo, selected_stations_json)

    _ghcn_data = os.path.join(clim, 'ghcn', 'ghcn_daily_summaries_4FEB2022')
    _snotel_data = os.path.join(clim, 'snotel', 'snotel_records')
    datafile = os.path.join(uy, 'uy.data')
    csv_ = os.path.join(uy, 'uy.csv')
    # write_basin_datafile(selected_stations_json, gage_json_, _ghcn_data, datafile, csv_)

    ppt_zones_ = os.path.join(uy, 'met', 'ppt_zones.shp')
    # attribute_precip_zones(ppt_zones_geo, csv_, out_shp=ppt_zones_)

# ========================= EOF ====================================================================
