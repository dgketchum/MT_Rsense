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

from utils.elevation import elevation_from_coordinate


def get_station_metadata(basin_shp, ghcn_shp, gages_shp, snotel_shp, out_json, buffer=10):
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


def create_precip_zones(basin_shp, huc_shp, station_meta, zones_out, n_stations=12):
    """use all snotel stations, then gather n most 'spread out' met stations"""

    with open(station_meta, 'r') as js:
        stations = json.load(js)

    met = {k: v for k, v in stations.items() if
           v['type'] == 'ghcn' and int(v['start'][:4]) < 1991 and int(v['end'][:4]) > 2020}
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

    ppt_zones = {}
    for k, v in met.items():
        zone = v['zone']
        huc_geo = [shape(h[0]['geometry']) for h in huc_intersect if h[0]['properties']['zone'] == zone]
        ppt_zone_geo = cascaded_union(huc_geo)
        ppt_zones[v['zone']] = {'type': 'Feature',
                                'geometry': mapping(ppt_zone_geo),
                                'properties': OrderedDict([('FID', zone),
                                                           ('PPT_ZONE', zone),
                                                           ('station', k)])}

    fields = [('FID', 'int:9'),
              ('PPT_ZONE', 'int:9'),
              ('staton', 'str:254')]

    meta['schema'] = {'type': 'Feature', 'properties': OrderedDict(
        fields), 'geometry': 'Polygon'}

    with fiona.open(zones_out, 'w', **meta) as dst:
        for k, v in ppt_zones.items():
            dst.write(v)
            # does not match schema for some reason


def attribute_precip_zones(ppt_zones_shp, csv_dir, out_shp):
    csv_l = [os.path.join(csv_dir, x) for x in os.listdir(csv_dir) if x.endswith('.csv')]
    csv_d = {os.path.basename(x).split('.')[0].split('_')[-1]: x for x in csv_l}

    with fiona.open(ppt_zones_shp, 'r') as src:
        features = [f for f in src]
        meta = src.meta

    ppt_fields = [('PPT_{}'.format(str(x).rjust(2, '0')), 'float:11.3') for x in range(1, 13)]
    fields = [('FID', 'int:9'),
              ('PPT_ZONE', 'int:9'),
              ('PPT_HRU_ID', 'int:9'),
              ('HRU_PSTA', 'int:9')]
    fields += ppt_fields
    meta['schema'] = {'type': 'Feature', 'properties': OrderedDict(
        fields), 'geometry': 'Polygon'}

    with fiona.open(out_shp, 'w', **meta) as dst:
        ct = 0
        for f in features:
            ct += 1
            _id = int(f['properties']['id'])

            record = OrderedDict([('FID', ct),
                                  ('PPT_ZONE', _id),
                                  ('PPT_HRU_ID', f['properties']['HRU_ID']),
                                  ('HRU_PSTA', ct)])

            sta_csv = csv_d[str(_id)]
            df = read_csv(sta_csv, index_col=0, infer_datetime_format=True, parse_dates=True)
            df = df.resample('M').agg(DataFrame.sum, skipna=False)
            for m in range(1, 13):
                data = [r['precip'] for i, r in df.iterrows() if i.month == m]
                mean_ = sum(data) / len(data)
                record.update({'PPT_{}'.format(str(m).rjust(2, '0')): mean_})

            feat = {'type': 'Feature', 'properties': OrderedDict(
                record), 'geometry': f['geometry']}
            dst.write(feat)


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
    gage_shp_ = os.path.join(d, 'gages', 'gage_loc_usgs', 'selected_gages_aea.shp')
    sta_json = os.path.join(uy, 'stations.json')
    # get_station_metadata(basin_, ghcn_shp_aea, gage_shp_, snotel_shp_, sta_json)

    huc_ = os.path.join(d, 'boundaries', 'hydrography', 'HUC_Boundaries', 'merge_huc12_aea.shp')
    ppt_zones_out = os.path.join(uy, 'ppt_zone_geometries.shp')
    create_precip_zones(basin_, huc_, sta_json, ppt_zones_out)

    _ghcn_data = os.path.join(clim, 'ghcn', 'ghcn_daily_summaries_4FEB2022')

# ========================= EOF ====================================================================
