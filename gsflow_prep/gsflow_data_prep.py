import os
import json
from time import sleep
from collections import OrderedDict

import fiona
import numpy as np
from shapely.geometry import shape, Point, mapping
from shapely.ops import cascaded_union
from pandas import read_csv, date_range
from rtree import index
from scipy.stats.stats import linregress

from utils.elevation import elevation_from_coordinate


def get_gage_stations(basin_shp, gages_shp, out_json):
    """Select USGS gages within the bounds of basin_shp write to .json"""

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
    """Select GHCN stations within the bounds of basin_shp write to .json"""

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


def met_zones_geometries(station_meta, hru, zones_out, out_stations):
    """Select 'spread out' meteorology stations (in 3 dimensions, using K means clustering),
    by selecting long-period stations near the center of each cluster,
    collect and assign the nearest HUC 12 basins to each station,
    and find PRMS model domain HRU ID.
    Finally, build the attribute table and write both selected stations (json, shp) and precipitation zones (shp).
    """

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

    [met[k].update({'zone': i}) for i, k in enumerate(met.keys(), start=1)]

    met_norm_coords = [(s, Point(normx(x), normy(y), normz(z))) for s, x, y, z in zip(s_ids, xx, yy, zz)]

    # TODO: use rtree 3D nearest to pair HRU to met stations
    idx = index.Index()
    hru_station = {}
    with fiona.open(hru, 'r') as src:
        meta = src.meta
        for f in src:
            _id = f['properties']['HRU_ID']
            idx.insert(_id, shape(f['geometry']).bounds)
            hru_cent = shape(f['geometry']).centroid
            x, y = normx(hru_cent.x), normy(hru_cent.y)
            z = normz(f['properties']['HRU_ID'])
            norm_center = Point(x, y, z)
            dist = [norm_center.distance(mnc[1]) for mnc in met_norm_coords]
            dist_idx = dist.index(min(dist))
            hru_station[_id] = (s_ids[dist_idx], shape(f['geometry']))

    ppt_zones = {}
    for k, v in met.items():
        pt = Point(v[pj][1], v[pj][0])
        buf_pt = pt.buffer(1).bounds
        hru = [x for x in idx.intersection(buf_pt)][0]

        zone = v['zone']

        zone_geo = [vv[1] for kk, vv in hru_station.items() if vv[0] == k]
        zone_geo = cascaded_union(zone_geo)

        if not pt.intersects(zone_geo):
            hru = 0
        ppt_zones[v['zone']] = {'type': 'Feature',
                                'geometry': mapping(zone_geo),
                                'properties': OrderedDict([('FID', zone),
                                                           ('MET_ZONE', zone),
                                                           ('MET_HRU_ID', hru),
                                                           ('STAID', k)])}
    meta['schema'] = {'type': 'Feature', 'properties': OrderedDict([('FID', 'int:9'),
                                                                    ('MET_ZONE', 'int:9'),
                                                                    ('MET_HRU_ID', 'int:9'),
                                                                    ('STAID', 'str:254')]),
                      'geometry': 'Polygon'}

    with fiona.open(zones_out, 'w', **meta) as dst:
        print('write {}'.format(zones_out))
        for k, v in ppt_zones.items():
            dst.write(v)

    with open(out_stations, 'w') as dst:
        print('write {}'.format(out_stations))
        dst.write(json.dumps(met))

    station_shp_file = out_stations.replace('.json', '.shp')
    meta['schema']['geometry'] = 'Point'

    with fiona.open(station_shp_file, 'w', **meta) as dst:
        print('write {}'.format(station_shp_file))
        for k, v in ppt_zones.items():
            sta = [vv for kk, vv in met.items() if vv['zone'] == k][0]
            pt = mapping(Point(sta[pj][1], sta[pj][0]))
            f = {'type': 'Feature',
                 'geometry': pt,
                 'properties': OrderedDict([('FID', k),
                                            ('MET_ZONE', k),
                                            ('MET_HRU_ID', v['properties']['MET_HRU_ID']),
                                            ('STAID', v['properties']['STAID'])])}
            dst.write(f)


def calculate_monthly_lapse_rates(csv, station_meta):
    mdf = read_csv(csv, sep=' ', infer_datetime_format=True, index_col=0, parse_dates=True)
    mdf = mdf.groupby(mdf.index.month).mean()
    with open(station_meta, 'r') as js:
        stations = json.load(js)

    tmin_lapse, tmax_lapse = [], []
    for temp in ['tmin', 'tmax']:
        for month in range(1, 13):
            temps, elevations = [], []
            cols = [c for c in mdf.columns if temp in c]
            d = mdf[cols]
            [temps.append(d['{}_{}'.format(s, temp)].loc[month]) for s in stations.keys()]
            [elevations.append(v['elev']) for k, v in stations.items()]
            regression = linregress(elevations, temps)
            if temp == 'tmin':
                tmin_lapse.append('{:.3f}'.format(regression.slope * 1000.))
            else:
                tmax_lapse.append('{:.3f}'.format(regression.slope * 1000.))

    print('tmax_lapse = {}'.format(', '.join(tmax_lapse)))
    print('tmin_lapse = {}'.format(', '.join(tmin_lapse)))

    print('station elevations')
    elevs = sorted([(v['zone'], v['elev']) for k, v in stations.items()], key=lambda x: x[0])
    print(', '.join([str(x[1]) for x in elevs]))


def attribute_precip_zones(ppt_zones_shp, csv, out_shp):
    """Write collected met data (created during datafile prep) mean precipitation data to precipitation zones
    shapefile."""
    mdf = read_csv(csv, sep=' ', infer_datetime_format=True, index_col=0, parse_dates=True)

    with fiona.open(ppt_zones_shp, 'r') as src:
        features = [f for f in src]
        meta = src.meta

    meta['schema']['properties'].update({'HRU_PSTA': 'int:9'})
    meta['schema']['properties'].update({'STAID': 'str:254'})

    var_tuples = [('TMAX', 'tmax'), ('TMIN', 'tmin'), ('PPT', 'precip')]
    for _var, _ in var_tuples:
        add_ = dict(('{}_{}'.format(_var, str(x).rjust(2, '0')), 'float:11.3') for x in range(1, 13))
        meta['schema']['properties'].update(add_)

    with fiona.open(out_shp, 'w', **meta) as dst:
        ct = 0
        for f in features:
            ct += 1
            _id = int(f['properties']['FID'])
            staname = f['properties']['STAID']

            record = OrderedDict([('FID', _id),
                                  ('MET_ZONE', _id),
                                  ('STAID', staname),
                                  ('MET_HRU_ID', f['properties']['MET_HRU_ID']),
                                  ('HRU_PSTA', _id)])

            df = mdf[[c for c in mdf.columns if staname in c]]
            resamp_method = {c: 'sum' if 'precip' in c else 'mean' for c in df.columns}
            df = df.resample('M').agg(resamp_method)
            for uvar, lvar in var_tuples:
                for m in range(1, 13):
                    data = [r['{}_{}'.format(staname, lvar)] for i, r in df.iterrows() if i.month == m]
                    record.update({'{}_{}'.format(uvar, str(m).rjust(2, '0')): np.nanmean(data)})

            props = OrderedDict([(k, record[k]) for k in meta['schema']['properties'].keys()])
            feat = {'type': 'Feature', 'properties': props, 'geometry': f['geometry']}

            dst.write(feat)
    print('wrote {}'.format(out_shp))


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
