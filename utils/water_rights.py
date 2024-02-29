import os
from subprocess import check_call

import shapely.errors
from tqdm import tqdm
from geopandas import read_file, GeoDataFrame
from shapely.geometry import Polygon, MultiPolygon
from pandas import to_datetime, Timestamp

OGR = '/home/dgketchum/miniconda3/envs/opnt/bin/ogr2ogr'
OGRINFO = '/home/dgketchum/miniconda3/envs/opnt/bin/ogrinfo'

DROP_UNIQUE = ['OBJECTID', 'POUID', 'ACREAGE', 'TR', 'SECNO', 'QSECTION', 'LLDSID', 'TRSSID', 'SHAPE_Leng',
               'SHAPE_Area', 'geometry']

DROP_RESMP = ['HISTRGTTYP', 'LATECLAIM', 'IRRTYPE', 'GOVTLOT']


def clip_data_to_basin(gdb, basin_shp_dir, pou_dir, append_str='pou', overwrite=False):
    shapes = [os.path.join(basin_shp_dir, x) for x in os.listdir(basin_shp_dir) if x.endswith('.shp')]
    cmd = [OGRINFO, '--config', '-sql', "CREATE SPATIAL INDEX ON {}".format(gdb.strip('.shp')), gdb]
    check_call(cmd)
    for s in shapes:
        splt = os.path.basename(s).replace('.shp', '_{}.shp'.format(append_str))
        o = os.path.join(pou_dir,  splt)
        if os.path.exists(o) and not overwrite:
            print(o, 'exists, skipping')
            continue
        cmd = [OGRINFO, '--config', '-sql', "CREATE SPATIAL INDEX ON {}".format(s.strip('.shp')), s]
        check_call(cmd)
        cmd = [OGR, '-progress', '-f', 'ESRI Shapefile', '-clipsrc', s, o, gdb]
        print(' '.join(cmd))
        check_call(cmd)


def get_flat_priority(pou_src, out_file):
    flat = GeoDataFrame(columns=['id', 'DT', 'geo', 'obj'])
    pou = read_file(pou_src)
    pou = pou[(pou['PURPOSE'] == 'IRRIGATION') & (pou['WRSTATUS'] == 'ACTIVE')]
    pou['ENFRPRIDAT'] = [to_datetime(x) for x in pou['ENFRPRIDAT']]
    pou = pou.rename(columns={'geometry': 'geo', 'ENFRPRIDAT': 'dt'})
    pou = pou.sort_values(by='dt')
    pou = pou[['dt', 'geo', 'OBJECTID']]
    pou = pou.reset_index(drop=True)
    good_rows = [i for i, x in enumerate(pou['dt']) if isinstance(x, Timestamp)]
    pou = pou.loc[good_rows]
    pou = pou.astype({'OBJECTID': int})

    first, covered = True, None
    ct = 0
    for i, (dt, g, obj) in pou.iterrows():
        if first:
            flat.loc[ct] = [ct, dt, g, obj]
            ct += 1
            first = False
        else:
            try:
                equal = [i for i, x in enumerate(flat['geo']) if g.equals_exact(x, 1.0)]
                if any(equal):
                    continue
                inter = [i for i, x in enumerate(flat['geo']) if g.intersects(x)]
                if not any(inter):
                    flat.loc[ct] = [ct, dt, g, obj]
                    ct += 1
                else:
                    for ix in inter:
                        g = g.difference(flat.loc[ix]['geo'])
                    if g.area > 0:
                        flat.loc[ct] = [ct, dt, g, obj]
                        ct += 1
            except shapely.errors.GEOSException:
                print(i, 'errored')

    good_rows = [i for i, x in enumerate(flat['geo']) if isinstance(x, Polygon) or isinstance(x, MultiPolygon)]
    flat = flat.loc[good_rows]
    geo = flat['geo']
    flat['DT'] = [str(x)[:10] for x in flat['DT']]
    flat['dt_int'] = [int(''.join(x.split('-'))) for x in flat['DT']]
    flat.drop(columns=['geo'], inplace=True)
    gdf = GeoDataFrame(flat, geometry=geo, crs='EPSG:32100')
    gdf.to_file(out_file)


if __name__ == '__main__':
    wr = '/media/research/IrrigationGIS/Montana/water_rights'
    wa = '/media/research/IrrigationGIS/Montana/water_availability'
    pou_c = os.path.join(wa, 'basin_wr_gdb')
    pou_f = os.path.join(wr, 'pou_flat')
    _clip = [os.path.join(pou_c, x) for x in os.listdir(pou_c) if x.endswith('pou.shp')]
    # for c in _clip:
    #     if '41K' not in c:
    #         continue
    #     f = os.path.join(pou_f, os.path.basename(c))
    #     print('processing {}'.format(f))
    #     get_flat_priority(c, f)

    shapes = os.path.join(root, 'gage_basins')
    pou_d = os.path.join(root, 'pou')
    gdb_ = os.path.join(root, 'wr_gdb', 'wrpou.shp')
    clip_data_to_basin(gdb_, basin_shp_dir=shapes, pou_dir=pou_d)
# ========================= EOF ====================================================================
