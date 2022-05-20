import os
import subprocess
from copy import deepcopy

import numpy as np
import fiona
from shapely.geometry import shape
from geopandas import read_file
from pandas import concat, DataFrame
import warnings
from shapely.errors import ShapelyDeprecationWarning

warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

ogr = '/home/dgketchum/miniconda3/bin/ogr2ogr'


def convert(_dir):
    l = [os.path.join(_dir, x) for x in os.listdir(_dir) if x.endswith('.gdb')]
    for f in l:
        out = f.replace('.gdb', '.shp')
        cmd = [ogr, '-f', 'ESRI Shapefile', out, f]
        subprocess.call(cmd)
        print(out)


def join(features, dbf_dir, intersect):
    with fiona.open(intersect) as src:
        inter = [shape(f['geometry']) for f in src][0]

    dbf = [os.path.join(dbf_dir, db_) for db_ in os.listdir(dbf_dir) if db_.endswith('.dbf') and 'Fabric' in db_]
    regions = list(set([f.strip('.dbf').split('_')[-1] for f in dbf]))

    master = DataFrame()
    df_ct, cols = 0, []
    for region in regions:
        shp = [f for f in features if region in f][0]
        dbf_ = [x for x in dbf if region in x]
        first = True
        for db in dbf_:
            if first:
                df = read_file(db)
                df.index = df['hru_id']
                first = False
            else:
                c = read_file(db)
                c.index = c['hru_id']
                c.drop(columns=['hru_id', 'geometry'], inplace=True)
                df = concat([df, c], axis=1)

        df_ct += 1
        ct = 0

        df = df.loc[:, ~df.columns.duplicated()]

        for f in ['hru_id_nat', 'hru_id_reg', 'region']:
            df[f] = [np.nan for x in range(df.shape[0])]

        with fiona.open(shp) as src:
            for f in src:
                p = f['properties']
                geo = shape(f['geometry'])
                intersection = geo.intersection(inter).area / geo.area
                if intersection > 0.1:
                    idx = p['hru_id_reg']
                    df.loc[idx, 'hru_id_nat'] = p['hru_id_nat']
                    df.loc[idx, 'hru_id_reg'] = p['hru_id_reg']
                    df.loc[idx, 'region'] = p['region']
                    try:
                        df.loc[idx, 'geometry'] = geo
                    except ValueError:
                        i = np.argmax([p.area for p in list(geo.geoms)])
                        df.loc[idx, 'geometry'] = geo.geoms[i]
                    ct += 1
                    if ct % 1000 == 0:
                        print(ct, ' features')

        df = df.loc[~np.isnan(df['hru_id_nat'])]

        if df_ct > 1:
            common_cols = set(master.columns).intersection(set(df.columns))
            master = concat([master[common_cols], df[common_cols]], ignore_index=True)

        else:
            master = deepcopy(df)

    master = master.loc[:, ~master.columns.duplicated()]
    _file = os.path.join(dbf_dir, 'prms_params_carter.shp')
    master.to_file(_file, crs='EPSG:4326')
    DataFrame(master.drop(columns=['geometry'])).to_csv(_file.replace('.shp', '.csv'))


if __name__ == '__main__':
    d = '/media/research/IrrigationGIS/Montana/geospatial_fabric'
    # intersect_ = os.path.join(d, 'huc6_MT_intersect_dissolve.shp')
    intersect_ = '/media/research/IrrigationGIS/Montana/geospatial_fabric/carter_basin_wgs.shp'
    feats = [os.path.join(d, 'nhru_10U.shp'), os.path.join(d, 'nhru_17.shp')]
    join(feats, d, intersect_)

# ===============================================================
