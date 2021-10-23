import os
from copy import copy
from tqdm import tqdm
import numpy as np
# from fuzzywuzzy import process, fuzz
from geopandas import read_file, GeoDataFrame, array
from shapely.ops import unary_union
from shapely.geometry import Polygon
from pandas import Series, to_datetime, Timestamp

DROP_UNIQUE = ['OBJECTID', 'POUID', 'ACREAGE', 'TR', 'SECNO', 'QSECTION', 'LLDSID', 'TRSSID', 'SHAPE_Leng',
               'SHAPE_Area', 'geometry']

DROP_RESMP = ['HISTRGTTYP', 'LATECLAIM', 'IRRTYPE', 'GOVTLOT']


# def ap_fuzz(x, y):
#     pext = process.extract(x, y, limit=1, scorer=fuzz.token_sort_ratio)[0]
#     _str, score = pext[0], pext[1]
#     return [_str, score]


# def fuzzy_join(wr_shp, cad_shp, out_name):
#     cad = read_file(cad_shp)
#     cad = cad[~cad['OwnerName'].isnull()]
#     wr_shp = read_file(wr_shp).drop(columns='geometry')
#     cad['key'] = cad.OwnerName.apply(lambda x: ap_fuzz(x, wr_shp.ALLOWNERS))
#     cad['fuzz_score'] = cad.key.apply(lambda x: x[1])
#     cad['key'] = cad.key.apply(lambda x: x[0])
#     df = cad.merge(wr_shp, left_on='key', right_on='ALLOWNERS')
#     df.to_csv(out_name)
#     df.to_file(out_name.replace('.csv', '.shp'))


def join_unary_pou_pod(pod_src, pou_src, pou_out):
    pod = read_file(pod_src)
    pod = pod[pod['SRCTYPE'] == 'SURFACE']
    pou = read_file(pou_src)
    pou = pou[(pou['PURPOSE'] == 'IRRIGATION') & (pou['WRSTATUS'] == 'ACTIVE')]
    geo = [(key, pou[pou['WRKEY'] == key]['geometry'].unary_union) for key in pou['WRKEY'].unique()]
    pou.drop(columns=DROP_UNIQUE + DROP_RESMP, inplace=True)
    pou = pou.groupby(['WRKEY'], as_index=False).agg(Series.mode)

    pou = pou.astype({c: str for c in pou.columns})
    pou['geometry'] = [x[1] for x in geo]

    addtl_drop = ['OBJECTID', 'WELLDEPTH', 'X_METERS', 'Y_METERS', 'XY_MAPPED']
    drop_cols = [x for x in list(pod.columns) if x in list(pou.columns) and x != 'WRKEY'] + addtl_drop
    pod.drop(columns=drop_cols, inplace=True)
    df = pou.merge(pod, on='WRKEY', how='inner')
    geo = df['geometry']
    df.drop(columns=['geometry'], inplace=True)
    gdf = GeoDataFrame(df, geometry=geo)
    gdf.to_file(pou_out)


def join_pou_pod_sections(pod_src, pou_src, pou_out):
    pod = read_file(pod_src)
    pod = pod[pod['SRCTYPE'] == 'SURFACE']
    pou = read_file(pou_src)
    pou = pou[(pou['PURPOSE'] == 'IRRIGATION') & (pou['WRSTATUS'] == 'ACTIVE')]
    pou = pou.astype({'OBJECTID': int})
    pou = pou.astype({c: str for c in pou.columns})


def get_flat_priority(pou_src, out_file):

    flat = GeoDataFrame(columns=['id', 'DT', 'geo'])
    pou = read_file(pou_src)
    pou = pou[(pou['PURPOSE'] == 'IRRIGATION') & (pou['WRSTATUS'] == 'ACTIVE')]
    pou['ENFRPRIDAT'] = [to_datetime(x) for x in pou['ENFRPRIDAT']]
    pou = pou.rename(columns={'geometry': 'geo', 'ENFRPRIDAT': 'dt'})
    pou = pou.sort_values(by='dt')
    pou = pou[['dt', 'geo']]
    pou = pou.reset_index(drop=True)
    good_rows = [i for i, x in enumerate(pou['dt']) if isinstance(x, Timestamp)]
    pou = pou.loc[good_rows]

    first, covered = True, None
    ct = 0
    for i, (dt, g) in tqdm(pou.iterrows(), total=pou.shape[0]):
        if first:
            flat.loc[ct] = [ct, dt, g]
            ct += 1
            first = False
        else:
            inter = [i for i, x in enumerate(flat['geo']) if g.intersects(x)]
            if not any(inter):
                flat.loc[ct] = [ct, dt, g]
                ct += 1
            else:
                for ix in inter:
                    try:
                        g = flat.loc[ix]['geo'].difference(g)
                    except KeyError:
                        pass
                if g.area > 0:
                    flat.loc[ct] = [ct, dt, g]
                    ct += 1

    good_rows = [i for i, x in enumerate(flat['geo']) if isinstance(x, Polygon)]
    flat = flat.loc[good_rows]
    geo = flat['geo']
    flat['DT'] = [str(x)[:10] for x in flat['DT']]
    flat['dt_int'] = [int(''.join(x.split('-'))) for x in flat['DT']]
    flat.drop(columns=['geo'], inplace=True)
    gdf = GeoDataFrame(flat, geometry=geo, crs='EPSG:32100')
    gdf.to_file(out_file)


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/Montana/water_rights/mt_wr'
    # pod_ = os.path.join(root, 'cfr_div.shp')
    pou_ = os.path.join(root, 'cfr_pou.shp')
    # out = os.path.join(root, 'cfr_pou_surfIrr_nonUnion.shp')
    # join_pou_pod_sections(pod_, pou_, out)
    out = os.path.join(root, 'cfr_pou_flat_dates.shp')
    get_flat_priority(pou_, out)
# ========================= EOF ====================================================================
