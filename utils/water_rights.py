import os
import numpy as np
import pandas as pd
from shapely import geometry
from geopandas import read_file, GeoDataFrame, GeoSeries
from pandas import Series, DataFrame
from fuzzywuzzy import process, fuzz

pou_cols = ['OBJECTID', 'WRKEY', 'WRID', 'VERID', 'WRNUMBER', 'ALLOWNERS', 'WRTYPE', 'HISTRGTTYP', 'WRSTATUS',
            'LATECLAIM', 'PRIDATE', 'ENFRPRIDAT', 'POUID', 'PURPNUMBER', 'PURPOSE', 'IRRTYPE', 'MAXACRES', 'FLWRTGPM',
            'FLWRTCFS', 'VOL', 'ACREAGE', 'COUNTY', 'TR', 'SECNO', 'QSECTION', 'GOVTLOT', 'LLDSID', 'TRSSID',
            'ABSTRACT', 'LAYER_DATE', 'SHAPE_Leng', 'SHAPE_Area', 'geometry']

NON_UNIQUE = ['WRKEY', 'WRID', 'VERID', 'WRNUMBER', 'ALLOWNERS', 'WRTYPE', 'HISTRGTTYP', 'WRSTATUS', 'LATECLAIM',
              'PRIDATE', 'ENFRPRIDAT', 'PURPNUMBER', 'PURPOSE', 'IRRTYPE', 'MAXACRES', 'FLWRTGPM', 'FLWRTCFS', 'VOL',
              'COUNTY', 'GOVTLOT', 'ABSTRACT', 'LAYER_DATE']

DROP_UNIQUE = ['OBJECTID', 'POUID', 'ACREAGE', 'TR', 'SECNO', 'QSECTION', 'LLDSID', 'TRSSID', 'SHAPE_Leng',
               'SHAPE_Area', 'geometry']

DROP_RESMP = ['HISTRGTTYP', 'LATECLAIM', 'IRRTYPE', 'GOVTLOT']

CONV_DTYPE= ['MAXACRES', 'FLWRTGPM', 'FLWRTCFS', 'VOL']


def ap_fuzz(x, y):
    pext = process.extract(x, y, limit=1, scorer=fuzz.token_sort_ratio)[0]
    _str, score = pext[0], pext[1]
    return [_str, score]


def fuzzy_join(wr_shp, cad_shp, out_name):
    cad = read_file(cad_shp)
    cad = cad[~cad['OwnerName'].isnull()]
    wr_shp = read_file(wr_shp).drop(columns='geometry')
    cad['key'] = cad.OwnerName.apply(lambda x: ap_fuzz(x, wr_shp.ALLOWNERS))
    cad['fuzz_score'] = cad.key.apply(lambda x: x[1])
    cad['key'] = cad.key.apply(lambda x: x[0])
    df = cad.merge(wr_shp, left_on='key', right_on='ALLOWNERS')
    df.to_csv(out_name)
    df.to_file(out_name.replace('.csv', '.shp'))


def join_pou_pod(pod_src, pou_src, pou_out):
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


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/Montana/water_rights/mt_wr'
    pod_ = os.path.join(root, 'cfr_div.shp')
    pou_ = os.path.join(root, 'cfr_pou.shp')
    out = os.path.join(root, 'cfr_pou_surfIrr.shp')
    join_pou_pod(pod_, pou_, out)
# ========================= EOF ====================================================================
