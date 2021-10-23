import os
from copy import copy
from tqdm import tqdm
import numpy as np
# from fuzzywuzzy import process, fuzz
from geopandas import read_file, GeoDataFrame, array
from shapely.ops import unary_union
from pandas import Series, to_datetime

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
    flat = GeoDataFrame(columns=['DT', 'geo'])
    pou = read_file(pou_src)
    pou = pou[(pou['PURPOSE'] == 'IRRIGATION') & (pou['WRSTATUS'] == 'ACTIVE')]
    pou['ENFRPRIDAT'] = [to_datetime(x) for x in pou['ENFRPRIDAT']]
    pou = pou.rename(columns={'geometry': 'geo', 'ENFRPRIDAT': 'DT'})
    sections = pou.groupby(['TR', 'SECNO'], as_index=False).agg({'DT': np.min})
    # pou = pou[pou['TR'] == '8N8W']

    sec_not_oldest = False
    for i, (tr, sec, dt) in tqdm(sections.iterrows(), total=sections.shape[0]):
        if tr == '12N18W' and sec == '1':
            d = pou[(pou['TR'] == tr) & (pou['SECNO'] == sec)]

            min_idx = d['DT'].idxmin()
            min_date = d.loc[min_idx]['DT']

            full_secs = d[d['QSECTION'].isnull()]

            if not full_secs.empty:
                if full_secs['DT'].min() == min_date:
                    geo = d.loc[d[d['DT'] == min_date]['SHAPE_Area'].idxmax()]['geo']
                    flat = flat.append({'DT': min_date, 'geo': geo}, ignore_index=True)
                    continue
                else:
                    sec_not_oldest = True

            if full_secs.empty or sec_not_oldest:
                dg = d.groupby(['QSECTION'], as_index=False).agg({'DT': np.min})
                geos = []
                for q in dg['QSECTION']:
                    geos.append(d.loc[d[d['QSECTION'] == q].index[0]]['geo'])
                dg['geo'] = geos
                dg['len'] = [len(x) for x in dg['QSECTION']]

                dg.sort_values(by='DT', inplace=True)
                first, base_d, base_g, old_union, covered = True, None, None, [], []
                old_g = None

                for i, (q, dt, g, l) in dg.iterrows():
                    if first:
                        base_g = g
                        base_d = dt
                        first = False
                        old_union.append(base_g)
                        old_g = base_g
                        covered = base_g
                        continue

                    if dt == base_d:
                        old_g = unary_union([g, old_g])
                        covered = copy(old_g)
                    else:
                        if g.intersects(old_g):
                            g = old_g.difference(g)
                            if g.is_empty:
                                continue
                            else:
                                g = covered.difference(g)
                                if g.is_empty:
                                    continue
                                covered = unary_union([g, covered])
                                if g.intersects(covered):
                                    continue
                                flat = flat.append({'DT': dt, 'geo': g}, ignore_index=True)
                        else:
                            g = covered.difference(g)
                            if g.is_empty:
                                continue
                            covered = unary_union([g, covered])
                            if g.intersects(covered):
                                continue
                            flat = flat.append({'DT': dt, 'geo': g}, ignore_index=True)
                if old_g:
                    flat = flat.append({'DT': base_d, 'geo': old_g}, ignore_index=True)
                else:
                    flat = flat.append({'DT': base_d, 'geo': base_g}, ignore_index=True)

            sec_not_oldest = False
        else:
            continue

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
