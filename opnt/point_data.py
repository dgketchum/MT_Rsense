import os
from copy import deepcopy
from calendar import monthrange

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import MultiPolygon


def point_etof(shp, data_dir, outshape, var='etof'):
    gdf = gpd.read_file(shp)

    fips_dct = {r['FID']: r['FIPS'] for i, r in gdf.iterrows()}

    summary = deepcopy(gdf)
    summary[var] = [-99.99 for _ in summary['FID']]
    summary['geo'] = [None for _ in summary['FID']]

    idx = max(summary.index.values) + 1

    for i, row in gdf.iterrows():
        fid = row['FID']
        file_ = os.path.join(data_dir, str(fid).rjust(3, '0'))

        if not os.path.exists(file_):
            continue

        df = pd.read_csv(file_, index_col=0, infer_datetime_format=True, parse_dates=True)
        df = df.rename(columns={list(df.columns)[0]: var})
        r_index = pd.date_range('2016-01-01', '2021-12-31', freq='D')

        if var == 'etof':
            df = df.reindex(r_index)
            df = df.interpolate()
        elif var == 'et':
            ndf = df.reindex(r_index)
            for ii, r in df.iterrows():
                val = r['et'] / 25.4
                end_day = monthrange(ii.year, ii.month)[1]
                ndf.loc[ii: '{}-{}-{}'.format(ii.year, ii.month, end_day)] = val / end_day
            df = deepcopy(ndf)
        else:
            raise NotImplementedError

        df['mday'] = ['{}-{}'.format(x.month, x.day) for x in df.index]

        if fips_dct[fid] == '067':
            target_range = pd.date_range('2016-05-09', '2016-09-22')
        elif fips_dct[fid] == '097':
            target_range = pd.date_range('2016-05-01', '2016-09-28')

        accept = ['{}-{}'.format(x.month, x.day) for x in target_range]
        df['mask'] = [1 if d in accept else 0 for d in df['mday']]
        df = df[df['mask'] == 1]
        if isinstance(row['geometry'], MultiPolygon):
            first = True
            for g in row['geometry'].geoms:
                if first:
                    summary.loc[i, 'geo'] = g
                    if var == 'etof':
                        summary.loc[i, var] = df[var].mean()
                    else:
                        summary.loc[i, var] = df[var].resample('A').sum().mean()

                    first = False
                else:
                    summary.loc[idx] = row
                    summary.loc[idx, 'geo'] = g
                    if var == 'etof':
                        summary.loc[idx, var] = df[var].mean()
                    else:
                        summary.loc[idx, var] = df[var].resample('A').sum().mean()

                    idx += 1
        else:
            summary.loc[i, 'geo'] = row['geometry']
            if var == 'etof':
                summary.loc[i, var] = df[var].mean()
            else:
                summary.loc[i, var] = df[var].resample('A').sum().mean()

    summary['openet_cu'] = [np.nan for _ in range(summary.shape[0])]
    for i, r in summary.iterrows():
        fip, itype = r['FIPS'], r['IType']
        if fip == '097':
            if itype == 'P':
                summary.loc[i, 'openet_cu'] = r['et'] - 4.09 - 0.5
            else:
                summary.loc[i, 'openet_cu'] = r['et'] - 5.46 - 2.0
        else:
            if itype == 'P':
                summary.loc[i, 'openet_cu'] = r['et'] - 3.93 - 0.5
            else:
                summary.loc[i, 'openet_cu'] = r['et'] - 5.24 - 2.0

    geos = summary['geo']
    summary.drop(columns=['geo', 'geometry'], inplace=True)
    summary['geometry'] = geos
    summary = summary.set_crs('epsg:4326')
    summary.to_file(outshape)
    summary.to_csv(outshape.replace('.shp', '.csv'))


if __name__ == '__main__':
    d = '/media/research/IrrigationGIS'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS'

    r = os.path.join(d, 'Montana', 'water_rights', 'hua', 'comparison_data')
    shp_ = os.path.join(r, 'sweetgrass_fields_comparison_et_dnrc.shp')
    etof_ = os.path.join(r, 'sweetgrass_fields_et')
    out_summary = os.path.join(r, 'sweetgrass_fields_comparison_et_cu.shp')
    point_etof(shp_, etof_, out_summary, var='et')
# ========================= EOF ====================================================================
