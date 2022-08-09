import os
from copy import deepcopy

import pandas as pd
import geopandas as gpd
from shapely.geometry import MultiPolygon


def point_etof(shp, etof_dir, outshape):
    gdf = gpd.read_file(shp)

    summary = deepcopy(gdf)
    summary['etof'] = [-99.99 for _ in summary['FID']]
    summary['geo'] = [None for _ in summary['FID']]

    idx = max(summary.index.values) + 1
    for i, row in gdf.iterrows():
        file_ = os.path.join(etof_dir, str(row['FID']).rjust(3, '0'))
        df = pd.read_csv(file_, index_col=0, infer_datetime_format=True, parse_dates=True)
        df = df.rename(columns={list(df.columns)[0]: 'etof'})
        r_index = pd.date_range('2016-01-01', '2021-12-31', freq='D')
        df = df.reindex(r_index)
        df = df.interpolate()
        df['mday'] = ['{}-{}'.format(x.month, x.day) for x in df.index]
        target_range = pd.date_range('2016-05-09', '2016-09-19')

        accept = ['{}-{}'.format(x.month, x.day) for x in target_range]
        df['mask'] = [1 if d in accept else 0 for d in df['mday']]
        df = df[df['mask'] == 1]
        if isinstance(row['geometry'], MultiPolygon):
            first = True
            for g in row['geometry'].geoms:
                if first:
                    summary.loc[i, 'geo'] = g
                    summary.loc[i, 'etof'] = df['etof'].mean()

                    first = False
                else:
                    summary.loc[idx] = row
                    summary.loc[idx, 'geo'] = g
                    summary.loc[idx, 'etof'] = df['etof'].mean()
                    idx += 1
        else:
            summary.loc[i, 'geo'] = row['geometry']
            summary.loc[i, 'etof'] = df['etof'].mean()

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
    shp_ = os.path.join(r, 'sweetgrass_fields_sample.shp')
    etof_ = os.path.join(r, 'sweetgrass_fields_etof')
    out_summary = os.path.join(r, 'sweetgrass_fields_comparison_etof.shp')
    point_etof(shp_, etof_, out_summary)
# ========================= EOF ====================================================================
