import os
from datetime import datetime

import numpy as np
from pandas import DataFrame, to_datetime, date_range, read_csv
import fiona
from rasterstats import zonal_stats


def zonal_snodas(in_shp, raster_dir, out_csv):
    geo = []
    bad_geo_ct = 0
    with fiona.open(in_shp) as src:
        for feat in src:
            try:
                _ = feat['geometry']['type']
                geo.append(feat)
            except TypeError:
                bad_geo_ct += 1

    rasters = [os.path.join(raster_dir, x) for x in os.listdir(raster_dir) if x.endswith('.tif')]
    df = DataFrame(columns=['mean', 'max', 'min'])
    drng = date_range('2010-10-01', '2021-05-04', freq='D')
    # snodas missing e.g., 2012-12-20
    for r, d in zip(rasters, drng):
        ds = r.split('.')[0][-8:]
        dt_str = '{}-{}-{}'.format(ds[:4], ds[4:6], ds[-2:])
        stats = zonal_stats(in_shp, r, stats=['mean', 'max', 'min'], categorical=False)
        df.loc[to_datetime(dt_str)] = stats[0]
        print('{} {} {:.3f}'.format(dt_str, d, stats[0]['mean']))

    print(df.shape[0])
    rng = date_range(df.index[0], df.index[-1])
    df = df.reindex(rng, fill_value=np.nan)
    df = df.fillna(method='ffill')
    print(df.shape[0])
    df.to_csv(out_csv)


def snodas_data_file(in_csv, out_csv):

    df = read_csv(in_csv, infer_datetime_format=True, index_col=['Unnamed: 0'],
                  parse_dates=True)

    df['mean'] = df['mean'] / 25.4

    with open(out_csv, 'w') as f:

        time_div = ['Year', 'Month', 'day', 'hr', 'min', 'sec']
        df['Year'] = [i.year for i in df.index]
        df['Month'] = [i.month for i in df.index]
        df['day'] = [i.day for i in df.index]
        for t_ in time_div[3:]:
            df[t_] = [0 for _ in df.index]

        df = df[['Year', 'Month', 'day', 'mean']]

        df.to_csv(f, sep=' ', header=False, index=False, float_format='%.1f')


if __name__ == '__main__':
    d = '/media/research/IrrigationGIS'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS'

    shp = os.path.join(d, 'Montana/upper_yellowstone/gsflow_prep/domain/carter_basin.shp')
    ras_dir = os.path.join(d, 'climate/snodas/rasters')
    out_csv_ = os.path.join(d, 'Montana/upper_yellowstone/gsflow_prep/snodas/carter_basin_snodas.csv')
    # zonal_snodas(shp, ras_dir, out_csv_)

    datafile_ = os.path.join(d, 'Montana/upper_yellowstone/gsflow_prep/uyws_carter_5000/input/snodas.data')
    snodas_data_file(out_csv_, datafile_)
# ========================= EOF ====================================================================

