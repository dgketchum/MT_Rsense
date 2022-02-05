import os
from collections import OrderedDict

import fiona
from pandas import read_csv, DataFrame


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
    root = '/home/dgketchum/Downloads/uyws'
    met = os.path.join(root, 'gridmet_precip_series')
    shp = os.path.join(root, 'ppt_zones_hru.shp')
    o_shp = os.path.join(root, 'ppt_zones.shp')
    attribute_precip_zones(shp, met, o_shp)
# ========================= EOF ====================================================================
