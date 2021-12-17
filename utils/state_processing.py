import os
from collections import OrderedDict

import fiona


def fiona_merge(out_shp, file_list):
    meta = fiona.open(file_list[0]).meta
    meta['schema'] = {'type': 'Feature', 'properties': OrderedDict(
        [('FID', 'int:9')]), 'geometry': 'Polygon'}
    ct = 1
    with fiona.open(out_shp, 'w', **meta) as output:
        for s in file_list:
            sub_ct = 0
            for feat in fiona.open(s):
                geo = feat['geometry']
                if geo['type'] != 'Polygon':
                    print(geo['type'])
                    continue
                p = feat['properties']
                if not p['SW_WDID1'] and p['GW_ID1']:
                    feat = {'type': 'Feature', 'properties': {'FID': ct},
                            'geometry': geo}
                    output.write(feat)
                    ct += 1
                    sub_ct += 1
            print('{} features in {}'.format(sub_ct, s))

    print('{} features'.format(ct))
    return None


if __name__ == '__main__':
    d = '/media/research/IrrigationGIS/training_data/irrigated/CO/cdss/irr_polys'
    fl = [os.path.join(d, x) for x in os.listdir(d) if '.shp' in x]
    out_s = os.path.join(d, 'co_gw.shp')
    fiona_merge(out_s, fl)
# ========================= EOF ====================================================================
