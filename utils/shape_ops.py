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


def merge_huc(pdir, outshp):
    shps = [[os.path.join(pdir, sd, 'Shape', x) for x in os.listdir(os.path.join(pdir, sd, 'Shape'))
             if x.endswith('WBDHU12.shp')] for sd in os.listdir(pdir)]
    shps = [item for sublist in shps for item in sublist]
    features, huc_codes, huc_names = [], [], []
    first = True
    for shp in shps:
        with fiona.open(shp, 'r') as src:
            if first:
                meta = src.meta
                first = False

            for f in src:
                code = f['properties']['HUC12']
                name_ = f['properties']['Name']

                if code not in huc_codes:
                    features.append(f)
                    huc_codes.append(code)

    with fiona.open(outshp, 'w', **meta) as dst:
        for f in features:
            dst.write(f)


if __name__ == '__main__':
    d = '/media/research/IrrigationGIS/training_data/irrigated/CO/cdss/irr_polys'
    fl = [os.path.join(d, x) for x in os.listdir(d) if '.shp' in x]
    out_s = os.path.join(d, 'co_gw.shp')
    fiona_merge(out_s, fl)
# ========================= EOF ====================================================================
