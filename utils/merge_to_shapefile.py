import os
import json
import fiona
from pandas import read_csv, DataFrame


def concatenate_openet_results(shp, out_json, csv_dir):
    d = {}
    with fiona.open(shp, 'r') as src:
        for f in src:
            d[f['properties']['FID']] = f
        meta = src.meta

    _files = [os.path.join(csv_dir, x) for x in os.listdir(csv_dir)]

    def _sort(x):
        s = os.path.basename(x).split('.csv')[0][20:].split('_')
        return int(s[0]), int(s[1])

    _files.sort(key=lambda x: _sort(x))
    for f in _files:
        first = True
        date_str = os.path.basename(f).split('.csv')[0][20:]
        df = read_csv(f, index_col='FID').drop(columns='time')
        df.columns = ['{}_{}'.format(x, date_str) for x in list(df.columns)]
        df = df.T.to_dict()
        for fid, val in df.items():
            if first:
                meta['schema']['properties'].update({k: 'float:11.3' for k in val.keys()})
                first = False
            d[fid]['properties'].update(val)

    features = [v for k, v in d.items()]
    layer = {
        'type': 'FeatureCollection',
        'features': features}

    with open(out_json, 'w') as dst:
        dst.write(json.dumps(layer))

    meta['driver'] = 'ESRI Shapefile'
    shp_file = out_json.replace('geojson', 'shp')
    with fiona.open(shp_file, 'w', **meta) as dst:
        for k, v in d.items():
            dst.write(v)

    # df = DataFrame(data=d)
    # meta['driver'] = 'GPKG'
    # gpkg_file = out_json.replace('geojson', 'gpkg')
    # with fiona.Env():
    #     with fiona.open(gpkg_file, 'w', **meta) as dst:
    #         dst.writerecords(features)


if __name__ == '__main__':
    in_shp = '/media/research/IrrigationGIS/Montana/openet/uy_fields_27SEP2021.shp'
    out_jsn = '/media/research/IrrigationGIS/Montana/openet/uy_ensemble_et.geojson'
    csv_d = '/media/research/IrrigationGIS/Montana/openet/csv/uy'
    concatenate_openet_results(in_shp, out_jsn, csv_d)
# ========================= EOF ====================================================================
