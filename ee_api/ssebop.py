import os
import sys

import ee
from openet import ssebop as model

from ee_api import is_authorized
from water_availability.basin_availability import BASINS

sys.path.insert(0, os.path.abspath('..'))
sys.setrecursionlimit(5000)

IRR = 'projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp'
BASINS = 'users/dgketchum/gages/gage_basins'

L7, L8 = 'LANDSAT/LE07/C02/T1_L2', 'LANDSAT/LC08/C02/T1_L2'


def export_etf(basin, year=2015, bucket=None, debug=False):
    basin = ee.FeatureCollection(BASINS).filterMetadata('STAID', 'equals', basin)

    s, e = '1987-01-01', '2021-12-31'
    irr_coll = ee.ImageCollection(IRR)
    coll = irr_coll.filterDate(s, e).select('classification')
    remap = coll.map(lambda img: img.lt(1))
    irr_min_yr_mask = remap.sum().gte(5)
    irr = irr_coll.filterDate('{}-01-01'.format(year), '{}-12-31'.format(year)).select('classification').mosaic()
    irr_mask = irr_min_yr_mask.updateMask(irr.lt(1))

    coll = ee.ImageCollection(L7).merge(ee.ImageCollection(L8))
    coll = coll.filterDate('{}-04-01'.format(year), '{}-10-31'.format(year)).filterBounds(basin.geometry())
    scenes = coll.aggregate_histogram('system:index').getInfo()
    scenes = [x[2:] for x in list(scenes.keys())]

    for img_id in scenes:

        if 'LE07' in img_id:
            sat = L7
        else:
            sat = L8

        img = ee.Image(os.path.join(sat, img_id))
        model_obj = model.Image.from_landsat_c2_sr(
            img,
            tcorr_source='FANO',
            et_reference_source='projects/openet/reference_et/gridmet/daily',
            et_reference_band='etr',
            et_reference_factor=1.0,
            et_reference_resample='nearest')

        etf = model_obj.et_fraction
        etf = etf.clip(basin.geometry()).mask(irr_mask).multiply(1000).int()

        if debug:
            point = ee.Geometry.Point([-112.6771, 46.3206])
            data = etf.sample(point, 30).getInfo()
            print(data['features'])

        task = ee.batch.Export.image.toCloudStorage(
            etf,
            description='ETF_{}'.format(img_id),
            bucket=bucket,
            region=basin.geometry(),
            scale=30)

        task.start()
        print(img_id)


if __name__ == '__main__':
    is_authorized()
    bucket_ = 'wudr'
    basin_ = '12334550'
    for y in [2016, 2018]:
        export_etf(basin_, y, bucket_, debug=False)

# ========================= EOF ================================================================================
