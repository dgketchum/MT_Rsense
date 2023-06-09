import os
import sys
from pprint import pprint

import ee
from openet import ssebop as ssebop_model

from ee_api import is_authorized
from ee_api.ee_utils import landsat_masked

sys.path.insert(0, os.path.abspath('..'))
sys.setrecursionlimit(5000)

IRR = 'projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp'
BASINS = 'users/dgketchum/gages/gage_basins'

L7, L8 = 'LANDSAT/LE07/C02/T1_L2', 'LANDSAT/LC08/C02/T1_L2'


def export_ndvi(basin, year=2015, bucket=None, debug=False):
    basin = ee.FeatureCollection(BASINS).filterMetadata('STAID', 'equals', basin)

    s, e = '1987-01-01', '2021-12-31'
    irr_coll = ee.ImageCollection(IRR)
    coll = irr_coll.filterDate(s, e).select('classification')
    remap = coll.map(lambda img: img.lt(1))
    irr_min_yr_mask = remap.sum().gte(5)
    irr = irr_coll.filterDate('{}-01-01'.format(year),
                              '{}-12-31'.format(year)).select('classification').mosaic()
    irr_mask = irr_min_yr_mask.updateMask(irr.lt(1))

    coll = landsat_masked(year, basin).map(lambda x: x.normalizedDifference(['B5', 'B4']))
    scenes = coll.aggregate_histogram('system:index').getInfo()

    for img_id in scenes:

        if '039028' in img_id:
            pass
        else:
            continue

        img = coll.filterMetadata('system:index', 'equals', img_id).first()
        img = img.clip(basin.geometry()).mask(irr_mask).multiply(1000).int()

        if debug:
            point = ee.Geometry.Point([-112.75608, 46.31405])
            data = img.sample(point, 30).getInfo()
            print(data['features'])

        task = ee.batch.Export.image.toCloudStorage(
            img,
            description='NDVI_{}'.format(img_id),
            bucket=bucket,
            region=basin.geometry(),
            crs='EPSG:5070',
            scale=30)

        task.start()
        print(img_id)


if __name__ == '__main__':
    is_authorized()
    bucket_ = 'wudr'
    basin_ = '12334550'
    for y in [2016]:
        export_ndvi(basin_, y, bucket_, debug=False)

# ========================= EOF ================================================================================
