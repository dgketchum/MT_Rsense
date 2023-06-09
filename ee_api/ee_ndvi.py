import os
import sys
from calendar import monthrange

import numpy as np
import ee
from ee_api import is_authorized

from water_availability.basin_availability import BASINS

sys.path.insert(0, os.path.abspath('..'))
sys.setrecursionlimit(5000)

import ee
import pandas as pd

from ee_utils import long_term_ndvi

TONGUE_FIELDS = 'users/dgketchum/fields/tongue_9MAY2023'
TONGUE_CENTR = 'users/dgketchum/fields/tongue_cent_19MAY2023'
COUNTIES = 'users/dgketchum/boundaries/western_17_counties'
IRR = 'projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp'

LMAP = {'LT05': 'LANDSAT/LT05/C02/T1_L2',
        'LE07': 'LANDSAT/LE07/C02/T1_L2',
        'LC08': 'LANDSAT/LC08/C02/T1_L2',
        'LC09': 'LANDSAT/LC09/C02/T1_L2'}


def calculate_ndvi_and_export(fc, roi, output_csv_path):
    ee.Initialize()

    s, e = '2021-07-01', '2021-07-31'
    ndvi = long_term_ndvi(roi, coll=True)
    ndvi = ndvi.toBands()

    def sample_regions(i, points):
        red = ee.Reducer.toCollection(i.bandNames())
        reduced = i.reduceRegions(points, red, 30)
        fc = reduced.map(lambda f: ee.FeatureCollection(f.get('features'))
                         .map(lambda q: q.copyProperties(f, None, ['features'])))
        return fc.flatten()

    s = sample_regions(ndvi, ee.FeatureCollection([fc.first()]))

    pass


if __name__ == '__main__':
    is_authorized()
    # roi_ = ee.FeatureCollection(COUNTIES)
    # roi_ = roi_.filterMetadata('GEOID', 'equals', '30017')

    roi_ = ee.FeatureCollection('users/dgketchum/boundaries/MT')

    fc_ = ee.FeatureCollection(TONGUE_CENTR)
    calculate_ndvi_and_export(fc_, roi_, 'ndvi_tongue.csv')

# ========================= EOF ================================================================================
