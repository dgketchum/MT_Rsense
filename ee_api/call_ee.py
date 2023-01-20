import os
import sys
from calendar import monthrange

import numpy as np
import ee
from ee_api import is_authorized

from water_availability.basin_availability import BASINS

sys.path.insert(0, os.path.abspath('..'))
sys.setrecursionlimit(5000)

RF_ASSET = 'projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp'

UMRB_CLIP = 'users/dgketchum/boundaries/umrb_ylstn_clip'
CMBRB_CLIP = 'users/dgketchum/boundaries/CMB_RB_CLIP'
CORB_CLIP = 'users/dgketchum/boundaries/CO_RB'

DNRC_BASINS = 'users/dgketchum/boundaries/DNRC_Basins'


def get_geomteries():
    bozeman = ee.Geometry.Polygon([[-111.19206055457778, 45.587493372544984],
                                   [-110.91946228797622, 45.587493372544984],
                                   [-110.91946228797622, 45.754947053477565],
                                   [-111.19206055457778, 45.754947053477565],
                                   [-111.19206055457778, 45.587493372544984]])

    navajo = ee.Geometry.Polygon([[-108.50192867920967, 36.38701227276218],
                                  [-107.92995297120186, 36.38701227276218],
                                  [-107.92995297120186, 36.78068624960868],
                                  [-108.50192867920967, 36.78068624960868],
                                  [-108.50192867920967, 36.38701227276218]])

    test_point = ee.Geometry.Point(-111.19206055457778, 45.587493372544984)

    return bozeman, navajo, test_point


def extract_naip(file_prefix, points_layer, roi, year):
    roi = ee.FeatureCollection(roi)
    naip = ee.ImageCollection("USDA/NAIP/DOQQ").filterDate('{}-01-01'.format(year), '{}-12-31'.format(year))
    naip = naip.filterBounds(roi).mosaic()

    props = ['POINT_TYPE']

    points = ee.FeatureCollection(points_layer)
    points = points.filterBounds(roi)

    plot_sample_regions = naip.sampleRegions(
        collection=points,
        properties=props,
        scale=1,
        tileScale=16)

    task = ee.batch.Export.table.toCloudStorage(
        plot_sample_regions,
        description='{}_{}'.format(file_prefix, year),
        bucket='wudr',
        fileNamePrefix='{}_{}'.format(file_prefix, year),
        fileFormat='CSV')

    task.start()


def extract_terraclimate_monthly(tables, years, description):
    fc = ee.FeatureCollection(tables)
    for yr in years:
        for m in range(1, 13):
            m_str, m_str_next = str(m).rjust(2, '0'), str(m + 1).rjust(2, '0')
            if m == 12:
                dataset = ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE').filterDate('{}-{}-01'.format(yr, m_str),
                                                                                     '{}-{}-31'.format(yr, m_str))
            else:
                dataset = ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE').filterDate('{}-{}-01'.format(yr, m_str),
                                                                                     '{}-{}-01'.format(yr, m_str_next))
            area = ee.Image.pixelArea()
            pet = dataset.select('pet').first().multiply(0.1).multiply(area).rename('etr')
            soil = dataset.select('soil').first().multiply(0.1).multiply(area).rename('sm')
            ppt = dataset.select('pr').first().multiply(area).rename('ppt')

            bands = pet.addBands([soil, ppt])
            data = bands.reduceRegions(collection=fc,
                                       reducer=ee.Reducer.sum())

            out_desc = '{}_{}_{}'.format(description, yr, m_str)
            task = ee.batch.Export.table.toCloudStorage(
                data,
                description=out_desc,
                bucket='wudr',
                fileNamePrefix=out_desc,
                fileFormat='CSV',
                selectors=['STAID', 'etr', 'sm', 'ppt'])

            task.start()
            print(out_desc)


def export_gridded_data(tables, bucket, years, description, features=None, min_years=0, debug=False):
    """
    Reduce Regions, i.e. zonal stats: takes a statistic from a raster within the bounds of a vector.
    Use this to get e.g. irrigated area within a county, HUC, or state. This can mask based on Crop Data Layer,
    and can mask data where the sum of irrigated years is less than min_years. This will output a .csv to
    GCS wudr bucket.
    :param features:
    :param bucket:
    :param tables: vector data over which to take raster statistics
    :param years: years over which to run the stats
    :param description: export name append str
    :param cdl_mask:
    :param min_years:
    :return:
    """
    fc = ee.FeatureCollection(tables)
    if features:
        fc = fc.filter(ee.Filter.inList('BASINNUM', features))
    cmb_clip = ee.FeatureCollection(CMBRB_CLIP)
    umrb_clip = ee.FeatureCollection(UMRB_CLIP)
    corb_clip = ee.FeatureCollection(CORB_CLIP)

    irr_coll = ee.ImageCollection(RF_ASSET)
    coll = irr_coll.filterDate('1987-01-01', '2021-12-31').select('classification')
    remap = coll.map(lambda img: img.lt(1))
    irr_min_yr_mask = remap.sum().gt(min_years)
    # sum = remap.sum().mask(irr_mask)

    for yr in years:
        for month in range(4, 11):
            s = '{}-{}-01'.format(yr, str(month).rjust(2, '0'))
            end_day = monthrange(yr, month)[1]
            e = '{}-{}-{}'.format(yr, str(month).rjust(2, '0'), end_day)

            irr = irr_coll.filterDate('{}-01-01'.format(yr), '{}-12-31'.format(yr)).select('classification').mosaic()
            irr_mask = irr_min_yr_mask.updateMask(irr.lt(1))

            annual_coll = ee.ImageCollection('users/dgketchum/ssebop/cmbrb').merge(
                ee.ImageCollection('users/hoylmanecohydro2/ssebop/cmbrb'))
            et_coll = annual_coll.filter(ee.Filter.date(s, e))
            et_cmb = et_coll.sum().multiply(0.00001).clip(cmb_clip.geometry())

            annual_coll = ee.ImageCollection('users/kelseyjencso/ssebop/corb').merge(
                ee.ImageCollection('users/dgketchum/ssebop/corb')).merge(
                ee.ImageCollection('users/dpendergraph/ssebop/corb'))
            et_coll = annual_coll.filter(ee.Filter.date(s, e))
            et_corb = et_coll.sum().multiply(0.00001).clip(corb_clip.geometry())

            annual_coll_ = ee.ImageCollection('projects/usgs-ssebop/et/umrb')
            et_coll = annual_coll_.filter(ee.Filter.date(s, e))
            et_umrb = et_coll.sum().multiply(0.00001).clip(umrb_clip.geometry())

            et_sum = ee.ImageCollection([et_cmb, et_corb, et_umrb]).mosaic()
            et = et_sum.mask(irr_mask)

            tclime = ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE").filterDate(s, e).select('aet')
            tclime_red = ee.Reducer.sum()
            tclime_sums = tclime.select('aet').reduce(tclime_red)
            swb_aet = tclime_sums.select('aet_sum').mask(irr_mask).multiply(0.0001)

            ppt, etr = extract_gridmet_monthly(yr, month)

            irr_mask = irr_mask.reproject(crs='EPSG:5070', scale=30)
            et = et.reproject(crs='EPSG:5070', scale=30).resample('bilinear')
            ppt = ppt.reproject(crs='EPSG:5070', scale=30).resample('bilinear')
            ppt_irr = ppt.mask(irr_mask).rename('ppt_irr')
            etr = etr.reproject(crs='EPSG:5070', scale=30).resample('bilinear')
            swb_aet = swb_aet.reproject(crs='EPSG:5070', scale=30).resample('bilinear')

            cc = et.subtract(swb_aet)

            area = ee.Image.pixelArea()
            irr = irr_mask.multiply(area).rename('irr')
            et = et.multiply(area).rename('et')
            cc = cc.multiply(area).rename('cc')
            ppt = ppt.multiply(area).rename('ppt')
            ppt_irr = ppt_irr.multiply(area).rename('ppt_irr')
            etr = etr.multiply(area).rename('etr')
            swb_aet = swb_aet.multiply(area).rename('swb_aet')

            if yr > 1986 and month in np.arange(4, 11):
                bands = irr.addBands([et, cc, ppt, etr, swb_aet, ppt_irr])
                select_ = ['BASINNUM', 'BASINNAME', 'irr', 'et', 'cc', 'ppt', 'etr', 'swb_aet', 'ppt_irr']
            else:
                bands = ppt.addBands([etr, ppt])
                select_ = ['BASINNUM', 'BASINNAME', 'ppt', 'etr']

            data = bands.reduceRegions(collection=fc,
                                       reducer=ee.Reducer.sum(),
                                       scale=30)

            if debug:
                pt = bands.sample(region=get_geomteries()[2],
                                  numPixels=1,
                                  scale=30)
                fields = pt.propertyNames().remove('.geo')
                p = data.first().getInfo()['properties']

            out_desc = '{}_{}_{}'.format(description, yr, month)
            task = ee.batch.Export.table.toCloudStorage(
                data,
                description=out_desc,
                bucket=bucket,
                fileNamePrefix=out_desc,
                fileFormat='CSV',
                selectors=select_)
            task.start()
            print(out_desc)


def extract_gridmet_monthly(year, month):
    m_str, m_str_next = str(month).rjust(2, '0'), str(month + 1).rjust(2, '0')
    if month == 12:
        dataset = ee.ImageCollection('IDAHO_EPSCOR/GRIDMET').filterDate('{}-{}-01'.format(year, m_str),
                                                                        '{}-{}-31'.format(year, m_str))
    else:
        dataset = ee.ImageCollection('IDAHO_EPSCOR/GRIDMET').filterDate('{}-{}-01'.format(year, m_str),
                                                                        '{}-{}-01'.format(year, m_str_next))
    pet = dataset.select('etr').sum().multiply(0.001).rename('gm_etr')
    ppt = dataset.select('pr').sum().multiply(0.001).rename('gm_ppt')
    return ppt, pet


def get_landcover_info(basin, glob='none'):
    roi = ee.FeatureCollection('users/dgketchum/boundaries/{}'.format(basin))

    dem = ee.Terrain.products(ee.Image('USGS/NED')).select('elevation')

    clay = ee.Image('projects/openet/soil/ssurgo_Clay_WTA_0to152cm_composite').select(['b1']).rename('clay')
    sand = ee.Image('projects/openet/soil/ssurgo_Sand_WTA_0to152cm_composite').select(['b1']).rename('sand')
    loam = ee.Image(100).subtract(clay).subtract(sand).rename('loam')
    ksat = ee.Image('projects/openet/soil/ssurgo_Ksat_WTA_0to152cm_composite').select(['b1']).rename('ksat')
    awc = ee.Image('projects/openet/soil/ssurgo_AWC_WTA_0to152cm_composite').select(['b1']).rename('awc')

    clay = clay.multiply(100).toUint16()
    sand = sand.multiply(100).toUint16()
    loam = loam.multiply(100).toUint16()

    ksat = ksat.multiply(100).toUint16()
    awc = awc.multiply(100).toUint16()

    nlcd = ee.Image('USGS/NLCD/NLCD2011').select('impervious').rename('nlcd').toUint8()
    landfire_cov = ee.Image('LANDFIRE/Vegetation/EVC/v1_4_0/CONUS').toUint16()
    landfire_type = ee.Image('LANDFIRE/Vegetation/EVT/v1_4_0/CONUS').toUint16()

    # pt = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([-110.64, 45.45])).set('FID', 1)])
    # data = dem.sampleRegions(collection=pt,
    #                          scale=30)
    # pprint(data.getInfo())

    for prop, var in zip(
        ['clay', 'sand', 'loam', 'ksat', 'awc', 'nlcd', 'elevation', 'landfire_cover', 'landfire_type'],
        [clay, sand, loam, ksat, awc, nlcd, dem, landfire_cov, landfire_type]):
        if 'nlcd' not in prop:
            continue
        desc = '{}_{}_{}'.format(prop, basin, glob)
        task = ee.batch.Export.image.toCloudStorage(
            var,
            fileNamePrefix=desc,
            region=roi.first().geometry(),
            description=desc,
            fileFormat='GeoTIFF',
            bucket='wudr',
            scale=30,
            maxPixels=1e13,
            crs="EPSG:5071")

        task.start()
        print(desc)


def attribute_irrigation(collection, polygons, years):
    """
    Extracts fraction of vector classified as irrigated. Been using this to attribute irrigation to
    field polygon coverages.
    :return:
    """
    fc = ee.FeatureCollection(polygons)
    for state in ['MT']:
        for yr in years:
            images = os.path.join(collection, '{}_{}'.format(state, yr))
            coll = ee.Image(images)
            tot = coll.select('classification').remap([0, 1, 2, 3], [1, 0, 0, 0]).reproject(crs='EPSG:5070', scale=30)
            area = ee.Image.pixelArea()

            tot = tot.multiply(area).rename('irr')
            areas = tot.reduceRegions(collection=fc,
                                      reducer=ee.Reducer.sum(),
                                      scale=30)

            task = ee.batch.Export.table.toCloudStorage(
                areas,
                description='{}_{}'.format(state, yr),
                bucket='wudr',
                fileNamePrefix='Milk_{}_{}'.format(state, yr),
                fileFormat='CSV',
                selectors=['HUC8', 'sum'])

            print(state, yr)
            task.start()


if __name__ == '__main__':
    is_authorized()
    # feats = [k for k, v in BASINS.items() if v[1]]
    # export_gridded_data(DNRC_BASINS, 'wudr', years=[i for i in range(1987, 2022)],
    #                     description='basins_30OCT2022', min_years=5, features=None)
    region = 'users/dgketchum/boundaries/MT'
    pts = 'users/dgketchum/hydrography/naip_pts'
    year_ = 2017
    desc = 'naip'
    extract_naip(desc, pts, region, year_)
# ========================= EOF ================================================================================
