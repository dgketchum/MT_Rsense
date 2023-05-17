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
TONGUE_FIELDS = 'users/dgketchum/fields/tongue_9MAY2023'
UCF_POU = 'users/dgketchum/fields/ucf_pou_10MAY2023'


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


def export_gridded_data(tables, bucket, years, description, features=None, min_years=0, debug=False,
                        join_col='STAID', extra_cols=None, volumes=False):
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
    ee.Initialize()
    fc = ee.FeatureCollection(tables)
    if features:
        fc = fc.filter(ee.Filter.inList('STAID', features))
    cmb_clip = ee.FeatureCollection(CMBRB_CLIP)
    umrb_clip = ee.FeatureCollection(UMRB_CLIP)
    corb_clip = ee.FeatureCollection(CORB_CLIP)

    eff_ppt_coll = ee.ImageCollection('users/dgketchum/expansion/ept')
    eff_ppt_coll = eff_ppt_coll.map(lambda x: x.rename('eff_ppt'))

    irr_coll = ee.ImageCollection(RF_ASSET)
    coll = irr_coll.filterDate('1987-01-01', '2021-12-31').select('classification')
    remap = coll.map(lambda img: img.lt(1))
    irr_min_yr_mask = remap.sum().gte(min_years)

    for yr in years:
        for month in range(1, 13):
            if month != 4:
                continue
            s = '{}-{}-01'.format(yr, str(month).rjust(2, '0'))
            end_day = monthrange(yr, month)[1]
            e = '{}-{}-{}'.format(yr, str(month).rjust(2, '0'), end_day)

            irr = irr_coll.filterDate('{}-01-01'.format(yr), '{}-12-31'.format(yr)).select('classification').mosaic()

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

            eff_ppt = eff_ppt_coll.filterDate(s, e).select('eff_ppt').mosaic()

            ppt, etr = extract_gridmet_monthly(yr, month)
            ietr = extract_corrected_etr(yr, month)

            area = ee.Image.pixelArea()

            irr_mask = irr_min_yr_mask.updateMask(irr.lt(1))
            et = et_sum.mask(irr_mask)
            eff_ppt = eff_ppt.mask(irr_mask).rename('eff_ppt')
            ietr = ietr.mask(irr_mask)
            irr_mask = irr_mask.reproject(crs='EPSG:5070', scale=30)
            irr = irr_mask.multiply(area).rename('irr')

            et = et.reproject(crs='EPSG:5070', scale=30).resample('bilinear').rename('et')
            eff_ppt = eff_ppt.reproject(crs='EPSG:5070', scale=30).resample('bilinear').rename('eff_ppt')
            ppt = ppt.reproject(crs='EPSG:5070', scale=30).resample('bilinear').rename('ppt')
            etr = etr.reproject(crs='EPSG:5070', scale=30).resample('bilinear').rename('etr')
            ietr = ietr.reproject(crs='EPSG:5070', scale=30).resample('bilinear').rename('ietr')

            cc = et.subtract(eff_ppt).rename('cc')

            if volumes:
                et = et.multiply(area)
                eff_ppt = eff_ppt.multiply(area)
                cc = cc.multiply(area)
                ppt = ppt.multiply(area)
                etr = etr.multiply(area)
                ietr = ietr.multiply(area)

            if yr > 1986 and month in range(4, 11):
                # bands = irr.addBands([et, cc, ppt, etr, eff_ppt, ietr])
                # select_ = [join_col, 'irr', 'et', 'cc', 'ppt', 'etr', 'eff_ppt', 'ietr']
                bands = irr.addBands([ppt])
                select_ = [join_col, 'irr']

            else:
                bands = ppt.addBands([etr])
                select_ = [join_col, 'ppt', 'etr']

            if extra_cols:
                select_ += extra_cols

            if debug:
                samp = fc.filterMetadata('FID', 'equals', 1403).geometry()
                field = bands.reduceRegions(collection=samp,
                                            reducer=ee.Reducer.sum(),
                                            scale=30)
                p = field.first().getInfo()['properties']
                print('propeteries {}'.format(p))

            if volumes:
                data = bands.reduceRegions(collection=fc,
                                           reducer=ee.Reducer.sum(),
                                           scale=30)
            else:
                data = bands.reduceRegions(collection=fc,
                                           reducer=ee.Reducer.mean(),
                                           scale=30)

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
    pet = dataset.select('etr').sum().multiply(0.001)
    ppt = dataset.select('pr').sum().multiply(0.001)
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


def extract_corrected_etr(year, month):
    m_str = str(month).rjust(2, '0')
    end_day = monthrange(year, month)[1]
    ic = ee.ImageCollection('projects/openet/reference_et/gridmet/monthly')
    band = ic.filterDate('{}-{}-01'.format(year, m_str), '{}-{}-{}'.format(year, m_str, end_day)).select('etr').first()
    return band.multiply(0.001)


if __name__ == '__main__':
    is_authorized()
    export_gridded_data(TONGUE_FIELDS, 'wudr', years=[i for i in range(1987, 2022)],
                        description='tongue_irr_9MAY2023', min_years=5, features=None,
                        join_col='FID', debug=False, volumes=True)

# ========================= EOF ================================================================================
