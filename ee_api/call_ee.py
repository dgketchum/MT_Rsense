import os
import sys
from calendar import monthrange
from pprint import pprint

import fiona
import numpy as np
from pandas import read_csv
import ee
from matplotlib import pyplot as plt
from scipy.stats import linregress

from ee_api import is_authorized

sys.path.insert(0, os.path.abspath('..'))
sys.setrecursionlimit(5000)

RF_ASSET = 'projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp'
# RF_ASSET = 'users/dgketchum/IrrMapper/IrrMapper_sw'
BASINS = 'users/dgketchum/gages/gage_basins'
COUNTIES = 'users/dgketchum/boundaries/western_11_co_study'

FIELDS = 'users/dgketchum/boundaries/MilkHUC8'

UMRB_CLIP = 'users/dgketchum/boundaries/umrb_ylstn_clip'
CMBRB_CLIP = 'users/dgketchum/boundaries/CMB_RB_CLIP'
CORB_CLIP = 'users/dgketchum/boundaries/CO_RB'

DNRC_BASINS = 'users/dgketchum/boundaries/DNRC_Basins'

FLUX_SHP = '/media/research/IrrigationGIS/ameriflux/select_flux_sites/select_flux_sites_impacts_ECcorrrected.shp'
FLUX_DIR = '/media/research/IrrigationGIS/ameriflux/ec_data/monthly'


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


def extract_gridded_data(tables, years=None, description=None, min_years=0, mask_swb_to_irr=True, volume=False,
                         mask_ppt_to_irr=False):
    """
    Reduce Regions, i.e. zonal stats: takes a statistic from a raster within the bounds of a vector.
    Use this to get e.g. irrigated area within a county, HUC, or state. This can mask based on Crop Data Layer,
    and can mask data where the sum of irrigated years is less than min_years. This will output a .csv to
    GCS wudr bucket.
    :param tables: vector data over which to take raster statistics
    :param years: years over which to run the stats
    :param description: export name append str
    :param cdl_mask:
    :param min_years:
    :return:
    """
    fc = ee.FeatureCollection(tables)
    cmb_clip = ee.FeatureCollection(CMBRB_CLIP)
    umrb_clip = ee.FeatureCollection(UMRB_CLIP)
    corb_clip = ee.FeatureCollection(CORB_CLIP)

    # fc = ee.FeatureCollection(ee.FeatureCollection(tables).filter(ee.Filter.eq('FID', 4333)))

    irr_coll = ee.ImageCollection(RF_ASSET)
    coll = irr_coll.filterDate('1991-01-01', '2020-12-31').select('classification')
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

            senay = ee.ImageCollection('projects/usgs-ssebop/et/conus/monthly/v4').filter(ee.Filter.date(s, e))
            senay = senay.select('et_actual').sum().divide(1000.0)

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

            tclime = ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE").filterDate(s, e).select('pr', 'pet', 'aet')
            tclime_red = ee.Reducer.sum()
            tclime_sums = tclime.select('pr', 'pet', 'aet').reduce(tclime_red)
            ppt = tclime_sums.select('pr_sum').multiply(0.001)
            etr = tclime_sums.select('pet_sum').multiply(0.0001)

            if mask_swb_to_irr:
                swb_aet = tclime_sums.select('aet_sum').mask(irr_mask).multiply(0.0001)
            else:
                swb_aet = tclime_sums.select('aet_sum').multiply(0.0001)

            irr_mask = irr_mask.reproject(crs='EPSG:5070', scale=30)
            et = et.reproject(crs='EPSG:5070', scale=30).resample('bilinear')
            ppt = ppt.reproject(crs='EPSG:5070', scale=30).resample('bilinear')
            if mask_ppt_to_irr:
                ppt = ppt.mask(irr_mask)
            etr = etr.reproject(crs='EPSG:5070', scale=30).resample('bilinear')
            swb_aet = swb_aet.reproject(crs='EPSG:5070', scale=30).resample('bilinear')

            cc = et.subtract(swb_aet)

            area = ee.Image.pixelArea()

            if volume:
                irr = irr_mask.multiply(area).rename('irr')
                et = et.multiply(area).rename('et')
                cc = cc.multiply(area).rename('cc')
                ppt = ppt.multiply(area).rename('ppt')
                etr = etr.multiply(area).rename('etr')
                swb_aet = swb_aet.multiply(area).rename('aet')
            else:
                irr = irr_mask.multiply(area).rename('irr')
                et = et.rename('et_ketchum')
                senay = senay.rename('et_senay')
                cc = cc.rename('cc')
                ppt = ppt.rename('ppt')
                etr = etr.rename('etr')
                swb_aet = swb_aet.rename('aet')

            selector = ['BASINNUM', 'BASINNAME']
            if volume:
                bands = irr.addBands([et, cc, ppt, etr, swb_aet])
                select_ = selector + ['irr', 'et', 'cc', 'ppt', 'etr', 'swb_aet']
            else:
                bands = irr.addBands([et, cc, ppt, etr, swb_aet, senay])
                select_ = selector + ['irr', 'et', 'cc', 'ppt', 'etr', 'swb_aet', 'senay']

            if volume:
                data = bands.reduceRegions(collection=fc,
                                           reducer=ee.Reducer.sum(),
                                           scale=30)
            else:
                data = bands.reduceRegions(collection=fc,
                                           reducer=ee.Reducer.mean(),
                                           scale=30)

            # fields = data.first().propertyNames().remove('.geo')
            # p = data.first().getInfo()['properties']

            out_desc = '{}_{}_{}'.format(description, yr, month)
            task = ee.batch.Export.table.toCloudStorage(
                data,
                description=out_desc,
                bucket='wudr',
                fileNamePrefix=out_desc,
                fileFormat='CSV',
                selectors=select_)

            task.start()
            print(out_desc)


def extract_flux_stations(shp):
    with fiona.open(shp, 'r') as src:
        dct = {}
        for feat in src:
            p = feat['properties']
            if p['basin'] == 'umrb':
                dct[p['site_id']] = p
                dct[p['site_id']]['clip_feat'] = UMRB_CLIP
                dct[p['site_id']]['geo'] = feat['geometry']['coordinates']
            elif p['basin'] == 'cmbrb':
                dct[p['site_id']] = p
                dct[p['site_id']]['clip_feat'] = CMBRB_CLIP
                dct[p['site_id']]['geo'] = feat['geometry']['coordinates']
            else:
                continue

    et_comp = []
    for site, props in dct.items():

        if props['basin'] == 'cmbrb':
            annual_coll = ee.ImageCollection('users/dgketchum/ssebop/cmbrb').merge(
                ee.ImageCollection('users/hoylmanecohydro2/ssebop/cmbrb'))
        elif props['basin'] == 'umrb':
            annual_coll = ee.ImageCollection('projects/usgs-ssebop/et/umrb')
        else:
            continue

        _file = '{}_monthly_data.csv'.format(props['site_id'])
        csv = os.path.join(FLUX_DIR, _file)
        df = read_csv(csv)
        df = df[df['ET_corr'].notna()]
        et_ssebop = []
        dates = [('{}-01'.format(x[:7]), x) for x in df.date.values]
        et_corr = [x for x in df['ET_corr'].values]

        geo = ee.Geometry.Point(props['geo'][0], props['geo'][1]).buffer(3.5 * 30.0)

        for et_ec, (s, e) in zip(et_corr, dates):
            et_coll = annual_coll.filter(ee.Filter.date(s, e))
            et = et_coll.sum().multiply(0.01)

            data = et.reduceRegion(geometry=geo,
                                   reducer=ee.Reducer.mean(),
                                   scale=30)
            try:
                ee_obj = data.getInfo()
                et_extract = ee_obj['et']
                if et_extract is None:
                    et_extract = np.nan
            except (ee.EEException, KeyError):
                et_extract = np.nan

            et_ssebop.append(et_extract)
            if not np.any(np.isnan([et_ec, et_extract])):
                et_comp.append((et_ec, et_extract))

            print('{}: {:.2f} et, {:.2f} ssebop, {}'.format(site, et_ec, et_extract, s))

        df['et_ssebop'] = et_ssebop
        _file = '{}_monthly_data_ee.csv'.format(props['site_id'])
        csv = os.path.join(FLUX_DIR, _file)
        df.to_csv(csv)

    scatter = '/home/dgketchum/Downloads/et_comp.png'
    et_ec = np.array([x[0] for x in et_comp])
    et_model = np.array([x[1] for x in et_comp])
    rmse = np.sqrt(np.mean((et_model - et_ec) ** 2))
    lin = np.arange(0, 200)
    plt.plot(lin, lin)
    # m, b = np.polyfit(et_ec, et_model, 1)
    m, b, r, p, stderr = linregress(et_ec, et_model)
    plt.annotate('{:.2f}x + {:.2f}\n rmse: {:.2f} mm/month\n r2: {:.2f}'.format(m, b, rmse, r ** 2),
                 xy=(0.05, 0.75), xycoords='axes fraction')
    plt.plot(lin, m * lin + b)
    plt.scatter(et_ec, et_model)
    plt.ylabel('SSEBop ET')
    plt.xlabel('Eddy Covariance')
    plt.savefig(scatter)


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

    extract_gridded_data(DNRC_BASINS, years=[i for i in range(1991, 2021)],
                         description='DNRC_Basins_3OCT2022', min_years=5,
                         mask_swb_to_irr=True, volume=True, mask_ppt_to_irr=True)

    # attribute_irrigation(RF_ASSET, FIELDS, years=[x for x in range(1986, 2022)])

    # get_landcover_info('huc6_MT_intersect_dissolve', glob='7MAR2022')
# ========================= EOF ================================================================================
