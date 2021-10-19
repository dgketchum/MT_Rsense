import numpy as np
import requests
import xarray as xr
from dask.array import where, unique, isin, from_array, map_blocks
from pandas import DataFrame

QA_VALUES = [322.0, 324.0, 328.0, 336.0, 352.0, 368.0, 386.0, 388.0,
             392.0, 400.0, 416.0, 432.0, 480.0, 834.0, 836.0, 840.0,
             848.0, 864.0, 880.0, 898.0, 904.0, 912.0, 928.0, 944.0,
             992.0]


def qualityDecode(value):
    product_id = 'CU_LC08.001'
    quality_layer = 'PIXELQA'
    quality_value = value
    response = requests.get('https://lpdaacsvc.cr.usgs.gov/appeears/api/quality/{}/{}/{}'.format(product_id,
                                                                                                 quality_layer,
                                                                                                 quality_value))
    quality_response = response.json()
    return quality_response


def mask_func(ndvi, qa, mask_values):
    ndvi_ = np.where(np.isin(qa, mask_values), ndvi, np.ma.masked)
    ndvi_ = (ndvi_ * 100).astype(np.int16)
    ndvi_[ndvi_ > 100] = 100
    ndvi_[ndvi_ < -100] = -100
    return ndvi_


def get_landsat_ndvi_time_series(_file, out_):
    chunks = {'time': 1, 'lat': 1000, 'lon': 1000}
    ds = xr.open_dataset(_file, chunks=chunks, )
    qa = ds['PIXELQA'].data

    valid_ = qa[:, 2000, 2000].compute()
    ds = ds.sel(time=ds['time'][~np.isnan(valid_)])
    qa = ds['PIXELQA'].data
    ndvi = (ds['SRB5'].data - ds['SRB4'].data) / (ds['SRB4'].data + ds['SRB5'].data)

    # unq = unique(da_qa).compute()
    quality_values = DataFrame(QA_VALUES, columns=['value']).dropna()
    quality_desc = DataFrame(columns=['value', 'Cloud_bits', 'Cloud_description', 'CS_bits', 'CS_description'])
    for index, row in quality_values.iterrows():
        quality_info = qualityDecode(str(int(row['value'])))
        df = DataFrame({'value': int(row['value']),
                        'Cloud_bits': quality_info['Cloud']['bits'],
                        'Cloud_description': quality_info['Cloud']['description'],
                        'CS_bits': quality_info['Cloud Shadow']['bits'],
                        'CS_description': quality_info['Cloud Shadow']['description']}, index=[index])

        quality_desc = quality_desc.append(df)

    mask_values = quality_desc[((quality_desc['Cloud_description'] == 'No') &
                                (quality_desc['CS_description'] == 'No'))]

    mask_values = mask_values['value'].values

    ndvi = map_blocks(mask_func, ndvi, qa, mask_values).compute()

    da = xr.DataArray(dims=['time', 'lat', 'lon'],
                      data=ndvi, coords={'lat': ds.lat,
                                         'lon': ds.lon,
                                         'time': ds.time})
    print('write nc')
    da.to_netcdf(out_)
    print('write tif')
    da.rio.to_raster(out_.replace('.nc', '.tif'))


if __name__ == '__main__':
    nc = '/media/research/IrrigationGIS/Montana/water_rights/landsat/nc/CU_LC08.001_30m_aid0001.nc'
    nc_out = '/media/research/IrrigationGIS/Montana/water_rights/landsat/nc/ndvi_2015.nc'
    get_landsat_ndvi_time_series(nc, nc_out)
# ========================= EOF ====================================================================
