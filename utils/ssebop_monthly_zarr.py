import os

import fsspec
import geopandas as gpd
import pandas as pd
import xarray as xr


def extract_zonal_stats(shapefile_path, zarr_path, endpoint_url):
    """"""
    fs = fsspec.filesystem('s3', anon=True, endpoint_url=endpoint_url)
    ds = xr.open_dataset(fs.get_mapper(zarr_path), engine='zarr', backend_kwargs={'consolidated': True})
    ds['time'] = pd.to_datetime(ds['time'].values)
    print(ds)

    gdf = gpd.read_file(shapefile_path)
    gdf = gdf.to_crs('epsg:4326')

    results = []
    for i, row in gdf.iterrows():
        clipped_ds = ds.sel(lat=slice(row['geometry'].bounds[1], row['geometry'].bounds[3]),
                            lon=slice(row['geometry'].bounds[0], row['geometry'].bounds[2]))

        masked_da = clipped_ds['et'].where(clipped_ds['et'] != clipped_ds['et'].rio.nodata)
        time_series = masked_da.mean(dim=['lat', 'lon'], skipna=True).values

        results.append(time_series)

    df = pd.DataFrame(results).T
    df.index = ds['time'].values
    df.to_csv('zonal_stats.csv', index_label='datetime')


if __name__ == '__main__':
    home = os.path.expanduser('~')
    shapefile_path_ = os.path.join(home, 'data', 'IrrigationGIS', 'gnss',
                                   'sierra_bd_merged', 'sierra_bd_merged_5071.shp')
    zarr_path_ = 's3://mdmf/gdp/ssebopeta_monthly.zarr'
    endpoint_url_ = 'https://usgs.osn.mghpcc.org/'

    extract_zonal_stats(shapefile_path_, zarr_path_, endpoint_url_)
# ========================= EOF ====================================================================
