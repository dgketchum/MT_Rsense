"""
This file contains Python functions for accessing weather and climate data from
the National Weather Service through both their thredds and NOMADS servers.

    thredds - http://motherlode.ucar.edu/thredds
    NOMADS  - http://nomads.ncep.noaa.gov/pub/data/nccf/com

In particular, these functions access the following forecast model datasets:

    WEATHER DATA
        Global Ensemble Forecast System (GEFS)
            https://www.ncdc.noaa.gov/data-access/model-data/model-datasets/global-ensemble-forecast-system-gefs
        North American Mesoscale (NAM) Forecasting System
            https://www.ncdc.noaa.gov/data-access/model-data/model-datasets/north-american-mesoscale-forecast-system-nam

    CLIMATE DATA
        Climate Forecasting System, version 2 (CFSv2)
            https://www.ncdc.noaa.gov/data-access/model-data/model-datasets/climate-forecast-system-version2-cfsv2


Created on Thu Aug 06 13:43:27 2015
@author: Von P. Walden, Washington State University
"""

from netcdf4 import Dataset
from datetime import date, timedelta
import sys
from numpy import argmin, argmax


def get_gefs_temps(lat, lon, date_range=None):
    """ Function to retrieve GEFS data from yesterday."""

    print('Opening connection to motherlode.ucar.edu...')

    dateNumber = (date.today() - timedelta(days=1)).strftime('%Y%m%d')  # For yesterday
    # dateNumber = date.today().strftime('%Y%m%d')                       # For today
    fn = 'http://thredds.ucar.edu/thredds/dodsC/grib/NCEP/GEFS/Global_1p0deg_Ensemble/members/GEFS_Global_1p0deg_Ensemble_' + dateNumber + '_0000.grib2'
    try:
        gefs = Dataset(fn)
    except:
        print(
        'Data is NOT currently available from http://motherlode.ucar.edu/thredds/dodsC/grib/NCEP/GFS/Global_0p5deg/files/GFS_Global_0p5deg_' + dateNumber + '_0000.grib2')
        sys.exit()

    # The "gfs" reference in python is a set of nested dictionaries.
    # Typing "gfs" at the python prompt will list the available variables.
    # However, it is more convenient to list the variables from the web:
    #   https://urldefense.proofpoint.com/v1/url?u=http://motherlode.ucar.edu/thredds/dodsC/grib/NCEP/GFS/Global_0p5deg/&k=EWEYHnIvm0nsSxnW5y9VIw%3D%3D%0A&r=SaGxFvOtc0S6i84p5aTSgw%3D%3D%0A&m=MDDizvB1ONxosQyHpM4thYu8V%2Bkp9Hji%2BDhdZ3ZPP4k%3D%0A&s=251a44412ee52155e0264c63f2edfce386ac88bac7991ae5c5679b1460e9020f
    # List the contents of a single variable like:
    #   gfs.Temperature_surface
    # This object is a dictionary with four variables: lat, lon, time, Temperature_surface

    # Load time series
    print('  Loading time...')
    gefsTime = gefs.variables['time2'][:]  # These times (in hours) are relative to the current day at 0000 UTC.

    # Load lat/lon grid
    print('  Loading latitude and longitude...')
    gefslat = gefs.variables['lat'][:]
    gefslon = gefs.variables['lon'][:]
    # Determine desired grid pixel given (lat, lon).
    #    ....Convert longitude to 0-360 degree.
    if lon < 0.:
        lon = 180. + (180. + lon)
    # ....Determine grid cell.
    ilat = argmax(lat > gefslat)
    ilon = argmin(lon > gefslon)

    # Extract the forecasted time series of Surface Temperature
    print('  Loading surface temperature...')
    gefsT2m = (gefs.variables['Temperature_height_above_ground_ens'][:, :, 0, ilat, ilon] - 273.15) * 1.8 + 32.

    return gefsTime, gefsT2m


def getNAMtemperatures(lat, lon):
    """ Function to retrieve NAM forecast data from yesterday."""

    print('Opening connection to motherlode.ucar.edu...')
    dateNumber = (date.today() - timedelta(days=1)).strftime('%Y%m%d')  # For yesterday
    #    dateNumber = date.today().strftime('%Y%m%d')                       # For today
    fn = 'http://thredds.ucar.edu/thredds/dodsC/grib/NCEP/NAM/CONUS_12km/NAM_CONUS_12km_' + dateNumber + '_0000.grib2'
    try:
        nam = Dataset(fn)
    except:
        print(
        'Data is NOT currently available from http://motherlode.ucar.edu/thredds/dodsC/grib/NCEP/GFS/Global_0p5deg/files/GFS_Global_0p5deg_' + dateNumber + '_0000.grib2')
        sys.exit()

    # Load time series
    print('  Loading time...')
    namtime = nam.variables['time1'][:]  # These times (in hours) are relative to the current day at 0000 UTC.

    # Load lat/lon grid
    print('  Loading latitude and longitude...')
    namlat = nam.variables['lat'][:]
    namlon = nam.variables['lon'][:]
    # Determine desired grid pixel given (lat, lon).
    #    ....Convert longitude to 0-360 degree.
    if lon < 0.:
        lon = 180. + (180. + lon)
    # ....Determine grid cell.
    ilat = argmax(lat > namlat)
    ilon = argmin(lon > namlon)

    # Extract the forecasted time series of Surface Temperature
    print('  Loading surface temperature...')
    namT2m = (nam.variables['Temperature_height_above_ground'][:, 0, ilat, ilon] - 273.15) * 1.8 + 32.

    return namtime, namT2m


def getCFSforecastData():
    """ Function to download CFSv2 data from yesterday."""
    import subprocess
    from datetime import date, timedelta
    dataDirectory = '/Users/vonw/data/cfs/'
    dateNumber = (date.today() - timedelta(days=1)).strftime('%Y%m%d')  # For yesterday

    # Create list of sixteen CFSv2 files to download and process.
    # 00Z
    fns = [
        'http://nomads.ncep.noaa.gov/pub/data/nccf/com/cfs/prod/cfs/cfs.' + dateNumber + '/00/time_grib_01/tmp2m.01.' + dateNumber + '00.daily.grb2 > ' + dataDirectory + 'tmp2m.01.' + dateNumber + '00.daily.grb2']
    fns.append(
        'http://nomads.ncep.noaa.gov/pub/data/nccf/com/cfs/prod/cfs/cfs.' + dateNumber + '/00/time_grib_02/tmp2m.02.' + dateNumber + '00.daily.grb2 > ' + dataDirectory + 'tmp2m.02.' + dateNumber + '00.daily.grb2')
    fns.append(
        'http://nomads.ncep.noaa.gov/pub/data/nccf/com/cfs/prod/cfs/cfs.' + dateNumber + '/00/time_grib_03/tmp2m.03.' + dateNumber + '00.daily.grb2 > ' + dataDirectory + 'tmp2m.03.' + dateNumber + '00.daily.grb2')
    fns.append(
        'http://nomads.ncep.noaa.gov/pub/data/nccf/com/cfs/prod/cfs/cfs.' + dateNumber + '/00/time_grib_04/tmp2m.04.' + dateNumber + '00.daily.grb2 > ' + dataDirectory + 'tmp2m.04.' + dateNumber + '00.daily.grb2')
    # 06Z
    fns.append(
        'http://nomads.ncep.noaa.gov/pub/data/nccf/com/cfs/prod/cfs/cfs.' + dateNumber + '/06/time_grib_01/tmp2m.01.' + dateNumber + '06.daily.grb2 > ' + dataDirectory + 'tmp2m.01.' + dateNumber + '06.daily.grb2')
    fns.append(
        'http://nomads.ncep.noaa.gov/pub/data/nccf/com/cfs/prod/cfs/cfs.' + dateNumber + '/06/time_grib_02/tmp2m.02.' + dateNumber + '06.daily.grb2 > ' + dataDirectory + 'tmp2m.02.' + dateNumber + '06.daily.grb2')
    fns.append(
        'http://nomads.ncep.noaa.gov/pub/data/nccf/com/cfs/prod/cfs/cfs.' + dateNumber + '/06/time_grib_03/tmp2m.03.' + dateNumber + '06.daily.grb2 > ' + dataDirectory + 'tmp2m.03.' + dateNumber + '06.daily.grb2')
    fns.append(
        'http://nomads.ncep.noaa.gov/pub/data/nccf/com/cfs/prod/cfs/cfs.' + dateNumber + '/06/time_grib_04/tmp2m.04.' + dateNumber + '06.daily.grb2 > ' + dataDirectory + 'tmp2m.04.' + dateNumber + '06.daily.grb2')
    # 12Z
    fns.append(
        'http://nomads.ncep.noaa.gov/pub/data/nccf/com/cfs/prod/cfs/cfs.' + dateNumber + '/12/time_grib_01/tmp2m.01.' + dateNumber + '12.daily.grb2 > ' + dataDirectory + 'tmp2m.01.' + dateNumber + '12.daily.grb2')
    fns.append(
        'http://nomads.ncep.noaa.gov/pub/data/nccf/com/cfs/prod/cfs/cfs.' + dateNumber + '/12/time_grib_02/tmp2m.02.' + dateNumber + '12.daily.grb2 > ' + dataDirectory + 'tmp2m.02.' + dateNumber + '12.daily.grb2')
    fns.append(
        'http://nomads.ncep.noaa.gov/pub/data/nccf/com/cfs/prod/cfs/cfs.' + dateNumber + '/12/time_grib_03/tmp2m.03.' + dateNumber + '12.daily.grb2 > ' + dataDirectory + 'tmp2m.03.' + dateNumber + '12.daily.grb2')
    fns.append(
        'http://nomads.ncep.noaa.gov/pub/data/nccf/com/cfs/prod/cfs/cfs.' + dateNumber + '/12/time_grib_04/tmp2m.04.' + dateNumber + '12.daily.grb2 > ' + dataDirectory + 'tmp2m.04.' + dateNumber + '12.daily.grb2')
    # 18Z
    fns.append(
        'http://nomads.ncep.noaa.gov/pub/data/nccf/com/cfs/prod/cfs/cfs.' + dateNumber + '/18/time_grib_01/tmp2m.01.' + dateNumber + '18.daily.grb2 > ' + dataDirectory + 'tmp2m.01.' + dateNumber + '18.daily.grb2')
    fns.append(
        'http://nomads.ncep.noaa.gov/pub/data/nccf/com/cfs/prod/cfs/cfs.' + dateNumber + '/18/time_grib_02/tmp2m.02.' + dateNumber + '18.daily.grb2 > ' + dataDirectory + 'tmp2m.02.' + dateNumber + '18.daily.grb2')
    fns.append(
        'http://nomads.ncep.noaa.gov/pub/data/nccf/com/cfs/prod/cfs/cfs.' + dateNumber + '/18/time_grib_03/tmp2m.03.' + dateNumber + '18.daily.grb2 > ' + dataDirectory + 'tmp2m.03.' + dateNumber + '18.daily.grb2')
    fns.append(
        'http://nomads.ncep.noaa.gov/pub/data/nccf/com/cfs/prod/cfs/cfs.' + dateNumber + '/18/time_grib_04/tmp2m.04.' + dateNumber + '18.daily.grb2 > ' + dataDirectory + 'tmp2m.04.' + dateNumber + '18.daily.grb2')

    # Download CFSv2 data from NOMADS
    for fn in fns:
        print('Downloading ' + fn + ' ...')
        #        returnCode = subprocess.call(['curl',fn])
        returnCode = subprocess.call(['curl', fn, '-o', fn[-30:]])
        if returnCode != 0:
            print('Problem downloading ' + fn + '.  FILE NOT DOWNLOADED...')

    # Convert all CFSv2 GRIB files to netCDF.
    # 00Z
    fns = ['tmp2m.01.' + dateNumber + '00.daily.grb2']
    fns.append('tmp2m.02.' + dateNumber + '00.daily.grb2')
    fns.append('tmp2m.03.' + dateNumber + '00.daily.grb2')
    fns.append('tmp2m.04.' + dateNumber + '00.daily.grb2')
    # 06Z
    fns.append('tmp2m.01.' + dateNumber + '06.daily.grb2')
    fns.append('tmp2m.02.' + dateNumber + '06.daily.grb2')
    fns.append('tmp2m.03.' + dateNumber + '06.daily.grb2')
    fns.append('tmp2m.04.' + dateNumber + '06.daily.grb2')
    # 12Z
    fns.append('tmp2m.01.' + dateNumber + '12.daily.grb2')
    fns.append('tmp2m.02.' + dateNumber + '12.daily.grb2')
    fns.append('tmp2m.03.' + dateNumber + '12.daily.grb2')
    fns.append('tmp2m.04.' + dateNumber + '12.daily.grb2')
    # 18Z
    fns.append('tmp2m.01.' + dateNumber + '18.daily.grb2')
    fns.append('tmp2m.02.' + dateNumber + '18.daily.grb2')
    fns.append('tmp2m.03.' + dateNumber + '18.daily.grb2')
    fns.append('tmp2m.04.' + dateNumber + '18.daily.grb2')

    # Now actual convert the files.
    for fn in fns:
        print('Converting ' + fn + ' to netCDF...')
        returnCode = subprocess.call(['/opt/local/bin/wgrib2', fn, '-netcdf', dataDirectory + '.nc'])
        if returnCode != 0:
            print('Problem converting ' + fn + '.  FILE NOT CONVERTED...')


def getCFSforecasts(lat, lon):
    import numpy as np
    from scipy.io import netcdf
    from datetime import date, timedelta
    dateNumber = (date.today() - timedelta(days=1)).strftime('%Y%m%d')  # For yesterday

    # Read 9-month forecasts
    fns = ['tmp2m.01.' + dateNumber + '00.daily.grb2.nc']
    fns.append('tmp2m.01.' + dateNumber + '06.daily.grb2.nc')
    fns.append('tmp2m.01.' + dateNumber + '12.daily.grb2.nc')
    for i, fn in enumerate(fns):
        cfs = netcdf.netcdf_file(fn)
        if i == 0:
            print('  Loading time...')
            cfsTime_9mo = cfs.variables['time'][:]
            # Load lat/lon grid
            print('  Loading latitude and longitude...')
            cfslat = cfs.variables['latitude'][:]
            cfslon = cfs.variables['longitude'][:]
            # Determine desired grid pixel given (lat, lon).
            #    ....Convert longitude to 0-360 degree.
            if lon < 0.:
                lon = 180. + (180. + lon)
            # ....Determine grid cell.
            ilat = np.argmin(lat > cfslat)
            ilon = np.argmin(lon > cfslon)
            # Load data
            print('  Loading temperature data...')
            cfsT2m_9mo = cfs.variables['TMP_2maboveground'][:, ilat, ilon] - 273.15
        else:
            cfsT2m_9mo = np.vstack((cfsT2m_9mo, cfs.variables['TMP_2maboveground'][:, ilat, ilon] - 273.15))
    cfsT2m_9mo = cfsT2m_9mo.T

    return cfsTime_9mo, cfsT2m_9mo

# gefsTime, Ts = getGEFStemperatures(+46.7,-117.2)
# from bokeh.plotting import figure, output_file, show
# Tools=['pan','box_zoom','resize','wheel_zoom','save','reset']
# Tmean = Ts.mean(axis=1)
# Tstd  = Ts.std(axis=1)
# p = figure(title='GEFS forecasts', plot_width=1000, plot_height=500,tools=Tools)
# p.xaxis.axis_label = 'Forecast Hour'
# p.yaxis.axis_label = '2-m Air Temperature (F)'
# p.patches(xs=[list(gefsTime[:]) + list(gefsTime[::-1])], ys=[list(Tmean[:]-Tstd[:]) + list(Tmean[::-1])], color='red', alpha=0.5 )
# p.patches(xs=[list(gefsTime[:]) + list(gefsTime[::-1])], ys=[list(Tmean[:]) + list(Tmean[::-1]+Tstd[::-1])], color='red', alpha=0.5 )
# p.line(gefsTime, Tmean, line_color='black', line_dash='solid')
# output_file('gefs.html')
# show(p)
