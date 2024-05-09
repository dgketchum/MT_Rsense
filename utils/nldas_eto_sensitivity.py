import os
import pytz
from datetime import timedelta

import numpy as np
import pandas as pd
import pynldas2 as nld
from refet import Daily, calcs

VAR_MAP = {'rsds': 'Rs (w/m2)',
           'humidity': 'mean',
           'min_temp': 'TMin (C)',
           'max_temp': 'TMax (C)',
           'temp': 'TAvg (C)',
           'wind': 'Windspeed (m/s)',
           'eto': 'ETo (mm)'}

RESAMPLE_MAP = {'rsds': 'sum',
                'humidity': 'mean',
                'min_temp': 'min',
                'max_temp': 'max',
                'wind': 'mean',
                'doy': 'first'}

RENAME_MAP = {v: k for k, v in RESAMPLE_MAP.items()}

COMPARISON_VARS = ['rsds', 'humidity', 'min_temp', 'max_temp', 'wind', 'eto']


def sensitivity(stations, station_data, results, station_type='ec', check_dir=None):
    if station_type == 'ec':
        kw = {'index': 'SITE_ID',
              'lat': 'LATITUDE',
              'lon': 'LONGITUDE',
              'elev': 'ELEVATION (METERS)',
              'start': 'START DATE',
              'end': 'END DATE'}
    elif station_type == 'agri':
        kw = {'index': 'id',
              'lat': 'latitude',
              'lon': 'longitude',
              'elev': 'elev_m',
              'start': 'record_start',
              'end': 'record_end'}
    else:
        raise NotImplementedError

    station_list = pd.read_csv(stations, index_col=kw['id'])

    for index, row in station_list.iterrows():

        if index != 'shbm':
            continue

        s = pd.to_datetime(row[kw['start']]) - timedelta(days=1)
        e = pd.to_datetime(row[kw['end']]) + timedelta(days=2)
        nldas = nld.get_bycoords((row[kw['lon']], row[kw['lat']]), start_date=s, end_date=e,
                                 variables=['temp', 'wind_u', 'wind_v', 'humidity', 'rsds'])

        central = pytz.timezone('US/Pacific')
        nldas = nldas.tz_convert(central)

        wind_u = nldas['wind_u']
        wind_v = nldas['wind_v']
        nldas['wind'] = np.sqrt(wind_v ** 2 + wind_u ** 2)

        nldas['min_temp'] = nldas['temp'] - 273.15
        nldas['max_temp'] = nldas['temp'] - 273.15
        nldas['doy'] = [i.dayofyear for i in nldas.index]

        nldas = nldas.resample('D').agg(RESAMPLE_MAP)
        nldas['rsds'] *= 0.0036
        nldas['ea'] = calcs._actual_vapor_pressure(pair=calcs._air_pressure(row[kw['elev']]),
                                                   q=nldas['humidity'])
        zw = 10.0 if station_type == 'ec' else row['anemom_height_m']

        def calc_eto(r, zw):
            eto = Daily(tmin=r['min_temp'],
                        tmax=r['max_temp'],
                        ea=r['ea'],
                        rs=r['rsds'],
                        uz=r['wind'],
                        zw=zw,
                        doy=r['doy'],
                        elev=row[kw['elev']],
                        lat=row[kw['lat']]).eto()[0]
            return eto

        nldas['eto'] = nldas.apply(calc_eto, zw=zw, axis=1)
        if check_dir:
            check_file = os.path.join(check_dir, '{}_nldas_daily.csv'.format(index))
            cdf = pd.read_csv(check_file, parse_dates=True, index_col='date')
            cdf.index = cdf.index.tz_localize(central)
            indx = [i for i in cdf.index if i in nldas.index]
            rsq = np.corrcoef(nldas.loc[indx, 'eto'], cdf.loc[indx, 'eto_asce'])[0, 0]
            print('PyNLDAS/Earth Engine r2: {:.3f}'.format(rsq))

        sdf_file = os.path.join(station_data, '{}_output.xlsx'.format(index))
        sdf = pd.read_excel(sdf_file, parse_dates=True, index_col='date')
        idx = [i for i in nldas.index if i in sdf.index]
        nldas = nldas.loc[idx].copy()


if __name__ == '__main__':
    d = '/media/research/IrrigationGIS/milk'
    # sta = os.path.join(d, '/eddy_covariance_data_processing/eddy_covariance_stations.csv'
    sta = os.path.join(d, 'bias_ratio_data_processing/ETo/'
                          'final_milk_river_metadata_nldas_eto_bias_ratios_long_term_mean.csv')
    sta_data = os.path.join(d, 'weather_station_data_processing/corrected_data')
    res = os.path.join(d, 'eddy_covariance_nldas_analysis')

    ee_check = os.path.join(d, 'weather_station_data_processing/NLDAS_data_at_stations')
    sensitivity(sta, sta_data, res, station_type='agri', check_dir=None)
# ========================= EOF ====================================================================
