import os
import json
import pytz
from datetime import timedelta

import numpy as np
import pandas as pd
import pynldas2 as nld
from refet import Daily, calcs
from scipy.stats import skew, kurtosis

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

VAR_MAP = {'rsds': 'Rs (w/m2)',
           'ea': 'Compiled Ea (kPa)',
           'min_temp': 'TMin (C)',
           'max_temp': 'TMax (C)',
           'temp': 'TAvg (C)',
           'wind': 'Windspeed (m/s)',
           'eto': 'ETo (mm)'}

RESAMPLE_MAP = {'rsds': 'mean',
                'humidity': 'mean',
                'min_temp': 'min',
                'max_temp': 'max',
                'wind': 'mean',
                'doy': 'first'}

RENAME_MAP = {v: k for k, v in VAR_MAP.items()}

COMPARISON_VARS = ['rsds', 'ea', 'min_temp', 'max_temp', 'wind', 'eto']


def error_distribution(stations, station_data, results, station_type='ec', check_dir=None):
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

    station_list = pd.read_csv(stations, index_col=kw['index'])

    errors = {}
    for index, row in station_list.iterrows():

        # if index != 'crsm':
        #     continue

        print(index)

        try:
            sdf_file = os.path.join(station_data, '{}_output.xlsx'.format(index))
            sdf = pd.read_excel(sdf_file, parse_dates=True, index_col='date')

            pacific = pytz.timezone('US/Pacific')
            s = pd.to_datetime(sdf.index[0]) - timedelta(days=2)
            e = pd.to_datetime(sdf.index[-1]) + timedelta(days=2)

            sdf.index = sdf.index.tz_localize(pacific)
            sdf = sdf.rename(RENAME_MAP, axis=1)

            nldas = nld.get_bycoords((row[kw['lon']], row[kw['lat']]), start_date=s, end_date=e,
                                     variables=['temp', 'wind_u', 'wind_v', 'humidity', 'rsds'])

            nldas = nldas.tz_convert(pacific)

            wind_u = nldas['wind_u']
            wind_v = nldas['wind_v']
            nldas['wind'] = np.sqrt(wind_v ** 2 + wind_u ** 2)

            nldas['min_temp'] = nldas['temp'] - 273.15
            nldas['max_temp'] = nldas['temp'] - 273.15
            nldas['doy'] = [i.dayofyear for i in nldas.index]

            nldas = nldas.resample('D').agg(RESAMPLE_MAP)
            nldas['ea'] = calcs._actual_vapor_pressure(pair=calcs._air_pressure(row[kw['elev']]),
                                                       q=nldas['humidity'])
            _zw = 10.0 if station_type == 'ec' else row['anemom_height_m']

            def calc_eto(r, zw):
                eto = Daily(tmin=r['min_temp'],
                            tmax=r['max_temp'],
                            ea=r['ea'],
                            rs=r['rsds'] * 0.0036,
                            uz=r['wind'],
                            zw=zw,
                            doy=r['doy'],
                            elev=row[kw['elev']],
                            lat=row[kw['lat']]).eto()[0]
                return eto

            nldas['eto'] = nldas.apply(calc_eto, zw=_zw, axis=1)
            if check_dir:
                check_file = os.path.join(check_dir, '{}_nldas_daily.csv'.format(index))
                cdf = pd.read_csv(check_file, parse_dates=True, index_col='date')
                cdf.index = cdf.index.tz_localize(pacific)
                indx = [i for i in cdf.index if i in nldas.index]
                rsq = np.corrcoef(nldas.loc[indx, 'eto'], cdf.loc[indx, 'eto_asce'])[0, 0]
                print('{} PyNLDAS/Earth Engine r2: {:.3f}'.format(row['station_name'], rsq))

            dct = {}
            for var in COMPARISON_VARS:
                s_var, n_var = '{}_station'.format(var), '{}_nldas'.format(var)
                df = pd.DataFrame(columns=[s_var], index=sdf.index, data=sdf[var].values)
                df.dropna(how='any', axis=0, inplace=True)
                df[n_var] = nldas.loc[df.index, var].values
                residuals = df[s_var] - df[n_var]
                mean_ = np.mean(residuals).item()
                variance = np.var(residuals).item()
                data_skewness = skew(residuals).item()
                data_kurtosis = kurtosis(residuals).item()
                dct[var] = (mean_, variance, data_skewness, data_kurtosis, np.size(residuals))

            errors[index] = dct.copy()

        except Exception as e:
            print('Exception at {}: {}'.format(index, e))
            errors[index] = 'exception'

    with open(results, 'w') as dst:
        json.dump(errors, dst, indent=4)


if __name__ == '__main__':
    d = '/media/research/IrrigationGIS/milk'
    # sta = os.path.join(d, '/eddy_covariance_data_processing/eddy_covariance_stations.csv'
    sta = os.path.join(d, 'bias_ratio_data_processing/ETo/'
                          'final_milk_river_metadata_nldas_eto_bias_ratios_long_term_mean.csv')
    sta_data = os.path.join(d, 'weather_station_data_processing/corrected_data')

    # error_json = os.path.join(d, 'eddy_covariance_nldas_analysis', 'error_distributions.json')
    error_json = os.path.join(d, 'weather_station_data_processing', 'error_analysis', 'error_distributions.json')

    ee_check = os.path.join(d, 'weather_station_data_processing/NLDAS_data_at_stations')
    error_distribution(sta, sta_data, error_json, station_type='agri', check_dir=None)
# ========================= EOF ====================================================================
