import json
import os
import warnings
from datetime import timedelta
from tqdm import tqdm

import numpy as np
import pandas as pd
import pynldas2 as nld
import pytz
from refet import Daily, calcs
from pandarallel import pandarallel
from scipy.stats import skew, kurtosis, norm

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

PACIFIC = pytz.timezone('US/Pacific')


def error_distribution(stations, station_data, results, station_type='ec', check_dir=None):
    kw = station_par_map(station_type)
    station_list = pd.read_csv(stations, index_col=kw['index'])

    errors = {}
    for index, row in tqdm(station_list.iterrows(), desc='Processing stations'):

        # if index != 'crsm':
        #     continue

        print(index)

        try:
            sdf_file = os.path.join(station_data, '{}_output.xlsx'.format(index))
            sdf = pd.read_excel(sdf_file, parse_dates=True, index_col='date')

            s = pd.to_datetime(sdf.index[0]) - timedelta(days=2)
            e = pd.to_datetime(sdf.index[-1]) + timedelta(days=2)

            sdf.index = sdf.index.tz_localize(PACIFIC)
            sdf = sdf.rename(RENAME_MAP, axis=1)

            nldas = get_nldas(row[kw['lon']], row[kw['lat']], row[kw['elev']], start=s, end=e)

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
                cdf.index = cdf.index.tz_localize(PACIFIC)
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


def error_propagation(json_file, station_meta, station_data, outfile, station_type='ec', num_samples=1000):
    pandarallel.initialize()

    kw = station_par_map(station_type)

    with open(json_file, 'r') as f:
        error_distributions = json.load(f)

    results = {}

    station_list = pd.read_csv(station_meta, index_col=kw['index'])

    for station, row in station_list.iterrows():

        errors = error_distributions[station]
        if errors == 'exception':
            print('Skipping station {} due to previous exception.'.format(station))
            continue

        print('Processing station: {}'.format(station))

        sdf_file = os.path.join(station_data, '{}_output.xlsx'.format(station))
        sdf = pd.read_excel(sdf_file, parse_dates=True, index_col='date')

        s = pd.to_datetime(sdf.index[0]) - timedelta(days=2)
        e = pd.to_datetime(sdf.index[-1]) + timedelta(days=2)

        pacific = pytz.timezone('US/Pacific')
        sdf.index = sdf.index.tz_localize(pacific)
        sdf = sdf.rename(RENAME_MAP, axis=1)

        nldas = get_nldas(row[kw['lon']], row[kw['lat']], row[kw['elev']], start=s, end=e)

        station_results = {var: [] for var in COMPARISON_VARS}

        def calc_eto(r):
            return Daily(
                tmin=r['min_temp'],
                tmax=r['max_temp'],
                ea=r['ea'],
                rs=r['rsds'] * 0.0036,
                uz=r['wind'],
                zw=10.0 if station_type == 'ec' else row['anemom_height_m'],
                doy=r['doy'],
                elev=row[kw['elev']],
                lat=row[kw['lat']]
            ).eto()[0]

        for var in COMPARISON_VARS:

            if var == 'eto':
                eto = calc_eto(nldas)
                station_results[var] = np.array(eto).mean().item(), np.array(eto).std().item()
                continue

            if var not in errors:
                print('Error data for variable {} not found in station {}, skipping.'.format(var, station))
                continue

            result = []
            mean_, variance, data_skewness, data_kurtosis, n = errors[var]
            stddev = np.sqrt(variance)
            sampled_errors = norm.rvs(loc=mean_, scale=stddev, size=num_samples)

            for error in sampled_errors:
                perturbed_nldas = nldas.copy()
                perturbed_nldas[var] += error
                eto_values = perturbed_nldas.parallel_apply(calc_eto, axis=1)
                result.append(eto_values.mean())

            station_results[var] = np.array(result).mean().item(), np.array(result).std().item()

        results[station] = station_results

        with open(outfile, 'w') as f:
            json.dump(results, f, indent=4)


def station_par_map(station_type):
    if station_type == 'ec':
        return {'index': 'SITE_ID',
                'lat': 'LATITUDE',
                'lon': 'LONGITUDE',
                'elev': 'ELEVATION (METERS)',
                'start': 'START DATE',
                'end': 'END DATE'}
    elif station_type == 'agri':
        return {'index': 'id',
                'lat': 'latitude',
                'lon': 'longitude',
                'elev': 'elev_m',
                'start': 'record_start',
                'end': 'record_end'}
    else:
        raise NotImplementedError


def get_nldas(lon, lat, elev, start, end):
    nldas = nld.get_bycoords((lon, lat), start_date=start, end_date=end,
                             variables=['temp', 'wind_u', 'wind_v', 'humidity', 'rsds'])

    nldas = nldas.tz_convert(PACIFIC)

    wind_u = nldas['wind_u']
    wind_v = nldas['wind_v']
    nldas['wind'] = np.sqrt(wind_v ** 2 + wind_u ** 2)

    nldas['min_temp'] = nldas['temp'] - 273.15
    nldas['max_temp'] = nldas['temp'] - 273.15
    nldas['doy'] = [i.dayofyear for i in nldas.index]

    nldas = nldas.resample('D').agg(RESAMPLE_MAP)
    nldas['ea'] = calcs._actual_vapor_pressure(pair=calcs._air_pressure(elev),
                                               q=nldas['humidity'])

    return nldas


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

    results_json = os.path.join(d, 'weather_station_data_processing', 'error_analysis', 'error_propagation.json')
    # error_propagation(error_json, sta, sta_data, results_json, station_type='agri', num_samples=10)
# ========================= EOF ====================================================================
