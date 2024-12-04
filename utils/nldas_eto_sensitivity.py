import json
import os
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
import pynldas2 as nld
import pytz
from pandarallel import pandarallel
from refet import Daily, calcs
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

COMPARISON_VARS = ['rn', 'vpd', 'tmean', 'u2', 'eto']

PACIFIC = pytz.timezone('US/Pacific')


def error_distribution(stations, station_data, results, out_data, station_type='ec', check_dir=None):
    kw = station_par_map(station_type)
    station_list = pd.read_csv(stations, index_col=kw['index'])

    errors = {}
    for i, (index, row) in enumerate(station_list.iterrows()):

        print('{} of {}: {}'.format(i + 1, len(station_list.keys()), index))
        try:
            sdf_file = os.path.join(station_data, '{}_output.xlsx'.format(index))
            sdf = pd.read_excel(sdf_file, parse_dates=True, index_col='date')

            s = pd.to_datetime(sdf.index[0]) - timedelta(days=2)
            e = pd.to_datetime(sdf.index[-1]) + timedelta(days=2)

            sdf.index = sdf.index.tz_localize(PACIFIC)
            sdf = sdf.rename(RENAME_MAP, axis=1)
            sdf['doy'] = [i.dayofyear for i in sdf.index]

            _zw = 10.0 if station_type == 'ec' else row['anemom_height_m']

            def calc_asce_params(r, zw):
                asce = Daily(tmin=r['min_temp'],
                             tmax=r['max_temp'],
                             ea=r['ea'],
                             rs=r['rsds'] * 0.0036,
                             uz=r['wind'],
                             zw=zw,
                             doy=r['doy'],
                             elev=row[kw['elev']],
                             lat=row[kw['lat']])

                vpd = asce.vpd[0]
                rn = asce.rn[0]
                u2 = asce.u2[0]
                mean_temp = asce.tmean[0]
                eto = asce.eto()[0]

                return vpd, rn, u2, mean_temp, eto

            asce_params = sdf.parallel_apply(calc_asce_params, zw=_zw, axis=1)
            sdf[['vpd', 'rn', 'u2', 'tmean', 'eto']] = pd.DataFrame(asce_params.tolist(), index=sdf.index)

            nldas = get_nldas(row[kw['lon']], row[kw['lat']], row[kw['elev']], start=s, end=e)
            asce_params = nldas.parallel_apply(calc_asce_params, zw=_zw, axis=1)
            nldas[['vpd', 'rn', 'u2', 'tmean', 'eto']] = pd.DataFrame(asce_params.tolist(), index=nldas.index)

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
                dct[var] = (mean_, variance, data_skewness, data_kurtosis,
                            [i.strftime('%Y-%m-%d') for i in residuals.index])

            dct['file'] = os.path.join(out_data, '{}.csv'.format(index))
            nldas = nldas.loc[sdf.index]
            nldas['obs_eto'] = sdf['eto']
            nldas.to_csv(dct['file'])
            errors[index] = dct.copy()

        except Exception as e:
            print('Exception at {}: {}'.format(index, e))
            errors[index] = 'exception'

    with open(results, 'w') as dst:
        json.dump(errors, dst, indent=4)


def error_propagation(json_file, station_meta, outfile, station_type='ec', num_samples=1000):

    kw = station_par_map(station_type)

    with open(json_file, 'r') as f:
        error_distributions = json.load(f)

    results = {}

    station_list = pd.read_csv(station_meta, index_col=kw['index'])

    first, out_vars = True, []

    for j, (station, row) in enumerate(station_list.iterrows()):

        errors = error_distributions[station]
        if errors == 'exception':
            print('Skipping station {} due to previous exception.'.format(station))
            continue

        print('{} of {}: {}'.format(j + 1, len(station_list.keys()), station))

        file_ = errors.pop('file')
        if not os.path.exists(file_):
            file_ = file_.replace('/media/research', '/home/dgketchum/data')
        nldas = pd.read_csv(file_, parse_dates=True, index_col='date')
        nldas.index = pd.DatetimeIndex([i.strftime('%Y-%m-%d') for i in nldas.index])
        station_results = {var: [] for var in COMPARISON_VARS}

        def calc_eto(r, mod_var, mod_vals):
            # modify the error-perturbed values with setattr
            asce = Daily(
                tmin=r['min_temp'],
                tmax=r['max_temp'],
                ea=r['ea'],
                rs=r['rsds'] * 0.0036,
                uz=r['u2'],
                zw=2.0,
                doy=r['doy'],
                elev=row[kw['elev']],
                lat=row[kw['lat']])

            setattr(asce, mod_var, mod_vals)

            return asce.eto()[0]

        for var in COMPARISON_VARS:

            if first:
                out_vars.append(var)

            result = []
            mean_, variance, data_skewness, data_kurtosis, dates = errors[var]
            dates = pd.DatetimeIndex(dates)
            stddev = np.sqrt(variance)

            if var == 'eto':
                eto_arr = nldas.loc[dates, var].values
                station_results[var] = np.mean(eto_arr), np.std(eto_arr)
                continue

            for i in range(num_samples):
                perturbed_nldas = nldas.loc[dates].copy()
                error = norm.rvs(loc=mean_, scale=stddev, size=perturbed_nldas.shape[0])
                perturbed_nldas[var] += error
                eto_values = perturbed_nldas.parallel_apply(calc_eto, mod_var=var,
                                                            mod_vals=perturbed_nldas[var].values,
                                                            axis=1)
                result.append(list(eto_values))

            station_results[var] = np.array(result).mean().item(), np.array(result).std().item()

        results[station] = station_results

        first = False

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
    if not os.path.isdir(d):
        d = '/home/dgketchum/data/IrrigationGIS/milk'
    # sta = os.path.join(d, '/eddy_covariance_data_processing/eddy_covariance_stations.csv'
    sta = os.path.join(d, 'bias_ratio_data_processing/ETo/'
                          'final_milk_river_metadata_nldas_eto_bias_ratios_long_term_mean.csv')
    sta_data = os.path.join(d, 'weather_station_data_processing/corrected_data')
    comp_data = os.path.join(d, 'weather_station_data_processing/comparison_data')

    # error_json = os.path.join(d, 'eddy_covariance_nldas_analysis', 'error_distributions.json')
    error_json = os.path.join(d, 'weather_station_data_processing', 'error_analysis', 'error_distributions_50.json')

    pandarallel.initialize(nb_workers=5)

    ee_check = os.path.join(d, 'weather_station_data_processing/NLDAS_data_at_stations')
    error_distribution(sta, sta_data, error_json, comp_data, station_type='agri', check_dir=None)

    results_json = os.path.join(d, 'weather_station_data_processing', 'error_analysis', 'error_propagation_50.json')
    error_propagation(error_json, sta, results_json, station_type='agri', num_samples=50)
# ========================= EOF ====================================================================
