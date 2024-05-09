from datetime import timedelta

import numpy as np
import pandas as pd
import pynldas2 as nld
import pytz
from refet import Daily


def sensitivity(stations, results):
    station_list = pd.read_csv(stations, index_col='SITE_ID')

    for index, row in station_list.iterrows():
        s = pd.to_datetime(row['START DATE']) - timedelta(days=1)
        e = pd.to_datetime(row['END DATE']) + timedelta(days=2)
        nldas = nld.get_bycoords((row['LONGITUDE'], row['LATITUDE']), start_date=s, end_date=e,
                                 variables=['temp', 'wind_u', 'wind_v', 'humidity', 'rsds'])

        central = pytz.timezone('US/Central')
        nldas = nldas.tz_convert(central)

        wind_u = nldas['wind_u']
        wind_v = nldas['wind_v']
        nldas['wind'] = np.sqrt(wind_v ** 2 + wind_u ** 2)

        nldas['min_temp'] = nldas['temp'] - 273.15
        nldas['max_temp'] = nldas['temp'] - 273.15
        nldas['doy'] = [i.dayofyear for i in nldas.index]
        nldas.drop(columns=['temp'], inplace=True)

        nldas = nldas.resample('D').agg({
            'rsds': 'sum',
            'humidity': 'mean',
            'min_temp': 'min',
            'max_temp': 'max',
            'wind': 'mean',
            'doy': 'first',

        })
        nldas['rsds'] *= 0.0036

        def calc_eto(r):
            eto = Daily(tmin=r['min_temp'],
                        tmax=r['max_temp'],
                        ea=r['humidity'],
                        rs=r['rsds'],
                        uz=r['wind'],
                        zw=10.0,
                        doy=r['doy'],
                        elev=row['ELEVATION (METERS)'],
                        lat=row['LATITUDE']).eto()[0]
            return eto

        nldas['eto'] = nldas.apply(calc_eto, axis=1)

        return None


if __name__ == '__main__':
    sta = '/media/research/IrrigationGIS/milk/eddy_covariance_data_processing/eddy_covariance_stations.csv'
    res = '/media/research/IrrigationGIS/milk/eddy_covariance_nldas_analysis'
    sensitivity(sta, res)
# ========================= EOF ====================================================================
