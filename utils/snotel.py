import os
import numpy as np
from pandas import read_csv
from geopandas import GeoDataFrame
from shapely.geometry import Point
import requests
from io import StringIO

ABV = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
    "District of Columbia": "DC",
    "American Samoa": "AS",
    "Guam": "GU",
    "Northern Mariana Islands": "MP",
    "Puerto Rico": "PR",
    "United States Minor Outlying Islands": "UM",
    "U.S. Virgin Islands": "VI",
}


def snotel_average_max_swe(csv):
    df = read_csv(csv)
    df = df[df['Network'] == 'SNOTEL']
    df['ID'] = df['ID'].apply(lambda x: x.strip())
    df['mean_swe'] = [None for x in range(df.shape[0])]
    ct = 0
    for i, r in df.iterrows():
        try:
            st = ABV[r['State']].lower()
            url = 'https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/customGroupByMonthReport/' \
                  'daily/{}:{}:SNTL%7Cid=%22%22%7Cname/POR_BEGIN,POR_END/WTEQ::value'.format(r['ID'], st)
            resp = requests.get(url).text.splitlines()
            if len(resp) < 60:
                continue
            lines = '\n'.join([resp[58]] + resp[60:])
            sttrr = StringIO(lines)
            sdf = read_csv(sttrr)
            sdf = sdf.rename(columns={'Water Year': 'wy', 'Day': 'd'})
            s, e = sdf['wy'].loc[0], 2021
            mean_ = np.nanmean([sdf[(sdf['wy'] == y) & (sdf['d'] == 30)]['Apr'].values[0] for y in range(s, e)])
            df.loc[i, 'mean_swe'] = mean_
            print('{:.2f}'.format(mean_), r['Name'])
            ct += 1
        except Exception as e:
            print(r['Name'], r['ID'], e)
            pass

    print('{} of {} sites'.format(ct, df.shape[0]))
    _file = csv.replace('_list', '_1APR_swe')
    # df.to_csv(_file)
    geometry = [Point(x, y) for x, y in zip(df['Longitude'], df['Latitude'])]
    gdf = GeoDataFrame(df, geometry=geometry)
    _file = _file.replace('.csv', '.shp')
    gdf.to_file(_file)


if __name__ == '__main__':
    c = '/home/user/Downloads/snotel_list.csv'
    snotel_average_max_swe(c)
# ========================= EOF ====================================================================
