import os
from pprint import pprint
from collections import OrderedDict

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

MT_IWR_STATION_CODES = ['0110', '0364', '0392', '0412', '0636', '0739', '0755', '0770', '0780', '0802',
                        '0819', '0877', '1008', '1044', '1088', '1169', '1231', '1297', '1318', '1518',
                        '1552', '1692', '1722', '1938', '1974', '1984', '1995', '2104', '2221', '2275',
                        '2347', '2409', '2820', '2827', '3013', '3110', '3157', '3558', '3581', '3939',
                        '4084', '4358', '4506', '5043', '5572', '5596', '5668', '5690', '5761', '6472',
                        '7263', '7318', '7382', '8324']

MT_IWR_STATION_NAMES = ['Dillon',
                        'Wisdom',
                        'Jackson',
                        'Lakeview',
                        'Lima',
                        'Busby',
                        'Hardin',
                        'Hysham 25',
                        'Wyola',
                        'Yellowtail Dam',
                        'Chinook',
                        'Harlem',
                        'Townsend',
                        'Trident',
                        'Joliet',
                        'Red Lodge',
                        'Ekalaka',
                        'Ridgeway',
                        'Cascade 20',
                        'Cascade 5',
                        'Great Falls',
                        'Neihart',
                        'Sun River',
                        'Big Sandy',
                        'Fort Benton',
                        'Geraldine',
                        'Iliad',
                        'Loma',
                        'Shonkin',
                        'Miles City',
                        'Mizpah',
                        'Powderville',
                        'Glendive',
                        'Plevna',
                        'Denton',
                        'Lewistown',
                        'Roy',
                        'Winifred',
                        'Creston',
                        'Hungry Horse Dam',
                        'Kalispell',
                        'Olney',
                        'Polebridge',
                        'West Glacier',
                        'Whitefish',
                        'Bozeman Exp Farm',
                        'Bozeman MT State',
                        'Hebgen Dam',
                        'Cohagen',
                        'Jordan',
                        'Mosby',
                        'Babb',
                        'Cut Bank',
                        'Del Bonita',
                        'East Glacier',
                        'St Mary',
                        'Ryegate',
                        'Philipsburg Ranger Station',
                        'Fort Assinniboine',
                        'Guilford',
                        'Havre',
                        'Simpson',
                        'Boulder',
                        'Moccasin Exp Station',
                        'Raynesford',
                        'Stanford',
                        'Bigfork',
                        'Polson',
                        'Polson Kerr Dam',
                        'St Ignatius',
                        'Augusta',
                        'Austin',
                        'Helena',
                        'Holter Dam',
                        'Lincoln Ranger Station',
                        'Chester',
                        'Joplin',
                        'Tiber Dam',
                        'Eureka Ranger Station',
                        'Fortine',
                        'Libby Ranger Station',
                        'Libby',
                        'Troy',
                        'Alder',
                        'Ennis',
                        'Glen',
                        'Norris',
                        'Twin Bridges',
                        'Virginia City',
                        'Brockway',
                        'Circle',
                        'Fort Peck Power Plant',
                        'Vida',
                        'Lennep',
                        'Martinsdale',
                        'White Sulpher Spr',
                        'St Regis Ranger Stn',
                        'Superior',
                        'Lindbergh Lake',
                        'Missoula',
                        'Missoula WSO AP',
                        'Potomac',
                        'Seeley Lake Ranger Station',
                        'Melstone',
                        'Roundup',
                        'Cooke City',
                        'Gardiner',
                        'Livingston',
                        'Livingston FAA AP',
                        'Wilsall',
                        'Flatwillow',
                        'Content',
                        'Malta 35',
                        'Malta 7',
                        'Port of Morgan',
                        'Saco',
                        'Zortman',
                        'Conrad',
                        'Valier',
                        'Biddle',
                        'Broadus',
                        'Moorhead',
                        'Sonnette',
                        'Deer Lodge',
                        'Ovando',
                        'Mildred',
                        'Terry',
                        'Terry 21',
                        'Darby',
                        'Hamilton',
                        'Stevensville',
                        'Sula',
                        'Western Ag Research',
                        'Savage',
                        'Sidney',
                        'Bredette',
                        'Culbertson',
                        'Wolf Point',
                        'Birney',
                        'Brandenberg',
                        'Colstrip',
                        'Forsythe',
                        'Ingomar',
                        'Rock Springs',
                        'Heron',
                        'Thompson Falls Power',
                        'Trout Cr Ranger Station',
                        'Medicine Lake',
                        'Plentywood',
                        'Raymond Border Station',
                        'Redstone',
                        'Westby',
                        'Butte FAA AP',
                        'Divide',
                        'Columbus',
                        'Mystic Lake',
                        'Nye',
                        'Rapelje',
                        'Big Timber',
                        'Melville',
                        'Blackleaf',
                        'Choteau Airport',
                        'Fairfield',
                        'Gibson Dam',
                        'Goldbutte',
                        'Sunburst',
                        'Sweetgrass',
                        'Hysham',
                        'Glasgow WSO AP',
                        'Hinsdale',
                        'Opheim 10',
                        'Opheim 16',
                        'Harlowton',
                        'Judith Gap',
                        'Carlyle',
                        'Wibaux']


def write_ghcn_iwr_station_shapefile(stations_file, out_shp, csv_dir):
    iwr_stations = [x.upper() for x in MT_IWR_STATION_NAMES]
    missing = {}
    lines = []
    with open(stations_file, 'r') as fp:
        for l in fp.readlines():
            splt = l.split()
            if splt[-1] == 'HCN' and splt[4] == 'MT':
                values = splt[:5] + [' '.join(splt[5:-1])]
            elif splt[0][-4:] in MT_IWR_STATION_CODES and splt[4] == 'MT':
                values = splt[:5] + [' '.join(splt[5:])]
            elif splt[4] == 'MT':
                if splt[5] in iwr_stations:
                    if splt[5] not in missing.keys():
                        missing[splt[5]] = [splt[0]]
                    else:
                        missing[splt[5]].append(splt[0])
                continue
            else:
                continue

            print(values)
            lines.append(values)

    found = {}
    if len(missing.keys()) > 0:
        for name, stations in missing.items():
            length = 0
            for staid in stations:
                _file = os.path.join(csv_dir, '{}.csv'.format(staid))
                df = pd.read_csv(_file)
                if df.shape[0] > length:
                    found[name] = staid

    found_id = [v for k, v in found.items()]
    with open(stations_file, 'r') as fp:
        for l in fp.readlines():
            splt = l.split()
            if splt[0] in found_id:
                values = splt[:5] + [' '.join(splt[5:])]
                lines.append(values)

    gdf = gpd.GeoDataFrame(data=lines, columns=['STAID', 'LAT', 'LON', 'ELEV', 'STATE', 'NAME'])
    geo = [Point(float(r['LON']), float(r['LAT'])) for i, r in gdf.iterrows()]
    gdf.geometry = geo
    gdf = gdf.set_crs('epsg:4326')
    gdf.to_file(out_shp)


if __name__ == '__main__':
    d = '/media/research/IrrigationGIS'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS'

    uy = os.path.join(d, 'Montana', 'upper_yellowstone', 'gsflow_prep')
    clim = os.path.join(d, 'climate')
    stations_ = os.path.join(clim, 'stations')
    gages_ = os.path.join(clim, 'gages')

    ghcn_data_dir = os.path.join(d, 'climate', 'ghcn', 'ghcn_daily_summaries_4FEB2022')
    _txt = os.path.join(stations_, 'ghcnd-stations.txt')
    ghcn_shp_ = os.path.join(stations_, 'mt_arm_iwr_stations.shp')
    write_ghcn_iwr_station_shapefile(_txt, ghcn_shp_, ghcn_data_dir)

# ========================= EOF ====================================================================
