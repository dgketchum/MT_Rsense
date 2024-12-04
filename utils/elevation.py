import requests
import urllib
import geopandas as gpd

url = r'https://nationalmap.gov/epqs/pqs.php?'

from thredds import GridMet


def elevation_from_coordinate(lat, lon):
    params = {
        'output': 'json',
        'x': lon,
        'y': lat,
        'units': 'Meters'
    }

    result = requests.get((url + urllib.parse.urlencode(params)))
    elev = result.json()['USGS_Elevation_Point_Query_Service']['Elevation_Query']['Elevation']
    return elev


def gridmet_elevation(lat, lon):
    g = GridMet('elev', lat=lat, lon=lon)
    elev = g.get_point_elevation()
    return elev


if __name__ == '__main__':
    in_ = '/media/research/IrrigationGIS/swim/gridmet/gridmet_centroids_tongue.shp'
    out_ = '/media/research/IrrigationGIS/swim/gridmet/gridmet_centroids_tongue_elev.shp'
    df = gpd.read_file(in_)
    l = []
    for i, r in df.iterrows():
        elev = gridmet_elevation(r['lat'], r['lon'])
        l.append(i, elev)
    pass
# ========================= EOF ====================================================================
