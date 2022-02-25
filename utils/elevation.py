import requests
import urllib

url = r'https://nationalmap.gov/epqs/pqs.php?'


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


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
