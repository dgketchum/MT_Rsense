# Adapted from Olivier Hagolle and DevelopmentSeed
# https://github.com/olivierhagolle/LANDSAT-Download
import os
import urllib
import urllib2
import time
import re
import io
import sys
import csv
import math
import subprocess
from datetime import datetime, timedelta

from pkgutil import get_data
from landsat.search import Search
import web_tools


def connect_earthexplorer_proxy(proxy_info, usgs):
    print "Establishing connection to Earthexplorer with proxy..."
    # contruction d'un "opener" qui utilise une connexion proxy avec autorisation
    cookies = urllib2.HTTPCookieProcessor()
    proxy_support = urllib2.ProxyHandler({"http": "http://%(user)s:%(pass)s@%(host)s:%(port)s" % proxy_info,
                                          "https": "http://%(user)s:%(pass)s@%(host)s:%(port)s" % proxy_info})
    opener = urllib2.build_opener(proxy_support, cookies)

    # installation
    urllib2.install_opener(opener)
    # deal with csrftoken required by USGS as of 7-20-2016
    data = urllib2.urlopen("https://ers.cr.usgs.gov").read()
    m = re.search(r'<input .*?name="csrf_token".*?value="(.*?)"', data)
    if m:
        token = m.group(1)
    else:
        print "Error : CSRF_Token not found"
        # sys.exit(-3)
    # parametres de connection
    params = urllib.urlencode(dict(username=usgs['account'], password=usgs['passwd'], csrf_token=token))
    # utilisation

    request = urllib2.Request("https://ers.cr.usgs.gov", params, headers={})
    f = urllib2.urlopen(request)
    data = f.read()
    f.close()

    if data.find('You must sign in as a registered user to download data or place orders for USGS EROS products') > 0:
        print "Authentification failed"
        # sys.exit(-1)
    return


def connect_earthexplorer_no_proxy(usgs):
    # mkmitchel (https://github.com/mkmitchell) solved the token issue
    cookies = urllib2.HTTPCookieProcessor()
    opener = urllib2.build_opener(cookies)
    urllib2.install_opener(opener)

    data = urllib2.urlopen("https://ers.cr.usgs.gov").read()
    m = re.search(r'<input .*?name="csrf_token".*?value="(.*?)"', data)
    if m:
        token = m.group(1)
    else:
        print "Error : CSRF_Token not found"
        # sys.exit(-3)

    params = urllib.urlencode(dict(username=usgs['account'], password=usgs['passwd'], csrf_token=token))
    request = urllib2.Request("https://ers.cr.usgs.gov/login", params, headers={})
    f = urllib2.urlopen(request)

    data = f.read()
    f.close()
    if data.find('You must sign in as a registered user to download data or place orders for USGS EROS products') > 0:
        print "Authentification failed"
        # sys.exit(-1)
    return


def download_chunks(url, rep, nom_fic):
    """ Downloads large files in pieces
   inspired by http://josh.gourneau.com
  """
    try:
        req = urllib2.urlopen(url)
        # if downloaded file is html
        if req.info().gettype() == 'text/html':
            print "error : file is in html and not an expected binary file"
            lines = req.read()
            if lines.find('Download Not Found') > 0:
                raise TypeError
            else:
                with open("error_output.html", "w") as f:
                    f.write(lines)
                    print "result saved in ./error_output.html"
                    # sys.exit(-1)
        # if file too small
        total_size = int(req.info().getheader('Content-Length').strip())
        if (total_size < 50000):
            print "Error: The file is too small to be a Landsat Image"
            print url
            # sys.exit(-1)
        print nom_fic, total_size
        total_size_fmt = sizeof_fmt(total_size)

        # download
        downloaded = 0
        CHUNK = 1024 * 1024 * 8
        with open(rep + '/' + nom_fic, 'wb') as fp:
            start = time.clock()
            print('Downloading {0} ({1}):'.format(nom_fic, total_size_fmt))
            while True:
                chunk = req.read(CHUNK)
                downloaded += len(chunk)
                done = int(50 * downloaded / total_size)
                sys.stdout.write('\r[{1}{2}]{0:3.0f}% {3}ps'
                                 .format(math.floor((float(downloaded)
                                                     / total_size) * 100),
                                         '=' * done,
                                         ' ' * (50 - done),
                                         sizeof_fmt((downloaded // (time.clock() - start)) / 8)))
                sys.stdout.flush()
                if not chunk: break
                fp.write(chunk)
    except urllib2.HTTPError, e:
        if e.code == 500:
            pass  # File doesn't exist
        else:
            print "HTTP Error:", e.code, url
        return False
    except urllib2.URLError, e:
        print "URL Error:", e.reason, url
        return False

    return rep, nom_fic


def cycle_day(path):
    """ provides the day in cycle given the path number
    """
    cycle_day_path1 = 5
    cycle_day_increment = 7
    nb_days_after_day1 = cycle_day_path1 + cycle_day_increment * (path - 1)

    cycle_day_path = math.fmod(nb_days_after_day1, 16)
    if path >= 98:  # change date line
        cycle_day_path += 1
    print cycle_day_path
    return (cycle_day_path)


def next_overpass(date1, path, sat):
    """ provides the next overpass for path after date1
    """
    date0_L5 = datetime(1985, 5, 4)
    date0_L7 = datetime(1999, 1, 11)
    date0_L8 = datetime(2013, 5, 1)
    print 'date1: {}, type: {}'.format(date1, type(date1))
    print 'dateL5: {}, type: {}'.format(date0_L5, type(date0_L5))

    next_day = math.fmod((date1 - date0_L5).days - cycle_day(path) + 1, 16)
    if next_day != 0:
        date_overpass = date1 + timedelta(16 - next_day)
    else:
        date_overpass = date1
    return (date_overpass)


def sizeof_fmt(num):
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num < 1024.0:
            return "%3.1f %s" % (num, x)
        num /= 1024.0


def unzipimage(tgzfile, outputdir):
    success = 0
    if os.path.exists(outputdir + '/' + tgzfile + '.tgz'):
        print "\nunzipping..."
        try:
            if sys.platform.startswith('linux'):
                subprocess.call('mkdir ' + outputdir + '/' + tgzfile, shell=True)  # Unix
                subprocess.call('tar zxvf ' + outputdir + '/' + tgzfile + '.tgz -C ' + outputdir + '/' + tgzfile,
                                shell=True)  # Unix
            elif sys.platform.startswith('win'):
                subprocess.call('tartool ' + outputdir + '/' + tgzfile + '.tgz ' + outputdir + '/' + tgzfile,
                                shell=True)  # W32
            success = 1
        except TypeError:
            print 'Failed to unzip %s' % tgzfile
        os.remove(outputdir + '/' + tgzfile + '.tgz')
    return success


def get_credentials(usgs_path):
    print 'USGS txt path: {}'.format(usgs_path)
    with file(usgs_path) as f:
        (account, passwd) = f.readline().split(' ')
        if passwd.endswith('\n'):
            passwd = passwd[:-1]
        usgs = {'account': account, 'passwd': passwd}
        return usgs


def get_station_list_identifier(product):
    if product.startswith('LC8'):
        identifier = '4923'
        stations = ['LGN']
    elif product.startswith('LE7'):
        identifier = '3373'
        stations = ['EDC', 'SGS', 'AGS', 'ASN', 'SG1', 'CUB', 'COA']
    elif product.startswith('LT5'):
        identifier = '3119'
        stations = ['GLC', 'ASA', 'KIR', 'MOR', 'KHC', 'PAC',
                    'KIS', 'CHM', 'LGS', 'MGR', 'COA', 'MPS', 'CUB']
    else:
        raise NotImplementedError('Must provide valid product string...')

    return identifier, stations


def assemble_scene_id_list(ref_time, prow, grnd_stn, end_date, sat, delta=16):

    print 'assemble scene list'

    scene_id_list = []
    archive_found = False

    while ref_time < end_date:

        date_part = datetime.strftime(ref_time, '%Y%j')
        padded_pr = '{}{}'.format(str(prow[0]).zfill(3), str(prow[1]).zfill(3))

        if not archive_found:  # iterate through versions, holding latest
            for archive in ['00', '01', '02']:
                scene_str = '{}{}{}{}{}'.format(sat, padded_pr, date_part, grnd_stn, archive)
                if web_tools.verify_landsat_scene_exists(scene_str):
                    version = archive
                    archive_found = True
                    print 'using version: {}'.format(version)

        else:
            scene_str = '{}{}{}{}{}'.format(sat, padded_pr, date_part, grnd_stn, version)

            print 'add scene: {}, for {}'.format(scene_str,
                                                 datetime.strftime(ref_time, '%Y-%m-%d'))
            scene_id_list.append(scene_str)

            ref_time += timedelta(days=delta)

    return scene_id_list


def get_candidate_scenes_list(path_row, satellite, start_date, end_date=None,
                              max_cloud_cover=70, limit_scenes=100):
    """
    
    :param path_row: path, datetime obj
    :param satellite: 'LT5', 'LE7', or 'LC8'
    :param start_date: datetime object start image search
    :param end_date: datetime object finish image search
    :param max_cloud_cover: percent cloud cover according to USGS image metadata, float
    :param limit_scenes: max number scenese, int
    :return: reference overpass = str('YYYYDOY'), station str('XXX') len=3
    """
    print 'sat: {}'.format(satellite)
    if satellite == 'LT5':
        reference_overpass, station = web_tools.get_l5_overpass_data(path_row, start_date)
        print 'station: {}, ref time: {}'.format(station, reference_overpass)
        scene_list = assemble_scene_id_list(reference_overpass, path_row,
                                            station, end_date, satellite)
        print scene_list

        TODO: use
        web_tools.get_l5_overpass_data()
        to
        find
        station
        name

    elif satellite in ['LE7', 'LC8']:
        reference_overpass, station = web_tools.landsat_overpass_data(path_row,
                                                                      start_date, satellite)
        print 'station: {}, ref time: {}'.format(station, reference_overpass)

        scene_list = assemble_scene_id_list(reference_overpass, path_row,
                                            station, end_date, satellite)
        print scene_list

    else:
        raise NotImplementedError('Must choose a valid satellite')


def down_usgs_by_list(scene_list, output_dir, usgs_creds_txt):
    usgs_creds = get_credentials(usgs_creds_txt)
    connect_earthexplorer_no_proxy(usgs_creds)

    for product in scene_list:
        identifier, stations = get_station_list_identifier(product)
        base_url = 'https://earthexplorer.usgs.gov/download/'
        tail_string = '{}/{}/STANDARD/EE'.format(identifier, product)
        url = '{}{}'.format(base_url, tail_string)

        tgz_file = '{}.tgz'.format(product)
        download_chunks(url, output_dir, tgz_file)
        unzipimage(tgz_file, output_dir)

    return None


if __name__ == '__main__':
    home = os.path.expanduser('~')
    start = datetime(2014, 5, 1)
    end = datetime(2014, 5, 30)
    satellite = 'LC8'
    output = os.path.join(home, 'images', satellite)
    usgs_creds = os.path.join(home, 'images', 'usgs.txt')
    path_row = 37, 27
    print get_candidate_scenes_list(path_row, satellite, start, end)

# ===============================================================================
