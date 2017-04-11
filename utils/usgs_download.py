# Adapted from Olivier Hagolle and DevelopmentSeed
# https://github.com/olivierhagolle/LANDSAT-Download
import os
import urllib
import urllib2
import time
import re
import sys
import math
import subprocess

from landsat.search import Search


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
    return (cycle_day_path)


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


def get_candidate_scenes_list(path_row, start_date, end_date, satellite='L8',
                              max_cloud_cover=70, limit_scenes=100):
    path, row = path_row[0], path_row[1]

    if satellite == 'L8':
        searcher = Search()
        candidate_scenes = searcher.search(paths_rows='{},{},{},{}'.format(path, row, path, row),
                                           start_date=start_date,
                                           end_date=end_date,
                                           cloud_min=0,
                                           cloud_max=max_cloud_cover,
                                           limit=limit_scenes)
        results = candidate_scenes['results']
        scene_list = []

        for item in results:
            print item['sceneID']
            scene_list.append(item['sceneID'])

        return scene_list

    else:
        pass


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
    pass

# ===============================================================================
