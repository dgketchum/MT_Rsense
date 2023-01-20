import os
from subprocess import check_call, CalledProcessError
import fiona
from shapely.geometry import shape

home = os.path.expanduser('~')
conda = os.path.join(home, 'miniconda3', 'envs')
if not os.path.exists(conda):
    conda = conda.replace('miniconda3', 'miniconda')
EE = os.path.join(conda, 'metric', 'bin', 'earthengine')
GS = os.path.join(conda, 'metric', 'bin', 'gsutil')

OGR = '/usr/bin/ogr2ogr'

AEA = '+proj=aea +lat_0=40 +lon_0=-96 +lat_1=20 +lat_2=60 +x_0=0 +y_0=0 +ellps=GRS80 ' \
      '+towgs84=0,0,0,0,0,0,0 +units=m +no_defs'
WGS = '+proj=longlat +datum=WGS84 +no_defs'

os.environ['GDAL_DATA'] = 'miniconda3/envs/gcs/share/gdal/'


def to_equal_area(in_dir, out_dir):
    in_shp = [os.path.join(in_dir, x) for x in os.listdir(in_dir) if x.endswith('.shp')]
    for s in in_shp:
        out_shp = os.path.join(out_dir, os.path.basename(s))
        cmd = [OGR, '-f', 'ESRI Shapefile', '-s_srs', WGS, '-t_srs', AEA, out_shp, s]
        try:
            check_call(cmd)
            print(out_shp)
        except CalledProcessError:
            print('error', out_shp)


def calc_areas(_dir):
    shps = [os.path.join(_dir, x) for x in os.listdir(_dir) if x.endswith('.shp')]
    dp, intern, flu = 0, 0, 0
    for s in shps:
        area = 0
        with fiona.open(s, 'r') as src:
            error_ct = 0
            for f in src:
                try:
                    p = shape(f['geometry'])
                    area += p.area / 4046.86 * 1e-3
                except AttributeError:
                    error_ct += 1
                    pass
            if error_ct > 0:
                pass
        print('\n{} error count: {}\n'.format(os.path.basename(s), error_ct))
        if 'dp' in os.path.basename(s):
            dp += area
        if 'flu' in os.path.basename(s):
            flu += area
        else:
            intern += area
        print('{:.3f} {}'.format(area, os.path.basename(s)))
    print('\n\n\n\n{:.3f} intern\n{:.3f} dp\n{:.3f} flu'.format(intern, dp, flu))


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/Montana/geointernship/progress'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/Montana/geointernship/progress'
    aea_ = os.path.join(root, 'aea')
    _wgs = os.path.join(root, 'wgs')
    to_equal_area(_wgs, aea_)
    calc_areas(aea_)
# ========================= EOF ====================================================================
