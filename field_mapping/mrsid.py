import json
import os
from glob import glob
import subprocess
from subprocess import Popen, PIPE, check_call, call
from pathlib import Path
from pyproj import datadir

proj_dir = datadir.get_data_dir()

root = '/media/research/IrrigationGIS'
if not os.path.exists(root):
    root = '/home/dgketchum/data/IrrigationGIS'

_bin = os.path.join(root, 'MrSID_DSDK-9.5.4.4709-rhel6.x86-64.gcc531/Raster_DSDK/bin/')
INFO = os.path.join(_bin, 'mrsidinfo')
DECODE = os.path.join(_bin, 'mrsiddecode')

conda = '/home/dgketchum/miniconda3/envs/opnt/bin/'
if not os.path.exists(conda):
    conda = '/home/dgketchum/miniconda3/envs/metric/bin/'

TRANSLATE = os.path.join(conda, 'gdal_translate')
OVR = os.path.join(conda, 'gdaladdo')
VRT = os.path.join(conda, 'gdalbuildvrt')
GINFO = os.path.join(conda, 'gdalinfo')
PROJ = '+proj=utm +zone=12 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs +type=crs'


def tile(a):
    s = 60000
    t = [a[x:x + s, y:y + s] for x in range(0, a.shape[0], s) for y in range(0, a.shape[1], s)]
    return t


def segment_and_convert(ortho_dir):
    files = []

    for path in Path(ortho_dir).rglob('*.sid'):
        files.append(path)

    for pos_path in files:
        path = str(pos_path)
        _root = os.path.dirname(path)
        print(path)
        p = Popen([INFO, '-i', str(path)], stdin=PIPE, stdout=PIPE, stderr=PIPE)
        out = p.communicate()
        s = out[0].decode('utf-8').splitlines()
        w = int([x for x in s if 'width' in x][0].split()[1])
        h = int([x for x in s if 'height' in x][0].split()[1])
        s = 60000
        t = [((x, y), (x + s, y + s)) for x in range(0, h, s) for y in range(0, w, s)]

        removal = []
        for elem, c in enumerate(t):
            tiff = os.path.join(pos_path.parent, '{}.tif'.format(elem))
            btiff = tiff.replace('.tif', '_.tif')
            try:
                if os.path.exists(tiff):
                    print(tiff, 'exists')
                    removal.append(tiff)

                elif os.path.exists(btiff):
                    print(btiff, 'exists')

                else:
                    cmd = [DECODE, '-i', str(path), '-o', tiff, '-of', 'tifg', '-wf',
                           '-ulxy', str(c[0][1]), str(c[0][0]),
                           '-lrxy', str(c[1][1]), str(c[1][0])]
                    print('writing', tiff)
                    check_call(cmd)
                    removal.append(tiff)

                if not os.path.exists(btiff):
                    cmd = [TRANSLATE, '-a_srs', PROJ, '-co', 'TILED=YES',
                           '-co', 'BIGTIFF=YES', tiff, btiff]
                    print('writing', btiff)
                    check_call(cmd)
                    print('writing overviews')
                    cmd = [OVR, '-r', 'average', btiff]
                    check_call(cmd)
                    
            except subprocess.CalledProcessError as e:
                print(e)
                continue

        l = os.path.join(_root, 'list_.txt')
        f = open(l, 'w')
        cmd = ['ls', '--'] + glob('{}/*_.tif'.format(_root))
        call(cmd, stdout=f)
        f.close()
        cmd = [VRT, os.path.join(_root, 'catalog.vrt'), '-input_file_list', l, '-allow_projection_difference']
        check_call(cmd)
        for f in removal:
            os.remove(f)


if __name__ == '__main__':
    ortho = '/media/research/IrrigationGIS/Montana/naip/mt_n'
    segment_and_convert(ortho)
# ========================= EOF ====================================================================
