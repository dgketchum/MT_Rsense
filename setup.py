# ===============================================================================
# Copyright 2023 dgketchum
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===============================================================================

import os

from setuptools import setup, find_packages

tag = '0.0.01'

setup(name='MT_Rsense',
      version=tag,
      description='Montana DNRC Remote Sensing Tools',
      license='Apache',
      classifiers=[
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering :: GIS',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: Python :: 3'],
      keywords='landsat gridded meteorology hydrology remote sensing',
      author='David Ketchum',
      author_email='dgketchum@gmail.com',
      platforms='Posix; MacOS X; Windows',
      packages=find_packages(),
      download_url='https://github.com/{}/{}/archive/{}.tar.gz'.format('dgketchum', 'MT_Rsense', tag),
      url='https://github.com/dgketchum/MT_Rsense',
      install_requires=['numpy', 'geopy', 'pandas', 'requests', 'fiona',
                        'future', 'xarray', 'pyproj', 'rasterio',
                        'netcdf4', 'refet', 'bounds'])


# ============= EOF ==============================================================
