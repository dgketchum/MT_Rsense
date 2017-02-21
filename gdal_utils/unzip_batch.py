# ===============================================================================
# Copyright 2017 dgketchum
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

# standard library imports ======================================================
import os, zipfile, tarfile

# local imports ======================================================


def unzip_all(dir_name, _type='folders', replace_ext='exe'):
    if _type == 'folders':
        for item in os.listdir(dir_name):
            print 'dirtype {} \n dirs: {}'.format(type(os.listdir(dir_name)), os.listdir(dir_name))
            print 'folder name: {}'.format(item)
            if item.endswith('.zip'):
                file_name = os.path.join(dir_name, item)
                zip_ref = zipfile.ZipFile(file_name)
                print 'unzipping {}'.format(item)
                zip_ref.extractall(dir_name)
                zip_ref.close()
                os.remove(file_name)
        # if os.li
            elif file_name.endswith("tar.gz"):
                print 'unzipping {}'.format(file_name)
                tar = tarfile.open(item, "r:gz")
                tar.extractall(file_name.replace('tar.gz', replace_ext))
                tar.close()
            elif file_name.endswith("tar"):
                print 'unzipping {}'.format(file_name)
                tar = tarfile.open(item, "r:")
                tar.extractall(file_name.replace('tar', replace_ext))
                tar.close()

    elif _type == 'files':
        for item in os.listdir(dir_name):
            if item.endswith('.zip'):
                file_name = os.path.join(dir_name, item)
                zip_ref = zipfile.ZipFile(file_name)
                print 'unzipping {}'.format(item)
                zip_ref.extractall(file_name)
                zip_ref.close()
                os.remove(file_name)
            elif item.endswith("tar.gz"):
                file_name = os.path.join(dir_name, item)
                print 'unzipping {}'.format(file_name)
                tar = tarfile.open(file_name, "r:gz")
                tar.extractall(file_name.replace('tar.gz', replace_ext))
                tar.close()
                os.remove(file_name)
                unzip_all(dir_name, _type='files')
                return None
            elif item.endswith("tar"):
                file_name = os.path.join(dir_name, item)
                print 'unzipping {}'.format(file_name)
                tar = tarfile.open(file_name, "r:")
                tar.extractall(file_name.replace('tar', replace_ext))
                tar.close()


if __name__ == '__main__':
    home = os.path.expanduser('~')
    print 'home: {}'.format(home)
    root = os.path.abspath(os.path.join(home, os.pardir))
    images = os.path.join(root, 'images')
    tiles = os.path.join(images, 'executables')
    unzip_all(tiles, _type='files')


# ============= EOF ============================================================

