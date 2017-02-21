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
import numpy as np
import pandas
import os

# local imports ======================================================


def save_a_csv(save_path, fname):
    a = np.arange(100).reshape((10, 10))
    head_str = list('abcdefghig')
    df = pandas.DataFrame(a, columns=head_str, dtype=int)
    df.to_csv(os.path.join(save_path, fname))


if __name__ == '__main__':
    home = os.path.expanduser('~')
    print 'home: {}'.format(home)
    root = os.path.abspath(os.path.join(home, os.pardir))
    images = os.path.join(root, 'images')
    name = 'test_csv.csv'
    save_a_csv(images, name)


# ============= EOF ============================================================

