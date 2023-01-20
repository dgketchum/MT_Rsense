import os
import sys
from pprint import pprint
from copy import deepcopy

from numpy import ones_like, where, zeros_like
from pandas import read_csv, DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from geopandas import GeoDataFrame
from shapely.geometry import Point

abspath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(abspath)

INT_COLS = ['POINT_TYPE', 'YEAR', 'classification']
CLASS_NAMES = ['IRR', 'DRYL', 'WETl', 'UNCULT']


def consumer(arr):
    c = [(arr[x, x] / sum(arr[x, :])) for x in range(0, arr.shape[1])]
    return c


def producer(arr):
    c = [(arr[x, x] / sum(arr[:, x])) for x in range(0, arr.shape[0])]
    return c


def random_forest(csv):
    if not isinstance(csv, DataFrame):
        print('\n', csv)
        c = read_csv(csv, engine='python').sample(frac=1.0).reset_index(drop=True)
    else:
        c = csv

    c.drop(columns=['system:index', '.geo'], inplace=True)
    pt = c['POINT_TYPE'].values
    c['POINT_TYPE'] = [1 if x == 3 else 0 for x in pt]
    c['nd'] = (c['N'] - c['R']) / (c['N'] + c['R'])
    drop = (c['nd'] > 0.0) & (c['POINT_TYPE'] == 1)
    c = c[~drop]
    split = int(c.shape[0] * 0.7)

    df = deepcopy(c.loc[:split, :])
    y = df['POINT_TYPE'].values
    df.drop(columns=['POINT_TYPE'], inplace=True)
    df.dropna(axis=1, inplace=True)
    x = df.values

    val = deepcopy(c.loc[split:, :])
    y_test = val['POINT_TYPE'].values
    val.drop(columns=['POINT_TYPE'], inplace=True)
    val.dropna(axis=1, inplace=True)
    x_test = val.values

    rf = RandomForestClassifier(n_jobs=-1,
                                bootstrap=True)

    rf.fit(x, y)
    y_pred = rf.predict(x_test)

    cf = confusion_matrix(y_test, y_pred)
    pprint(cf)
    pprint(producer(cf))
    pprint(consumer(cf))

    return


if __name__ == '__main__':
    d = '/media/research/IrrigationGIS/Montana'
    if not os.path.exists(d):
        d = '/home/dgketchum/IrrigationGIS/Montana'

    r = os.path.join(d, 'naip_water_model')
    c = os.path.join(r, 'naip_2017.csv')
    random_forest(c)
# ========================= EOF ====================================================================
