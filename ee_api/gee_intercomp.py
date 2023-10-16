import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import ee
from ee_api import is_authorized

MODELS = {'ssebop': 'OpenET/SSEBOP/CONUS/GRIDMET/MONTHLY/v2_0',
          'ptjpl': 'OpenET/PTJPL/CONUS/GRIDMET/MONTHLY/v2_0',
          'eemetric': 'OpenET/EEMETRIC/CONUS/GRIDMET/MONTHLY/v2_0',
          'disalexi': 'OpenET/DISALEXI/CONUS/GRIDMET/MONTHLY/v2_0',
          'ensemble': 'OpenET/ENSEMBLE/CONUS/GRIDMET/MONTHLY/v2_0',
          'geesebal': 'OpenET/GEESEBAL/CONUS/GRIDMET/MONTHLY/v2_0'}


def fields_intercomparison(fields, models, start, end):
    desc = os.path.basename(fields)
    fields = ee.FeatureCollection(fields)
    # fields = fields.filterMetadata('FID', 'less_than', ee.Number(3))

    first, i, data = True, None, None
    s_doy = int(datetime.strptime(start, '%Y-%m-%d').strftime('%j'))
    e_doy = int(datetime.strptime(end, '%Y-%m-%d').strftime('%j'))
    for model, asset in models.items():
        c = ee.ImageCollection(asset).filterDate(start, end).filter(ee.Filter.dayOfYear(s_doy, e_doy))
        if model == 'ensemble':
            c = c.select('et_ensemble_mad')
        else:
            c = c.select('et')
        if first:
            i = c.sum().rename(model)
            first = False
        else:
            i = i.addBands([c.sum().rename(model)])

        data = i.reduceRegions(collection=fields,
                               reducer=ee.Reducer.mean(),
                               scale=30)

        # p = data.first().getInfo()['properties']
        # print('propeteries {}'.format(p))

    selectors = ['FID'] + list(models.keys())
    task = ee.batch.Export.table.toCloudStorage(
        data,
        description=desc,
        bucket='wudr',
        fileNamePrefix=desc,
        fileFormat='CSV',
        selectors=selectors)

    task.start()
    print(desc)


def plot_box_and_whiskers(csv_file):
    df = pd.read_csv(csv_file, index_col='FID')
    df /= 7.0
    overall_mean = np.median(df.values)

    sns.set(style="whitegrid")

    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(data=df, orient="v", palette="Set3")

    plt.title("ET Model Results (n= {} fields)".format(df.shape[0]))
    plt.ylabel("Mean Annual Growing Season ET (mm), 2016 - 2022")
    plt.xticks(rotation=45)

    ax.axhline(overall_mean, color='red', linestyle='--', label=f'Overall Median: {overall_mean:.2f}')

    plt.tight_layout()
    plt.legend()
    plt.savefig(csv_file.replace('.csv', '.png'))


if __name__ == '__main__':
    is_authorized()
    shp = 'users/dgketchum/fields/idaho_tv_mason'
    # fields_intercomparison(shp, MODELS, start='2016-04-01', end='2022-11-01')

    plot_box_and_whiskers('/home/dgketchum/Downloads/idaho_tv_mason.csv')
# ========================= EOF ================================================================================
