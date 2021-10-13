import os

from pandas import read_csv, to_datetime


def hydrograph(c):
    df = read_csv(c)
    if 'Unnamed: 0' in list(df.columns):
        df = df.rename(columns={'Unnamed: 0': 'dt'})
    try:
        df['dt'] = to_datetime(df['dt'])
    except KeyError:
        df['dt'] = to_datetime(df['datetimeUTC'])
    df = df.set_index('dt')
    try:
        df.drop(columns='datetimeUTC', inplace=True)
    except KeyError:
        pass
    try:
        df = df.tz_convert(None)
    except:
        pass
    return df


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
