import os
from copy import deepcopy
from datetime import timedelta, datetime
from calendar import monthrange

import pandas as pd
import numpy as np

lat_to_sunshine = {50: [5.99, 6.32, 8.24, 9.24, 10.68, 10.92, 10.99, 9.99, 8.46, 7.44, 6.08, 5.65],
                   49: [6.08, 6.36, 8.25, 9.20, 10.60, 10.82, 10.90, 9.94, 8.46, 7.48, 6.16, 5.75],
                   48: [6.17, 6.41, 8.26, 9.17, 10.52, 10.72, 10.81, 9.89, 8.45, 7.51, 6.24, 5.85],
                   47: [6.25, 6.45, 8.27, 9.14, 10.45, 10.63, 10.73, 9.84, 8.44, 7.54, 6.31, 5.95],
                   46: [6.33, 6.50, 8.28, 9.11, 10.38, 10.53, 10.65, 9.79, 8.43, 7.58, 6.37, 6.05],
                   45: [6.40, 6.54, 8.29, 9.08, 10.31, 10.46, 10.57, 9.75, 8.42, 7.61, 6.43, 6.14],
                   44: [6.48, 6.57, 8.29, 9.05, 10.25, 10.39, 10.49, 9.71, 8.41, 7.64, 6.50, 6.22],
                   43: [6.55, 6.61, 8.30, 9.02, 10.19, 10.31, 10.42, 9.66, 8.40, 7.67, 6.56, 6.31],
                   42: [6.61, 6.65, 8.30, 8.99, 10.13, 10.24, 10.35, 9.62, 8.40, 7.70, 6.62, 6.39],
                   41: [6.68, 6.68, 8.31, 8.96, 10.07, 10.16, 10.29, 9.59, 8.39, 7.72, 6.68, 6.47],
                   40: [6.75, 6.72, 8.32, 8.93, 10.01, 10.09, 10.22, 9.55, 8.39, 7.75, 6.73, 6.54]}

alfalfa_kc = [0.63, 0.73, 0.86, 0.99, 1.08, 1.13, 1.11, 1.06, 0.99, 0.91, 0.78, 0.64]


def modified_blaney_criddle_2d(df, lat_degrees=None):
    mid_months = [i for i, _ in enumerate(df.time.values) if pd.to_datetime(_).day == 15]
    t = df['MM'][mid_months].resample(time='1M').mean()
    t = t.groupby('time.month').mean() * 9 / 5 + 32

    p = df['PP'].resample(time='1M').sum()
    p = p.groupby('time.month').mean() / 25.4

    # https://stackoverflow.com/questions/54776283/how-to-call-the-xarrays-groupby-function-to-group-data-by-a-combination-of-year
    dtmm = df['MM'].grouppby([df.index.month, df.index.day]).mean() * 9 / 5 + 32
    yr_ind = pd.date_range('2000-01-01', '2000-12-31', freq='d')
    dtmm.index = yr_ind
    season_start = dtmm[dtmm > 50.].index[0]

    dtmn = df['MN'].groupby([df.index.month, df.index.day]).mean() * 9 / 5 + 32
    yr_ind = pd.date_range('2000-01-01', '2000-12-31', freq='d')
    dtmn.index = yr_ind
    season_end = dtmn.loc['2000-07-01':][dtmn < 28.].index[0]

    season_length = (season_end - season_start).days

    lat = round(lat_degrees)
    sunshine = lat_to_sunshine[lat]

    first_period = []
    d = season_start
    while d.day != 1:
        first_period.append(d)
        d += timedelta(days=1)

    midpoint = season_start + (d - season_start) / 2
    counter = (midpoint - season_start).days
    t_prev, t_next = t.loc[midpoint.month], t.loc[midpoint.month + 1]
    remaining_days = (midpoint - season_start).days
    month_len = monthrange(2000, midpoint.month)[1]
    month_fraction = remaining_days / month_len
    temp = t_prev + (month_fraction * (t_next - t_prev))

    day_prev, day_next = sunshine[midpoint.month - 1], sunshine[midpoint.month]
    daylight = (day_prev + (month_fraction * (day_next - day_prev))) * month_fraction

    dates, d_accum, pct_season = [midpoint], [counter], [counter / season_length]
    temps, pct_day_hrs = [temp], [daylight]
    for d, v in dtmm.loc[midpoint: season_end].iteritems():
        counter += 1
        if d.day == 15:
            dates.append(d)
            d_accum.append(counter)
            pct_season.append(counter / season_length)
            temps.append(t.loc[d.month])
            pct_day_hrs.append(sunshine[d.month - 1])

    second_period = []
    d = dates[-1]
    while d != season_end:
        second_period.append(d)
        d += timedelta(days=1)

    midpoint = second_period[0] + (second_period[-1] - second_period[0]) / 2
    t_prev, t_next = t.loc[midpoint.month], t.loc[midpoint.month + 1]
    remaining_days = (midpoint - dates[-1]).days
    month_len = monthrange(2000, midpoint.month)[1]
    month_fraction = remaining_days / month_len
    temp = t_prev + (month_fraction * (t_next - t_prev))
    accum_days_last = (midpoint - season_start).days

    day_prev, day_next = sunshine[midpoint.month - 1], sunshine[midpoint.month]
    daylight = (day_prev + (month_fraction * (day_next - day_prev))) * month_fraction

    dates.append(midpoint)
    d_accum.append(accum_days_last)
    pct_season.append(accum_days_last / season_length)
    temps.append(temp)
    pct_day_hrs.append(daylight)

    dates = [pd.to_datetime('2000-{}-{}'.format(d.month, d.day)) for d in dates]
    df = pd.DataFrame(np.array([d_accum, pct_season, temps, pct_day_hrs]).T,
                      columns=['accum_day', 'pct_season', 't', 'p'],
                      index=dates)

    df['f'] = df['t'] * df['p'] / 100.

    df['kt'] = df['t'] * 0.0173 - 0.314

    kc = pd.Series(alfalfa_kc, index=[d for d in yr_ind if d.day == 15])
    kc = kc.reindex(yr_ind)
    kc.iloc[0] = 0.6
    kc.iloc[-1] = 0.6
    kc = kc.interpolate()
    df['kc'] = kc.loc[df.index]
    df['k'] = df['kc'] * df['kt']
    df['u'] = df['k'] * df['f']
    df['ref_u'] = df['kt'] * df['f']
    return df, season_start, season_end


def modified_blaney_criddle(df, lat_degrees=None):
    mid_months = [d for d in df.index if d.day == 15]
    t = df['MM'].loc[mid_months].resample('M').mean()
    t = t.groupby(t.index.month).mean() * 9 / 5 + 32

    p = df['PP'].resample('M').sum()
    p = p.groupby(p.index.month).mean() / 25.4

    dtmm = df['MM'].groupby([df.index.month, df.index.day]).mean() * 9 / 5 + 32
    yr_ind = pd.date_range('2000-01-01', '2000-12-31', freq='d')
    dtmm.index = yr_ind
    season_start = dtmm[dtmm > 50.].index[0]

    dtmn = df['MN'].groupby([df.index.month, df.index.day]).mean() * 9 / 5 + 32
    yr_ind = pd.date_range('2000-01-01', '2000-12-31', freq='d')
    dtmn.index = yr_ind
    season_end = dtmn.loc['2000-07-01':][dtmn < 28.].index[0]

    season_length = (season_end - season_start).days

    lat = round(lat_degrees)
    sunshine = lat_to_sunshine[lat]

    first_period = []
    d = season_start
    while d.day != 1:
        first_period.append(d)
        d += timedelta(days=1)

    midpoint = season_start + (d - season_start) / 2
    counter = (midpoint - season_start).days
    t_prev, t_next = t.loc[midpoint.month], t.loc[midpoint.month + 1]
    remaining_days = (midpoint - season_start).days
    month_len = monthrange(2000, midpoint.month)[1]
    month_fraction = remaining_days / month_len
    temp = t_prev + (month_fraction * (t_next - t_prev))

    day_prev, day_next = sunshine[midpoint.month - 1], sunshine[midpoint.month]
    daylight = (day_prev + (month_fraction * (day_next - day_prev))) * month_fraction

    dates, d_accum, pct_season = [midpoint], [counter], [counter / season_length]
    temps, pct_day_hrs = [temp], [daylight]
    for d, v in dtmm.loc[midpoint: season_end].iteritems():
        counter += 1
        if d.day == 15:
            dates.append(d)
            d_accum.append(counter)
            pct_season.append(counter / season_length)
            temps.append(t.loc[d.month])
            pct_day_hrs.append(sunshine[d.month - 1])

    second_period = []
    d = dates[-1]
    while d != season_end:
        second_period.append(d)
        d += timedelta(days=1)

    midpoint = second_period[0] + (second_period[-1] - second_period[0]) / 2
    t_prev, t_next = t.loc[midpoint.month], t.loc[midpoint.month + 1]
    remaining_days = (midpoint - dates[-1]).days
    month_len = monthrange(2000, midpoint.month)[1]
    month_fraction = remaining_days / month_len
    temp = t_prev + (month_fraction * (t_next - t_prev))
    accum_days_last = (midpoint - season_start).days

    day_prev, day_next = sunshine[midpoint.month - 1], sunshine[midpoint.month]
    daylight = (day_prev + (month_fraction * (day_next - day_prev))) * month_fraction

    dates.append(midpoint)
    d_accum.append(accum_days_last)
    pct_season.append(accum_days_last / season_length)
    temps.append(temp)
    pct_day_hrs.append(daylight)

    dates = [pd.to_datetime('2000-{}-{}'.format(d.month, d.day)) for d in dates]
    df = pd.DataFrame(np.array([d_accum, pct_season, temps, pct_day_hrs]).T,
                      columns=['accum_day', 'pct_season', 't', 'p'],
                      index=dates)

    df['f'] = df['t'] * df['p'] / 100.

    df['kt'] = df['t'] * 0.0173 - 0.314

    kc = pd.Series(alfalfa_kc, index=[d for d in yr_ind if d.day == 15])
    kc = kc.reindex(yr_ind)
    kc.iloc[0] = 0.6
    kc.iloc[-1] = 0.6
    kc = kc.interpolate()
    df['kc'] = kc.loc[df.index]
    df['k'] = df['kc'] * df['kt']
    df['u'] = df['k'] * df['f']
    df['ref_u'] = df['kt'] * df['f']
    return df, season_start, season_end


if __name__ == '__main__':
    pass
# ========================= ========================================================
