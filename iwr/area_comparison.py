import os

import xarray as xr
from reference_et.modified_bcriddle import modified_blaney_criddle_2d


def area_comparison_iwr_asce(_dir, co_list):

    rename = {'pet': ('daily_mean_reference_evapotranspiration_grass', 'ETOS'),
              'pr': ('precipitation_amount', 'PP'),
              'tmmn': ('daily_minimum_temperature', 'MN'),
              'tmmx': ('daily_maximum_temperature', 'MX')}

    for co in co_list:
        l = [os.path.join(_dir, f) for f in os.listdir(_dir) if f.split('_')[3][:5] == co]
        first = True
        for f in l:
            var = os.path.basename(f).split('_')[0]
            if first:
                ds = xr.open_dataset(f).rename({rename[var][0]: rename[var][1]})
                first = False
            else:
                arr = xr.open_dataset(f).rename({rename[var][0]: rename[var][1]})
                ds = xr.merge([ds, arr])

        ds['MM'] = (ds['MN'] + ds['MX']) / 2
        etbc = modified_blaney_criddle_2d(ds)


if __name__ == '__main__':
    if __name__ == '__main__':
        d = '/media/research/IrrigationGIS'
        if not os.path.exists(d):
            d = '/home/dgketchum/data/IrrigationGIS'

        nc_dir = os.path.join(d, 'climate', 'gridmet_nc', 'mt_counties')
        co_l = ['30067', '30097']
        area_comparison_iwr_asce(nc_dir, co_l, )
# ========================= EOF ====================================================================
