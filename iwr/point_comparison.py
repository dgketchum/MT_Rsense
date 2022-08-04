import os

import pandas as pd
import numpy as np

from utils.agrimet import Agrimet, load_stations, MT_STATIONS
from reference_et.temperature import blaney_criddle
from reference_et.combination import pm_fao56, get_rn
from reference_et.rad_utils import extraterrestrial_r, calc_rso
from reference_et.modified_bcriddle import modified_blaney_criddle
from utils.elevation import elevation_from_coordinate


def get_mt_agrimet_stations(outdir):
    station_metadata = load_stations()

    for sid, props in station_metadata.items():
        if sid != 'bozm':
            continue
        out_file = os.path.join(outdir, '{}.csv'.format(sid))
        meta = props['properties']
        region = meta['region']
        if meta['state'] != 'MT':
            continue
        if region not in ['great_plains', 'pnro']:
            continue
        if os.path.exists(out_file):
            continue
        try:
            m, d, y = meta['install'].split('/')
            start_str = '{}-{}-{}'.format(y, str(m).rjust(2, '0'), str(d).rjust(2, '0'))
        except ValueError:
            start_str = '2000-01-01'

        a = Agrimet(station=sid, start_date=start_str,
                    end_date='2021-12-31', interval='daily', region=region)

        a.fetch_met_data(out_csv_file=out_file)
        print(out_file)


def point_comparison_eto(station_dir, out_figs, out_shp):
    stations = load_stations()
    station_files = [os.path.join(station_dir, x) for x in os.listdir(station_dir)]
    for f in station_files:
        sid = os.path.basename(f).split('.')[0]
        if sid != 'bozm':
            continue
        meta = stations[sid]
        coords = meta['geometry']['coordinates']
        coord_rads = np.array(coords) * np.pi / 180
        elev = elevation_from_coordinate(coords[1], coords[0])
        df = pd.read_csv(f, index_col=0, parse_dates=True, infer_datetime_format=True, header=0, skiprows=[1, 2, 3])
        tmean, tmax, tmin, wind, rs, rh = df['MM'], df['MX'], df['MN'], df['UA'], df['SR'], df['TA']
        ra = extraterrestrial_r(df.index, lat=coord_rads[1], shape=[df.shape[0]])
        rso = calc_rso(ra, elev)
        rn = get_rn(tmean, rs=rs, lat=coord_rads[1], tmax=tmax, tmin=tmin, rh=rh, elevation=elev, rso=rso)
        df['ETOS'] = pm_fao56(tmean, wind, rs=rs, tmax=tmax, tmin=tmin, rh=rh, elevation=elev, rn=rn)
        df['ETRS'] = df['ETOS'] * 1.2
        df['ETBC'] = modified_blaney_criddle(df, coords[1])
        pass


if __name__ == '__main__':
    _dir = '/media/research/IrrigationGIS/agrimet/mt_stations'
    fig_dir = '/media/research/IrrigationGIS/agrimet/comparison_figures'
    out_shp = '/media/research/IrrigationGIS/agrimet/shapefiles/comparison.shp'

    get_mt_agrimet_stations(_dir)
    point_comparison_eto(station_dir=_dir, out_figs=fig_dir, out_shp=out_shp)
# ========================= EOF ====================================================================
