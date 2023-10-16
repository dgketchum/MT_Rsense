import os
import json
from datetime import datetime

import numpy as np
from pandas import DataFrame, date_range, read_csv
from gsflow.prms.prms_parameter import ParameterRecord
from gsflow.control import ControlRecord
from gsflow.builder import builder_utils as bu
import matplotlib
import matplotlib.pyplot as plt

from gsflow_model_prep import StandardPrmsBuild, MontanaPrmsModel
from datafile import write_basin_datafile
from utils.thredds import GridMet
from utils.bounds import GeoBounds
import warnings
from gsflow.prms import PrmsData, PrmsParameters

warnings.simplefilter(action='ignore', category=DeprecationWarning)


class CbhruPrmsBuild(StandardPrmsBuild):

    def __init__(self, config):
        StandardPrmsBuild.__init__(self, config)
        self.data_file = os.path.join(self.cfg.data_folder,
                                      '{}_runoff.data'.format(self.proj_name_res))

    def write_datafile(self,):

        gages = self.cfg.prms_data_gages

        self.data_params = [ParameterRecord(name='nobs', values=[1, ], datatype=1)]
        self.data_params.append(ParameterRecord('nrain', values=[0], datatype=1))
        self.data_params.append(ParameterRecord('ntemp', values=[0], datatype=1))

        self.data_params.append(ParameterRecord('snowpack_init',
                                                np.zeros_like(self.ksat).ravel(),
                                                dimensions=[['nhru', self.nhru]],
                                                datatype=2))

        if not os.path.isfile(self.data_file):
            units = 'metric' if self.cfg.runoff_units == 1 else 'standard'
            write_basin_datafile(gages, self.data_file, units=units)

        self.data = PrmsData.load_from_file(self.data_file)

    def write_day_files(self):
        s = datetime.strptime(self.cfg.start_time, '%Y-%m-%d')
        e = datetime.strptime(self.cfg.end_time, '%Y-%m-%d')

        dt_range = date_range(start=self.cfg.start_time,
                              end=self.cfg.end_time,
                              freq='D')

        bounds = GeoBounds(north=self.lat.max(),
                           west=self.lon.min(),
                           east=self.lon.max(),
                           south=self.lat.min())

        req_vars = {'pr': 'precip_day',
                    'tmmx': 'tmax_day',
                    'tmmn': 'tmin_day',
                    'rmin': None,
                    'rmax': None,
                    'pet': 'potet_day',
                    'srad': 'swrad_day',
                    'vs': 'windspeed_day'}

        day_files = [v for k, v in req_vars.items() if v is not None] + ['humidity_day']
        day_files = [os.path.join(self.cfg.data_folder, d.replace('_', '.')) for d in day_files]
        cont_recs = [ControlRecord(os.path.basename(_file).replace('.', '_'), [_file]) for _file in day_files]
        [self.control_records.append(cr) for cr in cont_recs]

        for var_, day in req_vars.items():

            if all([os.path.exists(f) for f in day_files]):
                break

            gridmet = GridMet(start=s,
                              end=e,
                              variable=var_,
                              target_profile=self.raster_meta,
                              bbox=bounds,
                              clip_feature=[self.study_area.envelope])
            vals = gridmet.subset_daily_tif()

            if var_ == 'rmin':
                rmin = np.copy(vals)
                continue

            if var_ in ['tmmn', 'tmmx']:
                vals = np.where(vals > 0, vals - 273.15, np.zeros_like(vals))
                if self.cfg.temp_units == 0:
                    vals = (vals * 9 / 5) + 32.

            if var_ == 'pr':
                vals = np.where(vals > 30.0, np.ones_like(vals) * 29.9, vals)
                if self.cfg.precip_units == 1:
                    vals = vals / 25.4

            if var_ == 'vs':
                pass

            if var_ == 'pet':
                vals = vals / 25.4

            if var_ == 'rmax':
                rmax = np.copy(vals)
                vals = (rmin + rmax) / 2.
                _name = 'humidity'
                _file = os.path.join(self.cfg.data_folder, '{}.day'.format(_name))

            if day is not None:
                _name = day.split('_')[0]
                _file = os.path.join(self.cfg.data_folder, '{}.day'.format(_name))

            if os.path.exists(_file):
                continue

            rng = [str(x) for x in range(vals.shape[1] * vals.shape[2])]
            df = DataFrame(index=dt_range, columns=rng,
                           data=vals.reshape(vals.shape[0], vals.shape[1] * vals.shape[2]))

            time_div = ['Year', 'Month', 'day', 'hr', 'min', 'sec']
            df['Year'] = [i.year for i in df.index]
            df['Month'] = [i.month for i in df.index]
            df['day'] = [i.day for i in df.index]
            for t_ in time_div[3:]:
                df[t_] = [0 for _ in df.index]

            df = df[time_div + rng]

            with open(_file, 'w') as f:
                f.write('Generated by Ketchum\n')
                f.write('{}\t{}\n'.format(_name, self.nhru))
                f.write('########################################\n')
                df.to_csv(f, sep=' ', header=False, index=False, float_format='%.1f')
            print('write {}'.format(_file))

    def build_model(self):
        self._build_grid()
        self.write_datafile()
        self.write_day_files()
        self.build_parameters()
        self.build_controls()
        self.write_parameters()
        self.write_control()

    def write_parameters(self):

        rain_snow_adj = np.ones((self.nhru * self.nmonths), dtype=float)

        self.data_params.append(ParameterRecord('rain_cbh_adj', rain_snow_adj,
                                                dimensions=[['nhru', self.nhru], ['nmonths', self.nmonths]],
                                                datatype=2))

        self.data_params.append(ParameterRecord('snow_cbh_adj', rain_snow_adj,
                                                dimensions=[['nhru', self.nhru], ['nmonths', self.nmonths]],
                                                datatype=2))

        [self.parameters.add_record_object(rec) for rec in self.data_params]

        self.parameters.write(self.parameter_file)

    def write_control(self):
        self.control.add_record('elev_units', [self.cfg.elev_units])
        self.control.add_record('precip_units', [self.cfg.precip_units])
        self.control.add_record('temp_units', [self.cfg.temp_units])
        self.control.add_record('runoff_units', [self.cfg.runoff_units])

        self.control.precip_module = ['climate_hru']
        self.control.temp_module = ['climate_hru']
        self.control.et_module = ['climate_hru']
        self.control.solrad_module = ['climate_hru']

        if self.control_records is not None:
            [self.control.add_record(rec.name, rec.values) for rec in self.control_records]

        self.control.write(self.control_file)


class XyzDistBuild(StandardPrmsBuild):

    def __init__(self, config):
        StandardPrmsBuild.__init__(self, config)
        self.data_file = os.path.join(self.cfg.data_folder,
                                      '{}_xyz.data'.format(self.proj_name_res))

    def write_datafile(self):

        self.nmonths = 12

        ghcn = self.cfg.prms_data_ghcn
        stations = self.cfg.prms_data_stations
        gages = self.cfg.prms_data_gages

        with open(stations, 'r') as js:
            sta_meta = json.load(js)

        sta_iter = sorted([(v['zone'], v) for k, v in sta_meta.items()], key=lambda x: x[0])
        tsta_elev, tsta_nuse, tsta_x, tsta_y, psta_elev = [], [], [], [], []
        for _, val in sta_iter:

            if self.cfg.elev_units == 1:
                elev = val['elev'] / 0.3048
            else:
                elev = val['elev']

            tsta_elev.append(elev)
            tsta_nuse.append(1)
            tsta_x.append(val['proj_coords'][1])
            tsta_y.append(val['proj_coords'][0])
            psta_elev.append(elev)

        self.data_params = [ParameterRecord('nrain', values=[len(tsta_x)], datatype=1),

                            ParameterRecord('ntemp', values=[len(tsta_x)], datatype=1),

                            ParameterRecord('psta_elev', np.array(psta_elev, dtype=float).ravel(),
                                            dimensions=[['nrain', len(psta_elev)]], datatype=2),

                            ParameterRecord('psta_nuse', np.array(tsta_nuse, dtype=int).ravel(),
                                            dimensions=[['nrain', len(tsta_nuse)]], datatype=1),

                            ParameterRecord(name='ndist_psta', values=[len(tsta_nuse), ], datatype=1),

                            ParameterRecord('psta_x', np.array(tsta_x, dtype=float).ravel(),
                                            dimensions=[['nrain', len(tsta_x)]], datatype=2),

                            ParameterRecord('psta_y', np.array(tsta_y, dtype=float).ravel(),
                                            dimensions=[['nrain', len(tsta_y)]], datatype=2),

                            ParameterRecord('tsta_elev', np.array(tsta_elev, dtype=float).ravel(),
                                            dimensions=[['ntemp', len(tsta_elev)]], datatype=2),

                            ParameterRecord('tsta_nuse', np.array(tsta_nuse, dtype=int).ravel(),
                                            dimensions=[['ntemp', len(tsta_nuse)]], datatype=1),

                            ParameterRecord(name='ndist_tsta', values=[len(tsta_nuse), ], datatype=1),

                            ParameterRecord('tsta_x', np.array(tsta_x, dtype=float).ravel(),
                                            dimensions=[['ntemp', len(tsta_x)]], datatype=2),

                            ParameterRecord('tsta_y', np.array(tsta_y, dtype=float).ravel(),
                                            dimensions=[['ntemp', len(tsta_y)]], datatype=2),

                            bu.tmax_adj(self.nhru),

                            bu.tmin_adj(self.nhru),

                            ParameterRecord(name='nobs', values=[1, ], datatype=1),
                            ]

        if self.cfg.temp_units == 1:
            allrain_max = np.ones((self.nhru * self.nmonths)) * 3.3
        else:
            allrain_max = np.ones((self.nhru * self.nmonths)) * 38.0

        self.data_params.append(ParameterRecord('tmax_allrain_sta', allrain_max,
                                                dimensions=[['nhru', self.nhru], ['nmonths', self.nmonths]],
                                                datatype=2))

        self.data_params.append(ParameterRecord('snowpack_init',
                                                np.ones_like(self.ksat).ravel(),
                                                dimensions=[['nhru', self.nhru]],
                                                datatype=2))

        if not os.path.isfile(self.data_file):
            units = 'metric' if self.cfg.precip_units == 1 else 'standard'
            write_basin_datafile(gage_json=gages, data_file=self.data_file, station_json=stations, ghcn_data=ghcn,
                                 out_csv=None, units=units)

        self.data = PrmsData.load_from_file(self.data_file)

    def build_model(self):
        self._build_grid()
        self.write_datafile()
        self.build_parameters()
        self.build_controls()
        self.write_parameters()
        self.write_control()

    def write_parameters(self):
        if self.data_params is not None:
            [self.parameters.add_record_object(rec) for rec in self.data_params]

        self.parameters.write(self.parameter_file)

    def write_control(self):
        # 0: standard; 1: SI/metric
        units = 0
        self.control.add_record('elev_units', [units])
        self.control.add_record('precip_units', [units])
        self.control.add_record('temp_units', [units])
        self.control.add_record('runoff_units', [units])

        self.control.precip_module = ['xyz_dist']
        self.control.temp_module = ['xyz_dist']
        self.control.et_module = ['potet_jh']
        self.control.solrad_module = ['ccsolrad']

        if self.control_records is not None:
            [self.control.add_record(rec) for rec in self.control_records]

        self.control.write(self.control_file)


def plot_stats(stats, file=None):
    fig, ax = plt.subplots(figsize=(16, 6))
    stats = stats.loc['2017-01-01': '2017-12-31']
    ax.plot(stats.Date, stats.basin_cfs_1, color='r', linewidth=2.2, label="simulated")
    ax.plot(stats.Date, stats.runoff_1, color='b', linewidth=1.5, label="measured")
    ax.legend(bbox_to_anchor=(0.25, 0.65))
    ax.set_xlabel("Date")
    ax.set_ylabel("Streamflow, in cfs")
    # ax.set_ylim([0, 2000])

    if file:
        plt.savefig(file)
    else:
        plt.show()

    plt.close()


def read_calibration(params_dir):
    params_ = ['adjmix_rain',
               'tmax_allsnow',
               'srain_intcp',
               'wrain_intcp',
               'cecn_coef',
               'emis_noppt',
               'freeh2o_cap',
               'potet_sublim',
               'carea_max',
               'smidx_coef',
               'smidx_exp',
               'fastcoef_lin',
               'fastcoef_sq',
               'pref_flow_den',
               'sat_threshold',
               'slowcoef_lin',
               'slowcoef_sq',
               'soil_moist_max',
               'soil_rechr_max',
               'soil2gw_max',
               'ssr2gw_exp',
               'ssr2gw_rate',
               'transp_tmax',
               'gwflow_coef']

    dct = {k: None for k in params_}
    l = sorted([os.path.join(params_dir, x) for x in os.listdir(params_dir)])
    first = True
    for i, ll in enumerate(l):
        params = PrmsParameters.load_from_file(ll)
        if first:
            for p in params_:
                vals = params.get_values(p)
                dct[p] = vals.mean()
            first = False
            continue

        for p in params_:
            vals = params.get_values(p)
            new_val = vals.mean()
            delta = new_val - dct[p]
            print('{:.3f} {} delta'.format(delta, p))
            dct[p] = new_val
            if p == 'ssr2gw_exp':
                pass

        if i == len(l) - 1:
            print('final luca parameter values')
            for p in params_:
                vals = params.get_values(p)
                new_val = vals.mean()
                print('{:.3f} {} final values'.format(new_val, p))


def compare_parameters(model, csv):

    df = read_csv(csv)
    df = df.mean(axis=0)

    param_names = model.parameters.record_names

    comp_params = [x for x in df.index if x in param_names]

    for p in comp_params:
        mp = prms.parameters.get_values(p).mean()
        gf = df[p]
        print('p: {}, actual: {:.3f}, geofabric: {:.3f}'.format(p, mp, gf))

    pass


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/Montana/upper_yellowstone/gsflow_prep'
    matplotlib.use('TkAgg')

    conf = './model_files/uyws_parameters.ini'
    project = os.path.join(root, 'uyws_carter_5000')
    luca_dir = os.path.join(project, 'input', 'luca')
    stdout_ = os.path.join(project, 'output', 'stdout.txt')
    snodas = os.path.join(project, 'input', 'carter_basin_snodas.csv')

    csv = '/media/research/IrrigationGIS/Montana/geospatial_fabric/prms_params_carter.csv'

    prms_build = CbhruPrmsBuild(conf)
    prms_build.build_model()

    luca_params = os.path.join(luca_dir, 'calib1_round3_step2.par')
    # read_calibration(luca_dir)

    prms = MontanaPrmsModel(prms_build.control_file,
                            prms_build.parameter_file,
                            prms_build.data_file)

    prms.run_model(stdout_)
    stats_uncal = prms.get_statvar()
    fig_ = os.path.join(project, 'output', 'hydrograph_uncal.png')
    plot_stats(stats_uncal, fig_)

    prms = MontanaPrmsModel(prms_build.control_file,
                            luca_params,
                            prms_build.data_file)

    compare_parameters(prms, csv)

    prms.run_model()
    stats_cal = prms.get_statvar(snow=snodas)
    fig_ = os.path.join(project, 'output', 'hydrograph_cal.png')
    plot_stats(stats_cal, fig_)
    pass

# ========================= EOF ====================================================================
