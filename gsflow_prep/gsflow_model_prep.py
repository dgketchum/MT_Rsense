import os
import json
from copy import copy
from subprocess import call, Popen, PIPE, STDOUT
import time

import numpy as np
import pandas as pd
from pyproj import Transformer
import rasterio
import fiona
from affine import Affine
from shapely.geometry import shape
from scipy.ndimage.morphology import binary_erosion
from pandas.plotting import register_matplotlib_converters
import matplotlib
import matplotlib.pyplot as plt

import flopy
from flopy.utils import GridIntersect
import richdem as rd
from gsflow.builder import GenerateFishnet, FlowAccumulation, PrmsBuilder, ControlFileBuilder
from gsflow.builder.builder_defaults import ControlFileDefaults
from gsflow.builder import builder_utils as bu
from gsflow.prms.prms_parameter import ParameterRecord
from gsflow.prms import PrmsData, PrmsParameters
from gsflow.control import ControlFile
from gsflow.output import StatVar

from model_config import PRMSConfig
from gsflow_prep import PRMS_NOT_REQ
from datafile import write_basin_datafile

register_matplotlib_converters()
pd.options.mode.chained_assignment = None

# RichDEM flow-direction coordinate system:
# 234
# 105
# 876

d8_map = {5: 1, 6: 2, 7: 4, 8: 8, 1: 16, 2: 32, 3: 64, 4: 128}


class StandardPrmsBuild(object):

    def __init__(self, config):

        self.cfg = PRMSConfig(config)
        self.res = float(self.cfg.hru_cellsize)
        self.proj_name_res = '{}_{}'.format(self.cfg.project_name,
                                            self.cfg.hru_cellsize)

        for folder in ['hru_folder', 'parameter_folder', 'control_folder', 'data_folder', 'output_folder']:
            folder_path = os.path.join(self.cfg.project_folder,
                                       self.proj_name_res,
                                       getattr(self.cfg, folder))
            setattr(self.cfg, folder, folder_path)
            if not os.path.isdir(folder_path):
                os.makedirs(folder_path, exist_ok=True)

        self.parameters = None
        self.control = None
        self.data = None

        self.zeros = None

        with fiona.open(self.cfg.study_area_path, 'r') as src:
            self.raster_meta = src.meta
            self.basin_geo = [shape(f['geometry']) for f in src][0]
            self.prj = self.cfg.study_area_path.replace('.shp', '.prj')

        self.control_file = os.path.join(self.cfg.control_folder,
                                         '{}.control'.format(self.proj_name_res))

        self.parameter_file = os.path.join(self.cfg.parameter_folder,
                                           '{}.params'.format(self.proj_name_res))

        self.data_file = os.path.join(self.cfg.data_folder, '{}.data'.format(self.proj_name_res))

    def write_parameter_file(self):

        builder = PrmsBuilder(
            self.streams,
            self.cascades,
            self.modelgrid,
            self.dem.ravel(),
            hru_type=self.hru_lakeless.ravel(),
            hru_subbasin=self.hru_lakeless.ravel())

        self.parameters = builder.build()

        self.parameters.hru_lat = self.lat
        self.parameters.hru_lon = self.lon

        self.parameters.add_record_object(ParameterRecord('hru_x',
                                                          np.array(self.modelgrid.xcellcenters.ravel(),
                                                                   dtype=float).ravel(),
                                                          dimensions=[['nhru', len(self.lon)]],
                                                          datatype=2))

        self.parameters.add_record_object(ParameterRecord('hru_y',
                                                          np.array(self.modelgrid.ycellcenters.ravel(),
                                                                   dtype=float).ravel(),
                                                          dimensions=[['nhru', len(self.lat)]],
                                                          datatype=2))

        areas = np.ones_like(self.lat) * self.hru_area
        self.parameters.add_record_object(ParameterRecord('hru_area',
                                                          np.array(areas, dtype=float).ravel(),
                                                          dimensions=[['nhru', len(self.lat)]],
                                                          datatype=2))

        # self.build_lakes()
        self._build_veg_params()
        self._build_soil_params()

        [self.parameters.add_record_object(rec) for rec in self.data_params]

        [self.parameters.remove_record(rec) for rec in PRMS_NOT_REQ]

        self.parameters.write(self.parameter_file)

    def write_control_file(self):
        controlbuild = ControlFileBuilder(ControlFileDefaults())
        self.control = controlbuild.build(name='{}.control'.format(self.proj_name_res),
                                          parameter_obj=self.parameters)

        self.control.model_mode = ['PRMS']
        self.control.executable_desc = ['PRMS Model']
        self.control.executable_model = [self.cfg.prms_exe]
        self.control.cascadegw_flag = [0]
        self.control.et_module = ['potet_jh']
        self.control.precip_module = ['xyz_dist']
        self.control.temp_module = ['xyz_dist']
        self.control.solrad_module = ['ccsolrad']
        self.control.rpt_days = [7]
        self.control.snarea_curve_flag = [0]
        self.control.soilzone_aet_flag = [0]
        self.control.srunoff_module = ['srunoff_smidx']

        # 0: standard; 1: SI/metric
        units = 0
        self.control.add_record('elev_units', [units])
        self.control.add_record('precip_units', [units])
        self.control.add_record('temp_units', [units])
        self.control.add_record('runoff_units', [units])

        self.control.start_time = [int(d) for d in self.cfg.start_time.split(',')] + [0, 0, 0]

        self.control.subbasin_flag = [0]
        self.control.transp_module = ['transp_tindex']
        self.control.csv_output_file = [os.path.join(self.cfg.output_folder, 'output.csv')]
        self.control.param_file = [self.parameter_file]
        self.control.subbasin_flag = [0, ]
        self.control.parameter_check_flag = [0, ]

        self.control.add_record('end_time', [int(d) for d in self.cfg.end_time.split(',')] + [0, 0, 0])

        self.control.add_record('model_output_file', [os.path.join(self.cfg.output_folder, 'output.model')],
                                datatype=4)
        self.control.add_record('var_init_file', [os.path.join(self.cfg.output_folder, 'init.csv')],
                                datatype=4)
        self.control.add_record('data_file', [self.data_file], datatype=4)

        stat_vars = ['runoff',
                     'basin_tmin',
                     'basin_tmax',
                     'basin_ppt',
                     'basin_rain',
                     'basin_snow',
                     'basin_potsw',
                     'basin_potet',
                     'basin_net_ppt',
                     'basin_intcp_stor',
                     'basin_pweqv',
                     'basin_snowmelt',
                     'basin_snowcov',
                     'basin_sroff',
                     'basin_hortonian',
                     'basin_infil',
                     'basin_soil_moist',
                     'basin_recharge',
                     'basin_actet',
                     'basin_gwstor',
                     'basin_gwflow',
                     'basin_gwsink',
                     'basin_cfs',
                     'basin_ssflow',
                     'basin_imperv_stor',
                     'basin_lake_stor',
                     'basin_ssstor']

        self.control.add_record('statsON_OFF', values=[1], datatype=1)
        self.control.add_record('nstatVars', values=[len(stat_vars)], datatype=1)
        self.control.add_record('statVar_element', values=['1' for _ in stat_vars], datatype=4)
        self.control.add_record('statVar_names', values=stat_vars, datatype=4)
        self.control.add_record('stat_var_file', [os.path.join(self.cfg.output_folder, 'statvar.out')],
                                datatype=4)

        disp_vars = [('basin_cfs', '1'),
                     ('runoff', '1'),
                     ('basin_gwflow', '2'),
                     ('basin_sroff', '2'),
                     ('basin_ssflow', '2'),
                     ('basin_actet', '3'),
                     ('basin_potet', '3'),
                     ('basin_perv_et', '3'),
                     ('basin_pweqv', '4'),
                     ('basin_snow', '4'),
                     ('basin_snowdepth', '4'),
                     ('basin_snowmelt', '4')]

        self.control.add_record('dispVar_plot', values=[e[1] for e in disp_vars], datatype=4)
        self.control.add_record('statVar_names', values=stat_vars, datatype=4)
        self.control.add_record('dispVar_element', values=['1' for _ in disp_vars], datatype=4)

        self.control.add_record('gwr_swale_flag', [1])

        # remove gsflow control objects
        self.control.remove_record('gsflow_output_file')

        self.control.write(self.control_file)

    def write_datafile(self, units='metric'):

        self.nmonths = 12

        ghcn = self.cfg.prms_data_ghcn
        stations = self.cfg.prms_data_stations
        gages = self.cfg.prms_data_gages

        with open(stations, 'r') as js:
            sta_meta = json.load(js)

        sta_iter = sorted([(v['zone'], v) for k, v in sta_meta.items()], key=lambda x: x[0])
        tsta_elev, tsta_nuse, tsta_x, tsta_y, psta_elev = [], [], [], [], []
        for _, val in sta_iter:

            if units != 'metric':
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

        outlet_sta = self.modelgrid.intersect(self.pour_pt[0][0], self.pour_pt[0][1])
        outlet_sta = self.modelgrid.get_node([(0,) + outlet_sta])
        self.data_params.append(ParameterRecord('outlet_sta',
                                                values=[outlet_sta[0] + 1, ],
                                                dimensions=[['one', 1]],
                                                datatype=1))
        if units == 'metric':
            allrain_max = np.ones((self.nhru * self.nmonths)) * 3.3
            tmax_allrain = np.ones((self.nhru * self.nmonths)) * 3.3
            tmax_allsnow = np.ones((self.nhru * self.nmonths)) * 0.0
        else:
            allrain_max = np.ones((self.nhru * self.nmonths)) * 38.0
            tmax_allrain = np.ones((self.nhru * self.nmonths)) * 38.0
            tmax_allsnow = np.ones((self.nhru * self.nmonths)) * 32.0

        self.data_params.append(ParameterRecord('tmax_allrain_sta', allrain_max,
                                                dimensions=[['nhru', self.nhru], ['nmonths', self.nmonths]],
                                                datatype=2))

        self.data_params.append(ParameterRecord('tmax_allrain', tmax_allrain,
                                                dimensions=[['nhru', self.nhru], ['nmonths', self.nmonths]],
                                                datatype=2))

        self.data_params.append(ParameterRecord('tmax_allsnow', tmax_allsnow,
                                                dimensions=[['nhru', self.nhru], ['nmonths', self.nmonths]],
                                                datatype=2))

        self.data_params.append(ParameterRecord('snowpack_init',
                                                np.ones_like(self.ksat).ravel(),
                                                dimensions=[['nhru', self.nhru]],
                                                datatype=2))

        if not os.path.isfile(self.data_file):
            write_basin_datafile(station_json=stations,
                                 gage_json=gages,
                                 ghcn_data=ghcn,
                                 out_csv=None,
                                 data_file=self.data_file,
                                 units=units)

        self.data = PrmsData.load_from_file(self.data_file)

    def build_model_files(self):

        self._build_grid()
        self.write_datafile(units='standard')
        self.write_parameter_file()
        self.write_control_file()

    def write_raster_params(self, name, values=None):
        out_dir = os.path.join(self.cfg.raster_folder, 'resamples', self.cfg.hru_cellsize)
        if not isinstance(values, np.ndarray):
            values = self.parameters.get_values(name).reshape((self.modelgrid.nrow, self.modelgrid.ncol))
        _file = os.path.join(out_dir, '{}.tif'.format(name))

        with rasterio.open(_file, 'w', **self.raster_meta) as dst:
            dst.write(values, 1)

    def _build_grid(self):
        with fiona.open(self.cfg.study_area_path, 'r') as domain:
            geo = [f['geometry'] for f in domain][0]
            geo = shape(geo)
            self.bounds = geo.bounds

        self.modelgrid = GenerateFishnet(bbox=self.cfg.elevation,
                                         xcellsize=float(self.cfg.hru_cellsize),
                                         ycellsize=float(self.cfg.hru_cellsize))
        self.fishnet_file = os.path.join(self.cfg.hru_folder, 'fishnet.shp')
        self.modelgrid.write_shapefile(self.fishnet_file, prj=self.prj)
        self._prepare_rasters()

        x = self.modelgrid.xcellcenters.ravel()
        y = self.modelgrid.ycellcenters.ravel()
        self.nhru = (x * y).size
        self.hru_area = (float(self.cfg.hru_cellsize) ** 2) * 0.000247105
        trans = Transformer.from_proj('epsg:{}'.format(5071), 'epsg:4326', always_xy=True)
        self.lon, self.lat = trans.transform(x, y)

        self.zeros = np.zeros((self.modelgrid.nrow, self.modelgrid.ncol))
        self.nnodes = self.zeros.size

        self._build_domain_params()
        self._build_terrain_params(mode='richdem')

    def _build_terrain_params(self, mode='pygsflow'):
        """This method computes flow accumulation/direction rasters for both
        RichDEM and PyGSFLOW. RichDEM seems to fill depressions more effectively and is fast."""

        self.dem = rd.LoadGDAL(self.cfg.elevation, no_data=0.0)

        if np.any(self.dem == 0.0):
            for r in range(self.dem.shape[0]):
                d = self.dem[r, :].ravel()
                idx = np.arange(len(d))
                self.dem[r, :] = np.interp(idx, idx[d > 0.0], d[d > 0.0])

        if mode == 'richdem':
            # RichDEM flow accumulation and direction
            rd.FillDepressions(self.dem, epsilon=0.0001, in_place=True)
            self.dem = rd.rdarray(self.dem, no_data=0, dtype=float)
            rd_flow_accumulation = rd.FlowAccumulation(self.dem, method='D8')
            props = rd.FlowProportions(dem=self.dem, method='D8')

            # remap directions to pygsflow nomenclature
            dirs = np.ones_like(rd_flow_accumulation)
            for i in range(1, 9):
                dirs = np.where(props[:, :, i] == 1, np.ones_like(dirs) * i, dirs)

            rd_flow_directions = copy(dirs)
            for k, v in d8_map.items():
                rd_flow_directions[dirs == k] = v

            # manually flow corners and edges inward
            rd_flow_directions[0, 0] = 2
            rd_flow_directions[0, -1] = 8
            rd_flow_directions[-1, 0] = 128
            rd_flow_directions[-1, -1] = 32

            rd_flow_directions[0, 1:-1] = 4
            rd_flow_directions[1:-1, 0] = 1
            rd_flow_directions[1:-1, -1] = 16
            rd_flow_directions[-1, 1:-1] = 64

            self.flow_direction = rd_flow_directions
            self.flow_accumulation = rd_flow_accumulation

        elif mode == 'pygsflow':
            # pygsflow flow accumulation and direction

            fa = FlowAccumulation(self.dem,
                                  self.modelgrid.xcellcenters,
                                  self.modelgrid.ycellcenters,
                                  verbose=False)

            self.flow_direction = fa.flow_directions(dijkstra=True, breach=0.001)
            self.flow_accumulation = fa.flow_accumulation()

        else:
            raise NotImplementedError('Must choose between "pygsflow" and "richdem" for '
                                      'flow calculations')

        fa = FlowAccumulation(
            self.dem,
            self.modelgrid.xcellcenters,
            self.modelgrid.ycellcenters,
            hru_type=self.hru_lakeless,
            flow_dir_array=self.flow_direction,
            verbose=False)

        self.watershed = fa.define_watershed(self.pour_pt,
                                             self.modelgrid,
                                             fmt='xy')

        self.streams = fa.make_streams(self.flow_direction,
                                       self.flow_accumulation,
                                       threshold=100,
                                       min_stream_len=10)

        self.cascades = fa.get_cascades(streams=self.streams,
                                        pour_point=self.pour_pt, fmt='xy',
                                        modelgrid=self.modelgrid)

        self.hru_aspect = bu.d8_to_hru_aspect(self.flow_direction)
        self.hru_slope = bu.d8_to_hru_slope(self.flow_direction,
                                            self.dem,
                                            self.modelgrid.xcellcenters,
                                            self.modelgrid.ycellcenters)

    def _build_domain_params(self):
        ix = GridIntersect(self.modelgrid, method='vertex', rtree=True)
        shape_input = [('outlet', 'model_outlet_path'),
                       ('lake_id', 'lake_path'),
                       ('hru_type', 'study_area_path')]

        for param, path in shape_input:
            shp_file = getattr(self.cfg, path)
            feats = features(shp_file)
            data = copy(self.zeros)

            for i, f in enumerate(feats, start=1):
                geo = shape(f['geometry'])
                idx = ix.intersects(geo)
                for x in idx:
                    data[x[0]] = i

            outfile = os.path.join(self.cfg.hru_folder, '{}.txt'.format(param))
            if param == 'outlet':
                setattr(self, 'pour_pt', [[geo.x, geo.y]])
            if param == 'hru_type':
                erode = binary_erosion(data)
                border = erode < data
                setattr(self, 'border', border)
                lakeless = np.where(border, self.zeros + 3, data)
                setattr(self, 'hru_lakeless', lakeless)
                data = np.where(self.lake_id > 0, self.zeros + 2, data)

            setattr(self, param, data)
            np.savetxt(outfile, data, delimiter='  ')

    def _build_lakes(self):
        lakes = bu.lake_hru_id(self.lake_id)
        nlake = ParameterRecord(
            name='nlake', values=[np.unique(self.lake_id)], datatype=1, file_name=None
        )
        nlake_hrus = ParameterRecord(
            name='nlake_hrus', values=[np.count_nonzero(self.lake_id)], datatype=1, file_name=None
        )
        [self.parameters.add_record_object(l) for l in [lakes, nlake, nlake_hrus]]

    def _build_veg_params(self):
        self._prepare_lookups()
        covtype = bu.covtype(self.landfire_type, self.covtype_lut)
        covden_sum = bu.covden_sum(self.landfire_cover, self.covdensum_lut)
        covden_win = bu.covden_win(covtype.values, self.covdenwin_lut)
        rad_trncf = bu.rad_trncf(covden_win.values)
        snow_intcp = bu.snow_intcp(self.landfire_type, self.snow_intcp_lut)
        srain_intcp = bu.srain_intcp(self.landfire_type, self.srain_intcp_lut)
        wrain_intcp = bu.wrain_intcp(self.landfire_type, self.snow_intcp_lut)

        vars_ = [covtype, covden_sum, covden_win, rad_trncf, snow_intcp, srain_intcp,
                 wrain_intcp]

        for v in vars_:
            self.parameters.add_record_object(v)

        self.root_depth = bu.root_depth(self.landfire_type, self.rtdepth_lut)

    def _build_soil_params(self):
        cellsize = int(self.cfg.hru_cellsize)
        soil_type = bu.soil_type(self.clay, self.sand)

        # awc meters to inches
        self.awc = self.awc * 1000 / 25.4

        soil_moist_max = bu.soil_moist_max(self.awc, self.root_depth)
        soil_moist_init = bu.soil_moist_init(soil_moist_max.values)
        soil_rech_max = bu.soil_rech_max(self.awc, self.root_depth)
        soil_rech_init = bu.soil_rech_init(soil_rech_max.values)

        # ksat mircrometer/sec to inches/day
        self.ksat = self.ksat * 3.4 / 1000

        ssr2gw_rate = bu.ssr2gw_rate(self.ksat, self.sand, soil_moist_max.values)
        ssr2gw_sq = bu.ssr2gw_exp(self.nnodes)
        slowcoef_lin = bu.slowcoef_lin(self.ksat, self.hru_aspect.values, cellsize, cellsize)
        slowcoef_sq = bu.slowcoef_sq(self.ksat, self.hru_aspect.values, self.sand,
                                     soil_moist_max.values, cellsize, cellsize)

        # parameterize this
        sat_threshold = ParameterRecord('sat_threshold',
                                        np.ones_like(self.ksat).ravel(),
                                        dimensions=[['nhru', self.nhru]],
                                        datatype=2)

        hru_percent_imperv = bu.hru_percent_imperv(self.nlcd)
        hru_percent_imperv.values /= 100
        carea_max = bu.carea_max(self.nlcd) / 100

        vars_ = [soil_type, soil_moist_max, soil_moist_init, soil_rech_max, soil_rech_init,
                 ssr2gw_rate, ssr2gw_sq, slowcoef_lin, slowcoef_sq, hru_percent_imperv, carea_max,
                 self.hru_aspect, self.hru_slope, sat_threshold]

        for v in vars_:
            self.parameters.add_record_object(v)

    def _prepare_rasters(self):
        """gdal warp is > 10x faster for nearest, here, we resample a single raster using nearest, and use
        that raster's metadata to resample the rest with gdalwarp"""
        _int = ['landfire_cover', 'landfire_type', 'nlcd']
        _float = ['elevation', 'sand', 'clay', 'loam', 'awc', 'ksat']
        rasters = _int + _float

        first = True
        modelgrid = GenerateFishnet(self.cfg.elevation, xcellsize=1000, ycellsize=1000)

        for raster in rasters:

            in_path = getattr(self.cfg, raster)
            out_dir = os.path.join(self.cfg.raster_folder, 'resamples', self.cfg.hru_cellsize)
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)

            out_path = os.path.join(out_dir, '{}.tif'.format(raster))
            setattr(self.cfg, raster, out_path)

            txt = out_path.replace('.tif', '.txt')

            if os.path.exists(out_path) and os.path.exists(txt):
                with rasterio.open(out_path, 'r') as src:
                    a = src.read(1)
                    if raster in ['sand', 'clay', 'loam', 'ksat', 'awc']:
                        a /= 100.
                    if first:
                        self.raster_meta = src.meta
                        first = False
                setattr(self, raster, a)
                continue

            if raster in _float:
                rsample, _dtype = 'min', 'Float32'
            else:
                rsample, _dtype = 'nearest', 'UInt16'

            if first:
                robj = flopy.utils.Raster.load(in_path)

                array = robj.resample_to_grid(modelgrid, robj.bands[0], method=rsample, thread_pool=8)

                example_raster = os.path.join(out_dir, 'flopy_raster.tif')

                self.raster_meta = robj._meta
                sa = copy(self.raster_meta['transform'])
                transform = Affine(1000., sa[1], sa[2], sa[3], -1000., sa[5])

                self.raster_meta.update({'height': array.shape[0],
                                         'width': array.shape[1],
                                         'transform': transform})

                with rasterio.open(example_raster, 'w', **self.raster_meta) as ex:
                    ex.write(array, 1)
                first = False

            s = time.time()
            b = self.bounds
            warp = [self.cfg.gdal_warp_exe, in_path, out_path,
                    '-te', str(b[0]), str(b[1]), str(b[2] + self.res), str(b[3]),
                    '-ts', str(array.shape[1]), str(array.shape[0]),
                    '-multi', '-wo', '-wo NUM_THREADS=8',
                    '-ot', _dtype, '-r', rsample,
                    '-dstnodata', '0', '-srcnodata', '0', '-overwrite']

            call(warp, stdout=open(os.devnull, 'wb'))
            print('gdalwarp {} on {}: {} sec\n'.format(rsample, raster, time.time() - s))

            with rasterio.open(out_path, 'r') as src:
                a = src.read(1)
                if raster in ['sand', 'clay', 'loam', 'ksat', 'awc']:
                    a /= 100.
                if first:
                    self.raster_meta = src.raster_meta
                    first = False

            setattr(self, raster, a)
            np.savetxt(txt, a)

    def _prepare_lookups(self):
        req_remaps = ['covtype.rmp', 'covdenwin.rmp', 'srain_intcp.rmp',
                      'snow_intcp.rmp', 'rtdepth.rmp', 'covdensum.rmp',
                      'wrain_intcp.rmp']

        for rmp in req_remaps:
            rmp_file = os.path.join(self.cfg.remap_folder, rmp)
            lut = bu.build_lut(rmp_file)
            _name = '{}_lut'.format(rmp.split('.')[0])
            setattr(self, _name, lut)


class MontanaPrmsModel:

    def __init__(self, control_file, parameter_file, data_file):
        self.control_file = control_file
        self.parameter_file = parameter_file
        self.data_file = data_file
        self.control = ControlFile.load_from_file(control_file)
        self.parameters = PrmsParameters.load_from_file(parameter_file)
        self.data = PrmsData.load_from_file(data_file)
        self.statvar = None

    def run_model(self, stdout=None):

        for obj_, var_ in [(self.control, 'control'),
                           (self.parameters, 'parameters'),
                           (self.data, 'data')]:
            if not obj_:
                raise TypeError('{} is not set, run "write_{}_file()"'.format(var_, var_))

        buff = []
        normal_msg = 'normal termination'
        report, silent = True, False

        argv = [self.control.get_values('executable_model')[0], self.control_file]
        model_ws = os.path.dirname(self.control_file)
        proc = Popen(argv, stdout=PIPE, stderr=STDOUT, cwd=model_ws)

        while True:
            line = proc.stdout.readline()
            c = line.decode('utf-8')
            if c != '':
                for msg in normal_msg:
                    if msg in c.lower():
                        success = True
                        break
                c = c.rstrip('\r\n')
                if not silent:
                    print('{}'.format(c))
                if report:
                    buff.append(c)
            else:
                break
        if stdout:
            with open(stdout, 'w') as fp:
                if report:
                    for line in buff:
                        fp.write(line + '\n')

        return success, buff

    def get_statvar(self):
        self.statvar = StatVar.load_from_control_object(self.control)
        df = self.statvar.stat_df.drop(columns=['Hour', 'Minute', 'Second'])
        return df


def features(shp):
    with fiona.open(shp, 'r') as src:
        return [f for f in src]


def plot_stats(stats):
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(stats.Date, stats.basin_cfs_1, color='r', linewidth=2.2, label="simulated")
    ax.plot(stats.Date, stats.runoff_1, color='b', linewidth=1.5, label="measured")
    ax.legend(bbox_to_anchor=(0.25, 0.65))
    ax.set_xlabel("Date")
    ax.set_ylabel("Streamflow, in cfs")
    # ax.set_ylim([0, 2000])
    # plt.savefig('/home/dgketchum/Downloads/hydrograph.png')
    plt.show()
    plt.close()


if __name__ == '__main__':
    matplotlib.use('TkAgg')

    conf = './model_files/uyws_parameters.ini'
    stdout_ = '/media/research/IrrigationGIS/Montana/upper_yellowstone/gsflow_prep/uyws_carter_1000/out.txt'
    prms_build = StandardPrmsBuild(conf)
    prms_build.build_model_files()

    prms = MontanaPrmsModel(prms_build.control_file,
                            prms_build.parameter_file,
                            prms_build.data_file)
    prms.run_model(stdout_)

    stats = prms.get_statvar()
    plot_stats(stats)
    pass

# ========================= EOF ====================================================================
