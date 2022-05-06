import os
from copy import copy, deepcopy
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

register_matplotlib_converters()
pd.options.mode.chained_assignment = None

# RichDEM flow-direction coordinate system:
# 234
# 105
# 876

d8_map = {5: 1, 6: 2, 7: 4, 8: 8, 1: 16, 2: 32, 3: 64, 4: 128}


class StandardPrmsBuild:

    def __init__(self, config):

        self.cfg = PRMSConfig(config)
        self.res = float(self.cfg.hru_cellsize)
        self.proj_name_res = '{}_{}'.format(self.cfg.project_name,
                                            self.cfg.hru_cellsize)

        for folder in ['hru_folder', 'parameter_folder', 'control_folder', 'data_folder', 'output_folder']:
            folder_path = os.path.join(self.cfg.project_folder, self.proj_name_res, getattr(self.cfg, folder))
            setattr(self.cfg, folder, folder_path)
            if not os.path.isdir(folder_path):
                os.makedirs(folder_path, exist_ok=True)

        self.parameters = None
        self.control = None
        self.data = None
        self.data_params = []
        self.control_records = []

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

    def build_parameters(self):

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

        outlet_sta = self.modelgrid.intersect(self.pour_pt_coords[0][0], self.pour_pt_coords[0][1])
        outlet_sta = self.modelgrid.get_node([(0,) + outlet_sta])
        self.data_params.append(ParameterRecord('outlet_sta',
                                                values=[outlet_sta[0] + 1, ],
                                                dimensions=[['one', 1]],
                                                datatype=1))

        # self.build_lakes()
        self._build_veg_params()
        self._build_soil_params()

        [self.parameters.remove_record(rec) for rec in PRMS_NOT_REQ]

    def build_controls(self):
        controlbuild = ControlFileBuilder(ControlFileDefaults())
        self.control = controlbuild.build(name='{}.control'.format(self.proj_name_res),
                                          parameter_obj=self.parameters)

        self.control.model_mode = ['PRMS']
        self.control.executable_desc = ['PRMS Model']
        self.control.executable_model = [self.cfg.prms_exe]
        self.control.cascadegw_flag = [0]

        # self.control.rpt_days = [7]
        self.control.snarea_curve_flag = [0]
        self.control.soilzone_aet_flag = [1]
        self.control.srunoff_module = ['srunoff_smidx']

        self.control.start_time = [int(d) for d in self.cfg.start_time.split('-')] + [0, 0, 0]

        self.control.subbasin_flag = [0]
        self.control.transp_module = ['transp_tindex']
        self.control.csv_output_file = [os.path.join(self.cfg.output_folder, 'output.csv')]
        self.control.param_file = [self.parameter_file]
        self.control.subbasin_flag = [0, ]
        self.control.parameter_check_flag = [0, ]

        self.control.add_record('end_time', [int(d) for d in self.cfg.end_time.split('-')] + [0, 0, 0])

        self.control.add_record('model_output_file',
                                [os.path.join(self.cfg.output_folder, 'output.model')],
                                datatype=4)

        # self.control.add_record('var_init_file',
        #                         [os.path.join(self.cfg.output_folder, 'init.csv')],
        #                         datatype=4)

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
                     'basin_cms',
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

        disp_vars = [('basin_cms', '1'),
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

    def write_raster_params(self, name, values=None, out_file=None):

        if not isinstance(values, np.ndarray):
            values = self.parameters.get_values(name).reshape((self.modelgrid.nrow, self.modelgrid.ncol))

        if out_file is None:
            _file = os.path.join(self.cfg.hru_folder, '{}.tif'.format(name))
        else:
            _file = out_file

        with rasterio.open(_file, 'w', **self.raster_meta) as dst:
            dst.write(values, 1)

    def _build_grid(self):
        with fiona.open(self.cfg.study_area_path, 'r') as domain:
            geo = [f['geometry'] for f in domain][0]
            geo = shape(geo)
            self.study_area = geo
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
        self.hru_area = (float(self.cfg.hru_cellsize) ** 2) * 0.000247105  # acres
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

        self.watershed = self._watershed_recursion(fa)

        self.streams = fa.make_streams(self.flow_direction,
                                       self.flow_accumulation,
                                       threshold=4,
                                       min_stream_len=2)

        self.cascades = fa.get_cascades(streams=self.streams,
                                        pour_point=self.pour_pt_rowcol, fmt='rowcol',
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

            if param == 'outlet':
                setattr(self, 'pour_pt_rowcol', [[x[0][0], x[0][1]]])
                setattr(self, 'pour_pt_coords', [[geo.x, geo.y]])
            if param == 'hru_type':
                erode = binary_erosion(data)
                border = erode < data
                setattr(self, 'border', border)
                lakeless = np.where(border, self.zeros + 3, data)
                setattr(self, 'hru_lakeless', lakeless)
                data = np.where(self.lake_id > 0, self.zeros + 2, data)

            setattr(self, param, data)

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

        # awc meters to inches; to fraction
        self.awc = self.awc / 0.0254
        self.awc /= self.root_depth.reshape(self.awc.shape[0], self.awc.shape[1])
        self.awc[np.isinf(self.awc)] = 0.0001
        self.awc[np.isnan(self.awc)] = 0.0001

        soil_moist_max = bu.soil_moist_max(self.awc, self.root_depth)
        soil_moist_init = bu.soil_moist_init(soil_moist_max.values, factor=0.1)
        soil_rech_max = bu.soil_rech_max(self.awc, self.root_depth)
        soil_rech_init = bu.soil_rech_init(soil_rech_max.values, factor=0.1)

        # ksat mircrometer/sec to inches/day
        self.ksat = self.ksat * 86400. / 25400.
        self.ksat = np.where(self.ksat > 85.0, np.ones_like(self.ksat) * 85.0, self.ksat)

        ssr2gw_rate = bu.ssr2gw_rate(self.ksat, self.sand, soil_moist_max.values)
        ssr2gw_sq = bu.ssr2gw_exp(self.nnodes)
        slowcoef_lin = bu.slowcoef_lin(self.ksat, self.hru_aspect.values, cellsize, cellsize)
        slowcoef_sq = bu.slowcoef_sq(self.ksat, self.hru_aspect.values, self.sand,
                                     soil_moist_max.values, cellsize, cellsize)

        # parameterize this
        sat_threshold = ParameterRecord('sat_threshold',
                                        self.awc.ravel() * 3.,
                                        dimensions=[['nhru', self.nhru]],
                                        datatype=2)

        gwstor_init = ParameterRecord('gwstor_init',
                                      [0.5],
                                      dimensions=[['one', 1]],
                                      datatype=2)

        gwstor_min = ParameterRecord('gwstor_min',
                                     [0.1],
                                     dimensions=[['one', 1]],
                                     datatype=2)

        hru_percent_imperv = bu.hru_percent_imperv(self.nlcd)
        hru_percent_imperv.values /= 100
        carea_max = bu.carea_max(1 - self.nlcd / 100)

        vars_ = [soil_type, soil_moist_max, soil_moist_init, soil_rech_max, soil_rech_init,
                 ssr2gw_rate, ssr2gw_sq, slowcoef_lin, slowcoef_sq, hru_percent_imperv, carea_max,
                 self.hru_aspect, self.hru_slope, sat_threshold, gwstor_init, gwstor_min]

        for v in vars_:
            self.parameters.add_record_object(v)

    def _prepare_rasters(self):
        """gdal warp is > 10x faster for nearest, here, we resample a single raster using nearest, and use
        that raster's metadata to resample the rest with gdalwarp"""
        _int = ['landfire_cover', 'landfire_type', 'nlcd']
        _float = ['elevation', 'sand', 'clay', 'loam', 'awc', 'ksat']
        rasters = _int + _float

        first = True

        for raster in rasters:

            in_path = getattr(self.cfg, raster)

            out_path = os.path.join(self.cfg.hru_folder, '{}.tif'.format(raster))
            setattr(self.cfg, raster, out_path)

            if os.path.exists(out_path):
                with rasterio.open(out_path, 'r') as src:
                    a = src.read(1)
                    if raster in ['sand', 'clay', 'loam', 'ksat', 'awc']:
                        a /= 10000.
                    if first:
                        self.raster_meta = src.meta
                setattr(self, raster, a)
                continue

            if raster in _float:
                rsample, _dtype = 'min', 'Float32'
            else:
                rsample, _dtype = 'nearest', 'UInt16'

            if first:
                robj = flopy.utils.Raster.load(in_path)

                array = robj.resample_to_grid(self.modelgrid, robj.bands[0], method=rsample, thread_pool=8)

                example_raster = os.path.join(self.cfg.hru_folder, 'flopy_raster.tif')

                self.raster_meta = robj._meta
                sa = copy(self.raster_meta['transform'])
                transform = Affine(self.res, sa[1], sa[2], sa[3], -self.res, sa[5])

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
                    a /= 10000.
                if first:
                    self.raster_meta = src.raster_meta
                    first = False

            setattr(self, raster, a)

    def _prepare_lookups(self):
        req_remaps = ['covtype.rmp', 'covdenwin.rmp', 'srain_intcp.rmp',
                      'snow_intcp.rmp', 'rtdepth.rmp', 'covdensum.rmp',
                      'wrain_intcp.rmp']

        for rmp in req_remaps:
            rmp_file = os.path.join(self.cfg.remap_folder, rmp)
            lut = bu.build_lut(rmp_file)
            _name = '{}_lut'.format(rmp.split('.')[0])
            setattr(self, _name, lut)

    def _watershed_recursion(self, floacc):
        """Watersheds tend to be poorly built from flow accumulation, try different methods,
        this approach looks around for a better spot for the outlet"""

        def check_size(ws):
            shed_size_est = np.count_nonzero(self.hru_type >= 1)
            shed_size_fa = np.count_nonzero(ws)
            ws_frac = shed_size_fa / shed_size_est
            return ws_frac

        dir_map = {0: [-1, -1],
                   1: [-1, 0],
                   2: [-1, 1],
                   3: [0, -1],
                   4: [0, 0],
                   5: [0, 1],
                   6: [1, -1],
                   7: [1, 0],
                   8: [1, 1]}

        flo = np.array(self.flow_accumulation, dtype=int)
        dem = np.array(self.dem)
        size_frac = 0.0
        rc = self.pour_pt_rowcol[0]
        while size_frac < 0.7:
            fa = deepcopy(floacc)
            shed = fa.define_watershed([rc], self.modelgrid, fmt='rowcol')
            size_frac = check_size(shed)
            hood = flo[rc[0] - 1:rc[0] + 2, rc[1] - 1:rc[1] + 2]

            if hood.size == 0:
                flo_ = np.pad(flo, 1, 'constant', constant_values=0)
                rc_ = [rc[0] + 1, rc[1] + 1]
                hood = flo_[rc_[0] - 1:rc_[0] + 2, rc_[1] - 1:rc_[1] + 2]

            idx_add = dir_map[np.argmax(hood)]
            rc = [rc[0] + idx_add[0], rc[1] + idx_add[1]]

        self.pour_pt_rowcol = [rc]
        self.pour_pt_coords = [[self.modelgrid.xcellcenters[rc[0], rc[1]],
                                self.modelgrid.ycellcenters[rc[0], rc[1]]]]
        return shed


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
        df = self.statvar.stat_df
        df.drop(columns=['Hour', 'Minute', 'Second'], inplace=True)

        if self.control.get_record('runoff_units').values[0] == 0:
            raise AttributeError('Set runoff units to metric')

        # cms to m3 per day
        df['obs_q_vol_m3'] = 60 * 60 * 24 * df['runoff_1'] / 1e6
        df['pred_q_vol_m3'] = 60 * 60 * 24 * df['basin_cms_1'] / 1e6

        # ppt in inches, hru_area in acres
        hru_area = self.parameters.get_values('hru_area')[0]
        hru_active = np.count_nonzero(self.parameters.get_values('hru_type'))
        basin_area = hru_active * hru_area * 43560.
        ppt_meters = df['basin_ppt_1'] / 39.3701
        df['ppt_vol_m3'] = basin_area * ppt_meters / 1e6

        s, e = self.control.get_values('start_time'), self.control.get_values('end_time')
        try:
            df.index = pd.date_range('{}-{}-{}'.format(s[0], s[1], s[2]),
                                     '{}-{}-{}'.format(e[0], e[1], e[2]), freq='D')
            df.drop(columns=['Year', 'Month', 'Day'], inplace=True)
        except ValueError:
            pass
        return self.statvar.stat_df


def features(shp):
    with fiona.open(shp, 'r') as src:
        return [f for f in src]


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
