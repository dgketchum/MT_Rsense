import os
from copy import copy
from subprocess import call
import time

import numpy as np
from pyproj import Transformer
import rasterio
import fiona
from shapely.geometry import shape
import matplotlib.pyplot as plt
import richdem as rd

import flopy
import gsflow
from gsflow.builder import GenerateFishnet, FlowAccumulation, PrmsBuilder
from gsflow.builder import builder_utils as bu
from flopy.utils import GridIntersect

from model_config import PRMSConfig

_warp = '/home/dgketchum/miniconda3/envs/opnt/bin/gdalwarp'

# 234
# 105
# 876

d8_map = {5: 1, 6: 2, 7: 4, 8: 8, 1: 16, 2: 32, 3: 64, 4: 128}


class PRMSModel:

    def __init__(self, config):

        self.nnodes = None
        self.root_depth = None
        self.cfg = PRMSConfig(config)

        self.hru_aspect = None
        self.hru_slope = None
        self.modelgrid = None
        self.hru_params = None

        self.hru_type = None
        self.dem = None
        self.lat, self.lon = None, None
        self.streams = None
        self.cascades = None

        self.parameters = None

        self.zeros = None

        with fiona.open(self.cfg.study_area_path, 'r') as src:
            self.meta = src.meta
            self.basin_geo = [shape(f['geometry']) for f in src][0]
            self.prj = self.cfg.study_area_path.replace('.shp', '.prj')

    def build_default_model_files(self):

        prmsbuild = PrmsBuilder(
            self.streams,
            self.cascades,
            self.modelgrid,
            self.dem.ravel(),
            hru_type=self.hru_lakeless.ravel(),
            hru_subbasin=self.hru_lakeless.ravel())

        self.parameters = prmsbuild.build()
        self.parameters.hru_lat = self.lat
        self.parameters.hru_lon = self.lon
        self.build_veg_params()
        self.build_soil_params()

        param_file = os.path.join(self.cfg.prms_parameter_folder,
                                  '{}_{}.params'.format(self.cfg.proj_name,
                                                        self.cfg.hru_cellsize))
        self.parameters.write(param_file)

    def build_grid(self):
        self.modelgrid = GenerateFishnet(self.cfg.elevation, float(self.cfg.hru_cellsize),
                                         float(self.cfg.hru_cellsize), buffer=10)
        self.modelgrid.write_shapefile(self.cfg.hru_fishnet_path, prj=self.prj)
        x = self.modelgrid.xcellcenters.ravel()
        y = self.modelgrid.ycellcenters.ravel()
        trans = Transformer.from_proj('epsg:{}'.format(5071), 'epsg:4326', always_xy=True)
        self.lon, self.lat = trans.transform(x, y)

        # error on property call nnodes = nlayer * ncol * nrow in StructuredGrid
        self.zeros = np.zeros((self.modelgrid.nrow, self.modelgrid.ncol))
        self.nnodes = self.zeros.size

        self._prepare_rasters()
        self.build_domain_params()
        self.build_terrain_params()

    def build_terrain_params(self):

        self.dem = rd.LoadGDAL(self.cfg.elevation, no_data=0.0)
        rd.FillDepressions(self.dem, epsilon=0.0001, in_place=True)
        accum_d8 = rd.FlowAccumulation(self.dem, method='D8')
        props = rd.FlowProportions(dem=self.dem, method='D8')

        # remap directions to pygsflow nomenclature
        dirs = self.zeros + 1
        for i in range(1, 9):
            dirs = np.where(props[:, :, i] == 1, np.ones_like(dirs) * i, dirs)

        flow_directions = copy(dirs)
        for k, v in d8_map.items():
            flow_directions[dirs == k] = v

        fa = FlowAccumulation(
            np.array(self.dem),
            self.modelgrid.xcellcenters,
            self.modelgrid.ycellcenters,
            hru_type=self.hru_lakeless,
            flow_dir_array=flow_directions,
            verbose=True)

        self.streams = fa.make_streams(flow_directions, np.array(accum_d8), threshold=100, min_stream_len=10)
        self.cascades = fa.get_cascades(streams=self.streams, pour_point=self.cfg.model_outlet_path, fmt='shp',
                                        modelgrid=self.modelgrid)

        self.hru_aspect = bu.d8_to_hru_aspect(flow_directions)
        self.hru_slope = bu.d8_to_hru_slope(
            flow_directions,
            self.dem,
            self.modelgrid.xcellcenters,
            self.modelgrid.ycellcenters
        )

    def build_domain_params(self):

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

            outfile = os.path.join(self.cfg.parameter_folder, '{}.txt'.format(param))
            if param == 'outlet':
                setattr(self, 'pour_pt_rowcol', idx[0][0])
            if param == 'hru_type':
                setattr(self, 'hru_lakeless', data)
                data = np.where(self.lake_id > 0, self.zeros + 2, data)

            setattr(self, param, data)
            np.savetxt(outfile, data, delimiter="  ")

    def build_veg_params(self):
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

    def build_soil_params(self):

        cellsize = int(self.cfg.hru_cellsize)
        soil_type = bu.soil_type(self.clay, self.sand)
        soil_moist_max = bu.soil_moist_max(self.awc, self.root_depth)
        soil_moist_init = bu.soil_moist_init(soil_moist_max.values)
        soil_rech_max = bu.soil_rech_max(self.awc, self.root_depth)
        soil_rech_init = bu.soil_rech_init(soil_rech_max.values)
        ssr2gw_rate = bu.ssr2gw_rate(self.ksat, self.sand, soil_moist_max.values)
        ssr2gw_sq = bu.ssr2gw_exp(self.nnodes)
        slowcoef_lin = bu.slowcoef_lin(self.ksat, self.hru_aspect.values, cellsize, cellsize)
        slowcoef_sq = bu.slowcoef_sq(self.ksat, self.hru_aspect.values, self.sand,
                                     soil_moist_max.values, cellsize, cellsize)

        hru_percent_imperv = bu.hru_percent_imperv(self.nlcd)
        hru_percent_imperv.values /= 100
        carea_max = bu.carea_max(self.nlcd)

        vars_ = [soil_type, soil_moist_max, soil_moist_init, soil_rech_max, soil_rech_init,
                 ssr2gw_rate, ssr2gw_sq, slowcoef_lin, slowcoef_sq, hru_percent_imperv, carea_max]

        for v in vars_:
            self.parameters.add_record_object(v)

    def _prepare_rasters(self):
        _float = ['sand', 'clay', 'loam', 'awc', 'ksat', 'elevation']
        _int = ['landfire_cover', 'landfire_type', 'nlcd']
        rasters = _float + _int

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
                setattr(self, raster, a)
                continue

            if raster in _float:
                rsample, _dtype = 'min', 'Float32'
            else:
                rsample, _dtype = 'mode', 'UInt16'

            warp = [_warp, in_path, out_path, '-ts', str(self.modelgrid.ncol),
                    str(self.modelgrid.nrow), '-ot', _dtype, '-r', rsample,
                    '-dstnodata', '0', '-srcnodata', '0', '-overwrite']

            call(warp, stdout=open(os.devnull, 'wb'))

            with rasterio.open(out_path, 'r') as src:
                a = src.read(1)

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


def features(shp):
    with fiona.open(shp, 'r') as src:
        return [f for f in src]


def modify_params(control, model_ws):
    control = gsflow.GsflowModel.load_from_file(control, model_ws=model_ws, prms_only=True)
    hru_type = control.prms.parameters.get_values('hru_type')
    lake_id_ = control.prms.parameters.get_values('lake_hru_id')
    lake_id_ = np.where(hru_type == 2, np.ones_like(hru_type), np.zeros_like(hru_type))
    control.prms.parameters.set_values('lake_hru_id', lake_id_)
    control.prms.parameters.write()


if __name__ == '__main__':
    conf = './model_files/uyws_parameters.ini'
    model = PRMSModel(conf)
    model.build_grid()
    model.build_default_model_files()
# ========================= EOF ====================================================================
