import os
from copy import copy
from subprocess import check_call

import numpy as np
import shapefile
import fiona
from shapely.geometry import shape
import matplotlib.pyplot as plt
import richdem as rd

import flopy
import gsflow
from gsflow.builder import GenerateFishnet, FlowAccumulation
from flopy.utils import GridIntersect

from model_config import PRMSConfig

_warp = '/home/dgketchum/miniconda3/envs/opnt/bin/gdalwarp'


class PRMSModel:

    def __init__(self, config):
        self.hru_type = None
        self.cfg = PRMSConfig(config)
        self.zeros = None
        with fiona.open(self.cfg.study_area_path, 'r') as src:
            self.meta = src.meta
            self.basin_geo = [shape(f['geometry']) for f in src][0]
            self.prj = self.cfg.study_area_path.replace('.shp', '.prj')
            self.modelgrid = None
            self.hru_params = None

    def build_grid(self):
        self.modelgrid = GenerateFishnet(self.cfg.dem_orig_path, float(self.cfg.hru_cellsize),
                                         float(self.cfg.hru_cellsize), buffer=10)
        self.modelgrid.write_shapefile(self.cfg.hru_fishnet_path, prj=self.prj)

        self.zeros = np.zeros((self.modelgrid.nrow, self.modelgrid.ncol))

        self.build_raster_hru_attrs()
        self.build_vector_hru_attrs()

    def build_raster_hru_attrs(self):
        resamp = self.cfg.dem_resamp_path.replace('.txt', '.tif')
        warp = [_warp, self.cfg.dem_orig_path, resamp, '-ts', str(self.modelgrid.ncol),
                str(self.modelgrid.nrow), '-ot', 'Float32', '-r', 'min',
                '-dstnodata', '0', '-srcnodata', '0', '-overwrite']
        check_call(warp)
        dem = rd.LoadGDAL(resamp, no_data=0.0)
        rd.FillDepressions(dem, epsilon=0.0001, in_place=True)
        accum_d8 = rd.FlowAccumulation(dem, method='D8')
        d8_fig = rd.rdShow(accum_d8, figsize=(8, 5.5), axes=False, cmap='jet', show=True)

    def build_vector_hru_attrs(self):

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
            if param == 'hru_type':
                data = np.where(self.lake_id > 0, self.zeros + 2, data)
            setattr(self, param, data)
            np.savetxt(outfile, data, delimiter="  ")


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
# ========================= EOF ====================================================================
