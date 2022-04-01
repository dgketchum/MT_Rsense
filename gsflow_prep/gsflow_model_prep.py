import os
from copy import copy
from subprocess import call

import numpy as np
import rasterio
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

# 234
# 105
# 876

d8_map = {5: 1, 6: 2, 7: 4, 8: 8, 1: 16, 2: 32, 3: 64, 4: 128}


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
        self.modelgrid = GenerateFishnet(self.cfg.elevation, float(self.cfg.hru_cellsize),
                                         float(self.cfg.hru_cellsize), buffer=10)
        self.modelgrid.write_shapefile(self.cfg.hru_fishnet_path, prj=self.prj)

        self.zeros = np.zeros((self.modelgrid.nrow, self.modelgrid.ncol))

        self.prepare_rasters()
        self.build_vector_hru_attrs()
        self.build_raster_hru_attrs()

    def prepare_rasters(self):
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
                continue

            if raster in _float:
                rsample, _dtype = 'min', 'Float32'
            else:
                rsample, _dtype = 'mode', 'UInt16'

            warp = [_warp, in_path, out_path, '-ts', str(self.modelgrid.ncol),
                    str(self.modelgrid.nrow), '-ot', 'Float32', '-r', rsample,
                    '-dstnodata', '0', '-srcnodata', '0', '-overwrite']

            call(warp, stdout=open(os.devnull, 'wb'))

            with rasterio.open(out_path, 'r') as src:
                a = src.read(1)
            np.savetxt(txt, a)

    def build_raster_hru_attrs(self):

        dem = rd.LoadGDAL(resamp, no_data=0.0)
        rd.FillDepressions(dem, epsilon=0.0001, in_place=True)
        accum_d8 = rd.FlowAccumulation(dem, method='D8')
        props = rd.FlowProportions(dem=dem, method='D8')

        # remap directions to pygsflow nomenclature
        dirs = self.zeros + 1
        for i in range(1, 9):
            dirs = np.where(props[:, :, i] == 1, np.ones_like(dirs) * i, dirs)

        flow_directions = copy(dirs)
        for k, v in d8_map.items():
            flow_directions[dirs == k] = v

        fa = FlowAccumulation(
            np.array(dem),
            self.modelgrid.xcellcenters,
            self.modelgrid.ycellcenters,
            hru_type=self.hru_lakeless,
            flow_dir_array=flow_directions,
            verbose=True)

        strm_obj = fa.make_streams(flow_directions, np.array(accum_d8), threshold=100, min_stream_len=10)

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(1, 1, 1, aspect="equal")

        pmv = flopy.plot.PlotMapView(modelgrid=self.modelgrid, ax=ax)
        # plot the watershed boundary on top
        pc = pmv.plot_array(strm_obj.iseg, masked_values=[0, ])

        plt.colorbar(pc, shrink=0.7)
        plt.title("Upper Yellowstone Stream Segments")
        plt.show()

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
                setattr(self, 'hru_lakeless', data)
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
