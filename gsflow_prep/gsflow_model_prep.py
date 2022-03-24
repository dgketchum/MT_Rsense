import os
from copy import copy

import numpy as np
import fiona
from shapely.geometry import shape
import flopy
import gsflow
from gsflow.builder import GenerateFishnet, FlowAccumulation
from flopy.utils import GridIntersect

from model_config import PRMSConfig


class PRMSModel:

    def __init__(self, config):
        self.cfg = PRMSConfig(config)
        with fiona.open(self.cfg.study_area_path, 'r') as src:
            self.meta = src.meta
            self.basin_geo = [shape(f['geometry']) for f in src][0]
            self.prj = self.cfg.study_area_path.replace('.shp', '.prj')
            self.modelgrid = None
            self.hru_params = None

    def build_grid(self):
        self.modelgrid = GenerateFishnet(self.cfg.dem_orig_path, 2100, 2100, buffer=10)
        self.build_raster_hru_attrs()
        # self.build_vector_hru_attrs()

        fa = FlowAccumulation(
            flopy.utils.Raster.load(self.dem_resamp_path),
            self.modelgrid.xcellcenters,
            self.modelgrid.ycellcenters,
            hru_type=np.ones((self.modelgrid.nrow, self.modelgrid.ncol), dtype=int),
            verbose=False
        )
        flow_dir = fa.flow_directions(dijkstra=False)

    def build_raster_hru_attrs(self):
        # robj = flopy.utils.Raster.load(self.cfg.dem_orig_path)
        # dem_data = robj.resample_to_grid(
        #     self.modelgrid,
        #     robj.bands[0],
        #     method="bilinear",
        #     multithread=True,
        #     thread_pool=12
        # )
        # np.savetxt(self.dem_resamp_path, dem_data, delimiter="  ")

        remaps = [('DEM_ASPECT', 'aspect.rmp'),
                  ('COVDEN_SUM', 'covdensum.rmp'),
                  ('COVDEN_WIN', 'covdenwin.rmp'),
                  ('COV_TYPE', 'covtype.rmp'),
                  ('SNOW_INTCP', 'covtype_to_snow_intcp.rmp'),
                  ('SRAIN_INTC', 'covtype_to_srain_intcp.rmp'),
                  ('WRAIN_INTC', 'covtype_to_wrain_intcp.rmp'),
                  ('SOIL_RT_MX', 'rtdepth.rmp'),
                  ('SNOW_INTCP', 'snow_intcp.rmp'),
                  ('SRAIN_INTC', 'srain_intcp.rmp'),
                  ('temp_adj_x10.rmp'),
                  ('WRAIN_INTC', 'wrain_intcp.rmp')]


    def build_vector_hru_attrs(self):
        zeros = np.zeros((self.modelgrid.nrow, self.modelgrid.ncol))
        ix = GridIntersect(self.modelgrid, method='vertex', rtree=True)
        shape_input = [('hru_type', 'study_area_path'),
                       ('outlet', 'model_outlet_path'),
                       ('lake_id', 'lake_path')]

        for param, path in shape_input:
            shp_file = getattr(self.cfg, path)
            feats = features(shp_file)
            data = copy(zeros)
            for i, f in enumerate(feats, start=1):
                geo = shape(f['geometry'])
                idx = ix.intersects(geo)
                for x in idx:
                    data[x[0]] = i
            outfile = os.path.join(self.cfg.parameter_folder, '{}.txt'.format(param))
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
