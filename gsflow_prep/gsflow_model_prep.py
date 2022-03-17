import os

import gsflow
import numpy as np


def build_control(control, model_ws):
    control = gsflow.GsflowModel.load_from_file(control, model_ws=model_ws)
    pass


if __name__ == '__main__':
    d = '/media/research/IrrigationGIS/software/prms5.2.0_linux/projects'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS/software/prms5.2.0_linux/projects'

    model_workspace = os.path.join(d, 'uyws_carter')
    control_file = os.path.join(model_workspace, 'control', 'uyws.control')
    build_control(control_file, model_workspace)
# ========================= EOF ====================================================================
