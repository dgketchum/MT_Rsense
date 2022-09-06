import os
import fiona


def write_features(shp, out_dir):
    with fiona.open(shp, 'r') as src:
        meta = src.meta
        for f in src:
            name = f['properties']['BASINNUM']
            file_ = os.path.join(out_dir, '{}.shp'.format(name))
            with fiona.open(file_, 'w', **meta) as dst:
                dst.write(f)
                print(file_)


if __name__ == '__main__':
    d = '/media/research/IrrigationGIS/Montana'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS/Montana'

    basins = os.path.join(d, 'AdminBasinsShapefile', 'BasinBoundaries_USGSALB.shp')
    segments = os.path.join(d, 'water_availability', 'dnrc_basin_prms_segments_basinnum.shp')
    basin_wr_pod = os.path.join(d, 'water_availability', 'basin_wr_gdb')
    basin_features = os.path.join(d, 'water_availability', 'dnrc_basin_features')
    gdb = os.path.join(d, 'water_rights', 'wr_gdb', 'wrpod.shp')
    write_features(basins, basin_features)
    # clip_data_to_basin(gdb, basins, basin_wr_pod)
# ========================= EOF ====================================================================
