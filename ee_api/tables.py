import os

from numpy import nan
from pandas import read_csv, concat
from shapely.geometry import Polygon
from geopandas import GeoDataFrame


def concatenate_field_water_balance(extracts, out_filename, glob='glob', join_key='FID',
                                    template_geometry=None):
    l = [os.path.join(extracts, x) for x in os.listdir(extracts) if glob in x]
    l.reverse()

    first = True
    for csv in l:
        splt = os.path.basename(csv).split('_')
        y, m = int(splt[-2]), int(splt[-1].split('.')[0])
        if first:
            df = read_csv(csv, index_col=join_key)
            df.drop(columns=['irr'], inplace=True)
            df.columns = ['aet' if 'swb' in col else col for col in list(df.columns)]
            df.columns = ['{}_{}_{}'.format(col, y, m) for col in list(df.columns)]
            df[join_key] = df.index
            first = False
        else:
            c = read_csv(csv, index_col=join_key)
            c.drop(columns=['irr'], inplace=True)
            c.columns = ['aet' if 'swb' in col else col for col in list(c.columns)]
            c.columns = ['{}_{}_{}'.format(col, y, m) for col in list(c.columns)]
            df = concat([df, c], axis=1)

    df.to_csv(out_filename)

    if template_geometry:
        t_gdf = GeoDataFrame.from_file(template_geometry).to_crs('epsg:5071')
        geo = t_gdf['geometry']
        df = concat([df, t_gdf])
        gpd = GeoDataFrame(df, crs='epsg:5071', geometry=geo)
        gpd.to_file(out_filename.replace('.csv', '.shp'))


def to_polygon(j):
    if not isinstance(j, list):
        return nan
    try:
        return Polygon(j[0])
    except ValueError:
        return nan
    except TypeError:
        return nan
    except AssertionError:
        return nan


if __name__ == '__main__':
    _bucket = 'gs://wudr'
    south = False
    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'
    _tables = 'Montana/st_mary/ssebop_data/depth'
    tables = os.path.join(root, _tables, '2FEB2022')
    o_file = os.path.join(root, _tables, 'fields_waterbal_2FEB2022.csv')
    _glob = '2FEB2022'
    template_shape = os.path.join(root, 'Montana/st_mary/SMB Study Area Irrigation/smb_aea_fid.shp')
    concatenate_field_water_balance(tables, out_filename=o_file, glob=_glob, template_geometry=template_shape)

# ========================= EOF ====================================================================
