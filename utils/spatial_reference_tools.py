# ===============================================================================
# Copyright 2017 dgketchum
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===============================================================================
import os
import osr, ogr, gdal


def shp_spatial_reference(shapefile):
    """Get spatial reference from an ESRI .shp shapefile

    :param shapefile: ESRI type .shp
    :param definition_type: osr.SpatialReference type
    :return: spatial reference in specified format
    """
    ds = ogr.Open(shapefile)
    layer = ds.GetLayer()
    layer_srs = layer.GetSpatialRef()
    comp = layer_srs.ExportToProj4()
    print comp
    return comp


def tif_spatial_reference(raster):
    ds = gdal.Open(raster)
    projection = ds.GetProjection()
    srs = osr.SpatialReference(wkt=projection)
    if srs.IsProjected():
        return srs.GetAttrValue('projcs')
    else:
        return srs.GetAttrValue('geocs')


def check_same_reference_system(first_geo, second_geo):
    if first_geo.endswith('.tif'):
        pass
    elif first_geo.endswith('.shp'):
        first_srs = shp_spatial_reference(first_geo)
    else:
        raise NotImplementedError('Must provide either shapefile or tif raster.')

    if second_geo.endswith('.tif'):
        pass
    elif second_geo.endswith('.shp'):
        second_srs = shp_spatial_reference(second_geo)
    else:
        raise NotImplementedError('Must provide either shapefile or tif raster.')

    if first_srs == second_srs:
        return True
    else:
        return False


if __name__ == '__main__':
    home = os.path.expanduser('~')

# ===============================================================================
