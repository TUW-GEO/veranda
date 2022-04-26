import os
import glob
import pandas as pd
from osgeo import ogr
from shapely.geometry import Polygon
from geospade.crs import SpatialRef

from veranda.data.raster.base import GeoTiffData


def create_geotiff_data():
    subgrid_dirpath = r"D:\data\code\yeoda\2021_11_29__rq_workshop\data\Sentinel-1_CSAR\IWGRDH\parameters\datasets\par\B0104\EQUI7_EU010M"
    filepaths_E041N022T1 = glob.glob(os.path.join(subgrid_dirpath, r'E041N022T1\mmensig0\*VV*.tif'))
    filepaths_E041N021T1 = glob.glob(os.path.join(subgrid_dirpath, r'E041N021T1\mmensig0\*VV*.tif'))
    filepaths = filepaths_E041N021T1 + filepaths_E041N022T1
    gt_data = GeoTiffData.from_mosaic_filepaths(filepaths)

    return gt_data


def test_from_filepaths():
    assert True


def test_from_mosaic_filepaths():
    gt_data = create_geotiff_data()
    assert True


def test_ts_reading_from_polygon():
    polygon = Polygon(((50.860565, -0.791828), (51.404334, -0.570728), (50.852763, -0.162861)))
    polygon = ogr.CreateGeometryFromWkt(polygon.wkt)
    gt_data = create_geotiff_data()
    #gt_data._mosaic.plot(show=True, label_tiles=True)
    gt_data.select_polygon(polygon, sref=SpatialRef(4326))
    #gt_data._mosaic.plot(show=True, label_tiles=True)
    data = gt_data.read(n_cores=2)

    gt_data_ts = gt_data.select_xy(4147303, 2190252, inplace=False)
    assert True


def test_export():
    polygon = Polygon(((50.860565, -0.791828), (51.404334, -0.570728), (50.852763, -0.162861)))
    polygon = ogr.CreateGeometryFromWkt(polygon.wkt)
    gt_data = create_geotiff_data()
    gt_data.select_polygon(polygon, sref=SpatialRef(4326))
    data = gt_data.read(n_cores=2)

    out_dirpath = r'D:\data\tmp\veranda\export_test'
    file_register_dict = dict()
    layer_ids = list(range(1, gt_data.n_layers))
    out_filepaths = [os.path.join(out_dirpath, f"{layer_id}.tif") for layer_id in layer_ids]
    file_register_dict['filepath'] = out_filepaths
    file_register_dict['geom_id'] = [gt_data._mosaic.tile_names[0]] * len(out_filepaths)
    file_register_dict['layer_id'] = layer_ids
    out_file_register = pd.DataFrame(file_register_dict)
    out_gt_data = GeoTiffData.from_other(gt_data, file_register=out_file_register)
    out_gt_data.export()


def test_write():
    polygon = Polygon(((50.860565, -0.791828), (51.404334, -0.570728), (50.852763, -0.162861)))
    polygon = ogr.CreateGeometryFromWkt(polygon.wkt)
    gt_data = create_geotiff_data()
    gt_data.select_polygon(polygon, sref=SpatialRef(4326))
    data = gt_data.read(n_cores=2)

    n_rows = gt_data._mosaic.tiles[0].n_rows
    data_1 = data.sel(y=slice(None, data.y.data[int(n_rows/3)]))
    data_2 = data.sel(y=slice(data.y.data[int(2*n_rows / 3)], None))

    out_dirpath = r'D:\data\tmp\veranda\write_test'
    file_register_dict = dict()
    layer_ids = list(range(1, gt_data.n_layers + 1))
    out_filepaths = [os.path.join(out_dirpath, f"{layer_id}.tif") for layer_id in layer_ids]
    file_register_dict['filepath'] = out_filepaths
    file_register_dict['geom_id'] = [gt_data._mosaic.tile_names[0]] * len(out_filepaths)
    file_register_dict['layer_id'] = layer_ids
    out_file_register = pd.DataFrame(file_register_dict)
    with GeoTiffData.from_other(gt_data, file_register=out_file_register) as out_gt_data:
        out_gt_data.write(data_1)
        out_gt_data.write(data_2)


if __name__ == '__main__':
    import matplotlib

    matplotlib.use('Qt5Agg')
    #test_ts_reading_from_polygon()
    #test_export()
    test_write()