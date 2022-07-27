from veranda.raster.mosaic.geotiff import GeoTiffReader, GeoTiffWriter
from mosaic_common import *


def test_write_read_image_stack(simple_ds, mosaic, tmp_path):
    band_name = [data_var for data_var in simple_ds.data_vars][0]

    with GeoTiffWriter(mosaic, data=simple_ds, stack_dimension='time', dirpath=tmp_path,
                       fn_pattern='{time}.tif', fn_formatter={'time': lambda x: x.strftime('%Y%m%d')}) as gt_writer:
        gt_writer.export()
        filepaths = list(gt_writer.file_register['filepath'])

    with GeoTiffReader.from_filepaths(filepaths) as gt_reader:
        assert_reader_data(simple_ds, gt_reader, 0, 0, 10, 10, [band_name],
                           bands=[1], band_names=[band_name])
        assert_reader_data(simple_ds, gt_reader, 10, 12, 5, 5, [band_name],
                           bands=[1], band_names=[band_name])


def test_read_decoded_image_stack(complex_ds, mosaic, tmp_path):
    band_names = [data_var for data_var in complex_ds.data_vars]

    with GeoTiffWriter(mosaic, data=complex_ds, stack_dimension='time', dirpath=tmp_path,
                       fn_pattern='{time}.tif', fn_formatter={'time': lambda x: x.strftime('%Y%m%d')}) as gt_writer:
        gt_writer.export()
        filepaths = list(gt_writer.file_register['filepath'])

    with GeoTiffReader.from_filepaths(filepaths) as gt_reader:
        assert_reader_data(complex_ds, gt_reader, 0, 0, 10, 10, band_names[1:],
                           bands=[2], band_names=band_names[1:], auto_decode=True)
        assert_reader_data(complex_ds, gt_reader, 10, 12, 5, 5, band_names[1:],
                           bands=[2], band_names=band_names[1:], auto_decode=True)

    with GeoTiffReader.from_filepaths(filepaths) as gt_reader:
        assert_reader_data(complex_ds, gt_reader, 0, 0, 10, 10, band_names,
                           bands=(1, 2), band_names=band_names, auto_decode=False)
        assert_reader_data(complex_ds, gt_reader, 10, 12, 5, 5, band_names,
                           bands=(1, 2), band_names=band_names, auto_decode=False)

    complex_ds[band_names[0]].data = complex_ds[band_names[0]].data * 2. + 3.
    with GeoTiffReader.from_filepaths(filepaths) as gt_reader:
        assert_reader_data(complex_ds, gt_reader, 0, 0, 10, 10, band_names[:1],
                           bands=[1], band_names=band_names[:1], auto_decode=True)
        assert_reader_data(complex_ds, gt_reader, 10, 12, 5, 5, band_names[:1],
                           bands=[1], band_names=band_names[:1], auto_decode=True)


def test_write_after_selections(simple_ds, mosaic, tmp_path):
    band_name = [data_var for data_var in simple_ds.data_vars][0]
    layer_ids = [0, 5, 9]
    layers = list(simple_ds['time'].data[layer_ids])

    with GeoTiffWriter(mosaic, data=simple_ds, stack_dimension='time', dirpath=tmp_path,
                       fn_pattern='{time}.tif', fn_formatter={'time': lambda x: x.strftime('%Y%m%d')}) as gt_writer:
        gt_writer.select_layers(layers, inplace=True)
        write_pixel_window(gt_writer, 0, 0, 25, 30)
        write_pixel_window(gt_writer, 0, 30, 25, 30)
        write_pixel_window(gt_writer, 25, 0, 25, 30)
        write_pixel_window(gt_writer, 25, 30, 25, 30)
        filepaths = list(gt_writer.file_register['filepath'])

    simple_ds = simple_ds.sel({'time': layers})
    with GeoTiffReader.from_filepaths(filepaths) as gt_reader:
        assert_reader_data(simple_ds, gt_reader, 0, 0, 10, 10, [band_name],
                           bands=[1], band_names=[band_name])
        assert_reader_data(simple_ds, gt_reader, 45, 55, 5, 5, [band_name],
                           bands=[1], band_names=[band_name])
