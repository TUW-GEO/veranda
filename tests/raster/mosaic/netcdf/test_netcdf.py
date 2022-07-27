import gc
import os
from mosaic_common import *
from veranda.raster.mosaic.netcdf import NetCdfReader, NetCdfWriter


@pytest.fixture
def clean_garbage():
    try:
        yield
    finally:
        gc.collect()


def test_write_read_image_stack(simple_ds, mosaic, tmp_path, clean_garbage):
    data_var_name = [data_var for data_var in simple_ds.data_vars][0]
    dst_filepath = os.path.join(tmp_path, "test.nc")

    with NetCdfWriter.from_data(simple_ds, dst_filepath, mosaic=mosaic, stack_dimension='time') as nc_writer:
        nc_writer.export()

    with NetCdfReader.from_filepaths([dst_filepath]) as nc_reader:
        assert_reader_data(simple_ds, nc_reader, 0, 0, 10, 10, [data_var_name],
                           data_variables=[data_var_name])
        assert_reader_data(simple_ds, nc_reader, 10, 12, 5, 5, [data_var_name],
                           data_variables=[data_var_name])


def test_read_decoded_image_stack(complex_ds, mosaic, tmp_path, clean_garbage):
    data_var_names = [data_var for data_var in complex_ds.data_vars]
    dst_filepath = os.path.join(tmp_path, "test.nc")

    with NetCdfWriter.from_data(complex_ds, dst_filepath, mosaic=mosaic, stack_dimension='time') as nc_writer:
        nc_writer.export()
        filepaths = list(set(nc_writer.file_register['filepath']))

    with NetCdfReader.from_filepaths(filepaths) as nc_reader:
        assert_reader_data(complex_ds, nc_reader, 0, 0, 10, 10, data_var_names[1:],
                           data_variables=data_var_names[1:], auto_decode=True)
        assert_reader_data(complex_ds, nc_reader, 10, 12, 5, 5, data_var_names[1:],
                           data_variables=data_var_names[1:], auto_decode=True)

    with NetCdfReader.from_filepaths(filepaths) as nc_reader:
        assert_reader_data(complex_ds, nc_reader, 0, 0, 10, 10, data_var_names[:1],
                           data_variables=data_var_names[:1], auto_decode=False)
        assert_reader_data(complex_ds, nc_reader, 10, 12, 5, 5, data_var_names[:1],
                           data_variables=data_var_names[:1], auto_decode=False)

    complex_ds[data_var_names[0]].data = complex_ds[data_var_names[0]].data * 2. + 3.
    with NetCdfReader.from_filepaths(filepaths) as nc_reader:
        assert_reader_data(complex_ds, nc_reader, 0, 0, 10, 10, data_var_names[:1],
                           data_variables=data_var_names[:1], auto_decode=True)
        assert_reader_data(complex_ds, nc_reader, 10, 12, 5, 5, data_var_names[:1],
                           data_variables=data_var_names[:1], auto_decode=True)


def test_write_selections(simple_ds, mosaic, tmp_path, clean_garbage):
    data_var_name = [data_var for data_var in simple_ds.data_vars][0]
    dst_filepath = os.path.join(tmp_path, "test.nc")
    layer_ids = [0, 5, 9]
    layers = list(simple_ds['time'].data[layer_ids])

    with NetCdfWriter.from_data(simple_ds, dst_filepath, mosaic=mosaic, stack_dimension='time') as nc_writer:
        nc_writer.select_layers(layers, inplace=True)
        write_pixel_window(nc_writer, 0, 0, 25, 30)
        write_pixel_window(nc_writer, 0, 30, 25, 30)
        write_pixel_window(nc_writer, 25, 0, 25, 30)
        write_pixel_window(nc_writer, 25, 30, 25, 30)
        filepaths = list(set(nc_writer.file_register['filepath']))

    simple_ds = simple_ds.sel({'time': layers})
    with NetCdfReader.from_filepaths(filepaths) as nc_reader:
        assert_reader_data(simple_ds, nc_reader, 0, 0, 10, 10, [data_var_name],
                           data_variables=[data_var_name])
        assert_reader_data(simple_ds, nc_reader, 45, 55, 5, 5, [data_var_name],
                           data_variables=[data_var_name])
