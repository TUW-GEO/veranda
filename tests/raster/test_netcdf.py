# Copyright (c) 2017, Vienna University of Technology (TU Wien), Department
# of Geodesy and Geoinformation (GEO).
# All rights reserved.
#
# All information contained herein is, and remains the property of Vienna
# University of Technology (TU Wien), Department of Geodesy and Geoinformation
# (GEO). The intellectual and technical concepts contained herein are
# proprietary to Vienna University of Technology (TU Wien), Department of
# Geodesy and Geoinformation (GEO). Dissemination of this information or
# reproduction of this material is forbidden unless prior written permission
# is obtained from Vienna University of Technology (TU Wien), Department of
# Geodesy and Geoinformation (GEO).

"""
Test NetCDF I/O.
"""

import os
import shutil
import unittest
import numpy as np
import pandas as pd
import xarray as xr
from tempfile import mkdtemp

from geospade.crs import SpatialRef
from geospade.raster import Tile
from geospade.raster import MosaicGeometry

from veranda.raster.mosaic.netcdf import NetCdfReader, NetCdfWriter
from veranda.raster.native.netcdf import NetCdf4File, NetCdfXrFile


class NetCdf4Test(unittest.TestCase):

    """
    Testing read and write for NetCDF4 files using netCDF4 as a backend.
    """

    def setUp(self):
        """
        Define test file.
        """
        self.filepath = os.path.join(mkdtemp(), 'test.nc')

    def test_read_write(self):
        """
        Test a simple write and read operation.
        """
        data = np.ones((100, 100, 100), dtype=np.float32)
        dims = ['time', 'y', 'x']
        coords = {'time': pd.date_range('2000-01-01', periods=data.shape[0])}
        attr1 = {'unit': 'dB'}
        attr2 = {'unit': 'degree'}

        ds_ref = xr.Dataset({'sig': (dims, data, attr1),
                             'inc': (dims, data, attr2),
                             'azi': (dims, data, attr2)}, coords=coords)

        with NetCdf4File(self.filepath, mode='w', data_variables=['sig', 'inc', 'azi'],
                         nodatavals={'inc': -9999, 'azi': -9999}, dtypes={'inc': 'int32', 'azi': 'int32'}) as nc:
            nc.write(ds_ref)

        with NetCdf4File(self.filepath) as nc:
            ds = nc.read()
            np.testing.assert_array_equal(ds['sig'][:], ds_ref['sig'][:])
            np.testing.assert_array_equal(ds['inc'][:], ds_ref['inc'][:])
            np.testing.assert_array_equal(ds['azi'][:], ds_ref['azi'][:])

    def test_auto_decoding(self):
        """
        Test automatic decoding of mosaic variables.
        """
        data = np.ones((100, 100, 100), dtype=np.float32)
        dims = ['time', 'y', 'x']
        coords = {'time': pd.date_range('2000-01-01', periods=data.shape[0])}
        attr1 = {'unit': 'dB', 'scale_factor': 2, 'add_offset': 3, 'fill_value': -9999}
        attr2 = {'unit': 'degree', '_FillValue': -9999, 'scale_factor': 2, 'add_offset': 0}
        attr3 = {'unit': 'degree', '_FillValue': -9999}

        ds_ref = xr.Dataset({'sig': (dims, data, attr1),
                             'inc': (dims, data, attr2),
                             'azi': (dims, data, attr3)}, coords=coords)

        with NetCdf4File(self.filepath, mode='w') as nc:
            nc.write(ds_ref)

        with NetCdf4File(self.filepath, mode='r', auto_decode=False) as nc:
            ds = nc.read()
            np.testing.assert_array_equal(ds['sig'][:], ds_ref['sig'][:])
            np.testing.assert_array_equal(ds['inc'][:], ds_ref['inc'][:])
            np.testing.assert_array_equal(ds['azi'][:], ds_ref['azi'][:])

        with NetCdf4File(self.filepath, mode='r', auto_decode=True) as nc:
            ds = nc.read()
            np.testing.assert_array_equal(ds['sig'][:], ds_ref['sig'][:] * 2 + 3)
            np.testing.assert_array_equal(ds['inc'][:], ds_ref['inc'][:] * 2)
            np.testing.assert_array_equal(ds['azi'][:], ds_ref['azi'][:])

    def test_append(self):
        """
        Test appending to existing NetCDF file.
        """
        data = np.ones((100, 100, 100), dtype=np.float32)
        dims = ['time', 'y', 'x']
        coords = {'time': pd.date_range('2000-01-01', periods=data.shape[0])}

        ds_ref = xr.Dataset({'sig': (dims, data),
                             'inc': (dims, data)}, coords=coords)

        with NetCdf4File(self.filepath, mode='w') as nc:
            nc.write(ds_ref)

        with NetCdf4File(self.filepath, mode='a') as nc:
            nc.write(ds_ref)

        with NetCdf4File(self.filepath, mode='a') as nc:
            nc.write(ds_ref)

        with NetCdf4File(self.filepath) as nc:
            ds = nc.read()

            np.testing.assert_array_equal(
                ds['sig'][:], np.repeat(ds_ref['sig'][:], 3, axis=0))

    def test_chunksizes(self):
        """
        Test setting chunksize.
        """
        data = np.ones((100, 100, 100), dtype=np.float32)
        dims = ['time', 'y', 'x']
        coords = {'time': pd.date_range('2000-01-01', periods=data.shape[0])}

        ds_ref = xr.Dataset({'sig': (dims, data),
                             'inc': (dims, data)}, coords=coords)

        chunksizes = (100, 10, 10)
        with NetCdf4File(self.filepath, mode='w', data_variables=list(ds_ref.data_vars), chunksizes=chunksizes) as nc:
            nc.write(ds_ref)

        with NetCdf4File(self.filepath, mode='r') as nc:
            ds = nc.read()
            self.assertEqual(ds['sig'].data.chunksize, chunksizes)
            self.assertEqual(ds['inc'].data.chunksize, chunksizes)

    def test_chunk_cache(self):
        """
        Test setting chunk cache.
        """
        data = np.ones((100, 100, 100), dtype=np.float32)
        dims = ['time', 'y', 'x']
        coords = {'time': pd.date_range('2000-01-01', periods=data.shape[0])}

        ds_ref = xr.Dataset({'sig': (dims, data),
                             'inc': (dims, data)}, coords=coords)

        size = 1024 * 64
        nelems = 500
        preemption = 0.75

        var_chunk_cache = (size, nelems, preemption)
        with NetCdf4File(self.filepath, mode='w', data_variables=list(ds_ref.data_vars),
                         var_chunk_caches=var_chunk_cache) as nc:
            nc.write(ds_ref)
            self.assertEqual(var_chunk_cache,
                             nc.src['sig'].get_var_chunk_cache())

        with NetCdf4File(self.filepath, mode='r', data_variables=list(ds_ref.data_vars),
                         var_chunk_caches=var_chunk_cache) as nc:
            self.assertEqual(var_chunk_cache,
                             nc.src['sig'].get_var_chunk_cache())

    def test_time_units(self):
        """
        Test time series and time units.
        """
        data = np.ones((100, 100, 100), dtype=np.float32)
        dims = ['time', 'y', 'x']
        coords = {'time': pd.date_range('2000-01-01', periods=data.shape[0])}

        ds_ref = xr.Dataset({'sig': (dims, data),
                             'inc': (dims, data)}, coords=coords)

        time_units = 'days since 2000-01-01 00:00:00'
        with NetCdf4File(self.filepath, mode='w', attrs={'time': {'units': time_units}}) as nc:
            nc.write(ds_ref)

        with NetCdf4File(self.filepath, attrs={'time': {'units': time_units}}) as nc:
            ds = nc.read()
            np.testing.assert_array_equal(pd.DatetimeIndex(ds['time'].data),
                                          coords['time'])

    def test_geotransform(self):
        """
        Test computation of x and y coordinates.
        """
        xdim = 100
        ydim = 200
        data = np.ones((100, ydim, xdim), dtype=np.float32)
        dims = ['time', 'y', 'x']
        coords = {'time': pd.date_range('2000-01-01', periods=data.shape[0])}

        ref_ds = xr.Dataset({'sig': (dims, data),
                              'inc': (dims, data)}, coords=coords)

        geotrans = (3000000.0, 500.0, 0.0, 1800000.0, 0.0, -500.0)
        with NetCdf4File(self.filepath, mode='w', geotrans=geotrans) as nc:
            nc.write(ref_ds)

        with NetCdf4File(self.filepath) as nc:
            ds = nc.read()

        x = geotrans[0] + (0.5 + np.arange(xdim)) * geotrans[1] + \
            (0.5 + np.arange(xdim)) * geotrans[2]
        y = geotrans[3] + (0.5 + np.arange(ydim)) * geotrans[4] + \
            (0.5 + np.arange(ydim)) * geotrans[5]

        np.testing.assert_array_equal(ds['x'].values, x)
        np.testing.assert_array_equal(ds['y'].values, y)

    def test_non_temporal_read_and_write(self):
        """ Test read and write for a dataset not containing any temporal information. """
        data = np.ones((100, 100, 100), dtype=np.float32)
        dims = ['layer', 'y', 'x']
        coords = {'layer': range(data.shape[0])}
        attr1 = {'unit': 'dB'}
        attr2 = {'unit': 'degree'}

        ds_ref = xr.Dataset({'sig': (dims, data, attr1),
                             'inc': (dims, data, attr2),
                             'azi': (dims, data, attr2)}, coords=coords)

        with NetCdf4File(self.filepath, mode='w', data_variables=['sig', 'inc', 'azi'], stack_dims={'layer': None},
                         nodatavals={'inc': -9999, 'azi': -9999}, dtypes={'inc': 'int32', 'azi': 'int32'}) as nc:
            nc.write(ds_ref)

        with NetCdf4File(self.filepath) as nc:
            ds = nc.read()
            np.testing.assert_array_equal(ds['sig'][:], ds_ref['sig'][:])
            np.testing.assert_array_equal(ds['inc'][:], ds_ref['inc'][:])
            np.testing.assert_array_equal(ds['azi'][:], ds_ref['azi'][:])

    def tearDown(self):
        """
        Remove test file.
        """
        os.remove(self.filepath)


class NetCdf4XrTest(unittest.TestCase):

    """
    Testing read and write for NetCDF4 files using xarray as a backend.
    """

    def setUp(self):
        """
        Define test file.
        """
        self.filepath = os.path.join(mkdtemp(), 'test.nc')

    def test_read_write(self):
        """
        Test a simple write and read operation.
        """
        data = np.ones((100, 100, 100), dtype=np.float32)
        dims = ['time', 'y', 'x']
        coords = {'time': pd.date_range('2000-01-01', periods=data.shape[0])}
        attr1 = {'unit': 'dB'}
        attr2 = {'unit': 'degree'}

        ds_ref = xr.Dataset({'sig': (dims, data, attr1),
                             'inc': (dims, data, attr2),
                             'azi': (dims, data, attr2)}, coords=coords)

        with NetCdfXrFile(self.filepath, mode='w', data_variables=['sig', 'inc', 'azi'],
                          nodatavals={'inc': -9999, 'azi': -9999}, dtypes={'inc': 'int32', 'azi': 'int32'}) as nc:
            nc.write(ds_ref)

        with NetCdfXrFile(self.filepath) as nc:
            ds = nc.read()
            np.testing.assert_array_equal(ds['sig'][:], ds_ref['sig'][:])
            np.testing.assert_array_equal(ds['inc'][:], ds_ref['inc'][:])
            np.testing.assert_array_equal(ds['azi'][:], ds_ref['azi'][:])

    def test_auto_decoding(self):
        """
        Test automatic decoding of mosaic variables.
        """
        data = np.ones((100, 100, 100), dtype=np.float32)
        dims = ['time', 'y', 'x']
        coords = {'time': pd.date_range('2000-01-01', periods=data.shape[0])}
        attr1 = {'unit': 'dB', 'scale_factor': 2, 'add_offset': 3, 'fill_value': -9999}
        attr2 = {'unit': 'degree', '_FillValue': -9999, 'scale_factor': 2, 'add_offset': 0}
        attr3 = {'unit': 'degree', '_FillValue': -9999}

        ds_ref = xr.Dataset({'sig': (dims, data, attr1),
                             'inc': (dims, data, attr2),
                             'azi': (dims, data, attr3)}, coords=coords)

        with NetCdfXrFile(self.filepath, mode='w') as nc:
            nc.write(ds_ref)

        with NetCdfXrFile(self.filepath, mode='r', auto_decode=False) as nc:
            ds = nc.read()
            np.testing.assert_array_equal(ds['sig'][:], ds_ref['sig'][:])
            np.testing.assert_array_equal(ds['inc'][:], ds_ref['inc'][:])
            np.testing.assert_array_equal(ds['azi'][:], ds_ref['azi'][:])

        with NetCdfXrFile(self.filepath, mode='r', auto_decode=True) as nc:
            ds = nc.read()
            np.testing.assert_array_equal(ds['sig'][:], ds_ref['sig'][:] * 2 + 3)
            np.testing.assert_array_equal(ds['inc'][:], ds_ref['inc'][:] * 2)
            np.testing.assert_array_equal(ds['azi'][:], ds_ref['azi'][:])

    def test_chunksizes(self):
        """
        Test setting chunksize.
        """
        data = np.ones((100, 100, 100), dtype=np.float32)
        dims = ['time', 'y', 'x']
        coords = {'time': pd.date_range('2000-01-01', periods=data.shape[0])}

        ds_ref = xr.Dataset({'sig': (dims, data),
                             'inc': (dims, data)}, coords=coords)

        chunksizes = (100, 10, 10)
        with NetCdfXrFile(self.filepath, mode='w', data_variables=list(ds_ref.data_vars), chunksizes=chunksizes) as nc:
            nc.write(ds_ref)

        with NetCdfXrFile(self.filepath, mode='r') as nc:
            ds = nc.read()
            self.assertEqual(ds['sig'].data.chunksize, chunksizes)
            self.assertEqual(ds['inc'].data.chunksize, chunksizes)

    def test_time_units(self):
        """
        Test time series and time units.
        """
        data = np.ones((100, 100, 100), dtype=np.float32)
        dims = ['time', 'y', 'x']
        coords = {'time': pd.date_range('2000-01-01', periods=data.shape[0])}

        ds_ref = xr.Dataset({'sig': (dims, data),
                             'inc': (dims, data)}, coords=coords)

        time_units = 'days since 2000-01-01 00:00:00'
        with NetCdfXrFile(self.filepath, mode='w', attrs={'time': {'units': time_units}}) as nc:
            nc.write(ds_ref)

        with NetCdfXrFile(self.filepath) as nc:
            ds = nc.read()
            np.testing.assert_array_equal(pd.DatetimeIndex(ds['time'].data),
                                          coords['time'])

    def test_geotransform(self):
        """
        Test computation of x and y coordinates.
        """
        xdim = 100
        ydim = 200
        data = np.ones((100, ydim, xdim), dtype=np.float32)
        dims = ['time', 'y', 'x']
        coords = {'time': pd.date_range('2000-01-01', periods=data.shape[0])}

        ref_ds = xr.Dataset({'sig': (dims, data),
                              'inc': (dims, data)}, coords=coords)

        geotrans = (3000000.0, 500.0, 0.0, 1800000.0, 0.0, -500.0)
        with NetCdfXrFile(self.filepath, mode='w', geotrans=geotrans) as nc:
            nc.write(ref_ds)

        with NetCdfXrFile(self.filepath) as nc:
            ds = nc.read()

        x = geotrans[0] + (0.5 + np.arange(xdim)) * geotrans[1] + \
            (0.5 + np.arange(xdim)) * geotrans[2]
        y = geotrans[3] + (0.5 + np.arange(ydim)) * geotrans[4] + \
            (0.5 + np.arange(ydim)) * geotrans[5]

        np.testing.assert_array_equal(ds['x'].values, x)
        np.testing.assert_array_equal(ds['y'].values, y)

    def test_non_temporal_read_and_write(self):
        """ Test read and write for a dataset not containing any temporal information. """
        data = np.ones((100, 100, 100), dtype=np.float32)
        dims = ['layer', 'y', 'x']
        coords = {'layer': range(data.shape[0])}
        attr1 = {'unit': 'dB'}
        attr2 = {'unit': 'degree'}

        ds_ref = xr.Dataset({'sig': (dims, data, attr1),
                             'inc': (dims, data, attr2),
                             'azi': (dims, data, attr2)}, coords=coords)

        with NetCdfXrFile(self.filepath, mode='w', data_variables=['sig', 'inc', 'azi'], stack_dims=['layer'],
                         nodatavals={'inc': -9999, 'azi': -9999}, dtypes={'inc': 'int32', 'azi': 'int32'}) as nc:
            nc.write(ds_ref)

        with NetCdfXrFile(self.filepath) as nc:
            ds = nc.read()
            np.testing.assert_array_equal(ds['sig'][:], ds_ref['sig'][:])
            np.testing.assert_array_equal(ds['inc'][:], ds_ref['inc'][:])
            np.testing.assert_array_equal(ds['azi'][:], ds_ref['azi'][:])

    def tearDown(self):
        """
        Remove test file.
        """
        os.remove(self.filepath)


class NetCdfDataTest(unittest.TestCase):
    """
    Testing a NetCDF image stack.
    """

    def setUp(self):
        """
        Set up dummy mosaic set.
        """
        self.path = mkdtemp()
        if not os.path.isdir(self.path):
            os.makedirs(self.path)

    def test_write_read_image_stack(self):
        """
        Test writing and reading an image stack.
        """
        num_files = 50
        xsize = 60
        ysize = 50
        data_variable = 'mosaic'

        ds_tile = Tile(ysize, xsize, sref=SpatialRef(4326))
        ds_mosaic = MosaicGeometry.from_tile_list([ds_tile])

        dims = ['time', 'y', 'x']
        coords = {'time': pd.date_range('2000-01-01', periods=num_files),
                  'y': ds_tile.y_coords,
                  'x': ds_tile.x_coords}
        data = np.random.randn(num_files, ysize, xsize)
        ds = xr.Dataset({data_variable: (dims, data)}, coords=coords)
        dst_filepath = os.path.join(self.path, "test.nc")

        with NetCdfWriter.from_data(ds, dst_filepath, mosaic=ds_mosaic, file_dimension='time') as nc_writer:
            nc_writer.export()
            filepaths = list(set(nc_writer.file_register['filepath']))

        with NetCdfReader.from_filepaths(filepaths) as nc_reader:
            ts1 = nc_reader.select_px_window(0, 0, width=10, height=10, inplace=False)
            ts1.read(data_variables=data_variable)
            ts2 = nc_reader.select_px_window(10, 12, width=5, height=5, inplace=False)
            ts2.read(data_variables=data_variable)

        self.assertEqual(ts1.data[data_variable].shape, (num_files, 10, 10))
        np.testing.assert_equal(ts1.data[data_variable], data[:, :10, :10])

        self.assertEqual(ts2.data[data_variable].shape, (num_files, 5, 5))
        np.testing.assert_equal(ts2.data[data_variable], data[:, 10:15, 12:17])

    def test_read_decoded_image_stack(self):
        """
        Tests reading and decoding of an image stack.

        """
        num_files = 25
        xsize = 60
        ysize = 50
        data_variables = ['data1', 'data2']

        ds_tile = Tile(ysize, xsize, SpatialRef(4326))
        ds_mosaic = MosaicGeometry.from_tile_list([ds_tile])

        dims = ['time', 'y', 'x']
        coords = {'time': pd.date_range('2000-01-01', periods=num_files),
                  'y': ds_tile.y_coords,
                  'x': ds_tile.x_coords}
        data = np.ones((num_files, ysize, xsize))
        attr1 = {'unit': 'dB', 'scale_factor': 2, 'add_offset': 3, 'fill_value': -9999}
        attr2 = {'unit': 'dB', 'fill_value': -9999}
        ds = xr.Dataset({data_variables[0]: (dims, data, attr1), data_variables[1]: (dims, data, attr2)}, coords=coords)
        dst_filepath = os.path.join(self.path, "test.nc")

        with NetCdfWriter.from_data(ds, dst_filepath, mosaic=ds_mosaic, file_dimension='time') as nc_writer:
            nc_writer.export()
            filepaths = list(set(nc_writer.file_register['filepath']))

        with NetCdfReader.from_filepaths(filepaths) as nc_reader:
            ts1 = nc_reader.select_px_window(0, 0, width=10, height=10, inplace=False)
            ts1.read(data_variables=data_variables, auto_decode=True)
            ts2 = nc_reader.select_px_window(10, 12, width=5, height=5, inplace=False)
            ts2.read(data_variables=data_variables, auto_decode=True)
            np.testing.assert_equal(ts1.data[data_variables[1]], data[:, :10, :10])
            np.testing.assert_equal(ts2.data[data_variables[1]], data[:, 10:15, 12:17])

        with NetCdfReader.from_filepaths(filepaths) as nc_reader:
            ts1 = nc_reader.select_px_window(0, 0, width=10, height=10, inplace=False)
            ts1.read(data_variables=data_variables, auto_decode=False)
            ts2 = nc_reader.select_px_window(10, 12, width=5, height=5, inplace=False)
            ts2.read(data_variables=data_variables, auto_decode=False)
            np.testing.assert_equal(ts1.data[data_variables[0]], data[:, :10, :10])
            np.testing.assert_equal(ts2.data[data_variables[0]], data[:, 10:15, 12:17])

        data = data * 2. + 3.
        with NetCdfReader.from_filepaths(filepaths) as nc_reader:
            ts1 = nc_reader.select_px_window(0, 0, width=10, height=10, inplace=False)
            ts1.read(data_variables=data_variables, auto_decode=True)
            ts2 = nc_reader.select_px_window(10, 12, width=5, height=5, inplace=False)
            ts2.read(data_variables=data_variables, auto_decode=True)
            np.testing.assert_equal(ts1.data[data_variables[0]], data[:, :10, :10])
            np.testing.assert_equal(ts2.data[data_variables[0]], data[:, 10:15, 12:17])

    def test_write_selections(self):
        """ Tests writing mosaic after some select operations have been applied. """
        num_files = 10
        xsize = 60
        ysize = 50
        layer_ids = [0, 5, 9]
        data_variable = 'mosaic'

        ds_tile = Tile(ysize, xsize, SpatialRef(4326), name=0)
        ds_mosaic = MosaicGeometry.from_tile_list([ds_tile])

        dims = ['time', 'y', 'x']
        dates = pd.date_range('2000-01-01', periods=num_files)
        coords = {'time': pd.date_range('2000-01-01', periods=num_files),
                  'y': ds_tile.y_coords,
                  'x': ds_tile.x_coords}
        data = np.random.randn(num_files, ysize, xsize)
        ds = xr.Dataset({'mosaic': (dims, data)}, coords=coords)
        dst_filepath = os.path.join(self.path, "test.nc")
        layers = dates[layer_ids]
        with NetCdfWriter.from_data(ds, dst_filepath, mosaic=ds_mosaic, file_dimension='time') as nc_writer:
            nc_writer.select_layers(layers, inplace=True)
            ul_data = nc_writer.select_px_window(0, 0, height=25, width=30)
            ur_data = nc_writer.select_px_window(0, 30, height=25, width=30)
            ll_data = nc_writer.select_px_window(25, 0, height=25, width=30)
            lr_data = nc_writer.select_px_window(25, 30, height=25, width=30)
            nc_writer.write(ul_data.data)
            nc_writer.write(ur_data.data)
            nc_writer.write(ll_data.data)
            nc_writer.write(lr_data.data)
            filepaths = list(set(nc_writer.file_register['filepath']))

        with NetCdfReader.from_filepaths(filepaths) as nc_reader:
            ts1 = nc_reader.select_px_window(0, 0, width=10, height=10, inplace=False)
            ts1.read(data_variables=data_variable)
            ts2 = nc_reader.select_px_window(45, 55, width=5, height=5, inplace=False)
            ts2.read(data_variables=data_variable)
            np.testing.assert_equal(ts1.data[data_variable], data[layer_ids, :10, :10])
            np.testing.assert_equal(ts2.data[data_variable], data[layer_ids, 45:, 55:])

    def tearDown(self):
        """
        Remove test file.
        """
        shutil.rmtree(self.path)


if __name__ == '__main__':
    unittest.main()