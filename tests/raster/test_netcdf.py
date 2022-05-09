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
import unittest

import numpy as np
import pandas as pd
import xarray as xr

from veranda.raster.native.netcdf import NetCdf4File


class NetCdf4Test(unittest.TestCase):

    """
    Testing read and write for NetCDF4 files.
    """

    def setUp(self):
        """
        Define test file.
        """
        test_dir = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), 'temp')
        if not os.path.isdir(test_dir):
            os.makedirs(test_dir)
        self.filepath = os.path.join(test_dir, 'test.nc')

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
        dims = ['time', 'x', 'y']
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
        with NetCdf4File(self.filepath, mode='w', time_units=time_units) as nc:
            nc.write(ds_ref)

        with NetCdf4File(self.filepath, time_units=time_units) as nc:
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

    def tearDown(self):
        """
        Remove test file.
        """
        os.remove(self.filepath)


if __name__ == '__main__':
    unittest.main()