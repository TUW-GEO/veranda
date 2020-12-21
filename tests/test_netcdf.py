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

import netCDF4
import numpy as np
import pandas as pd
import xarray as xr

from veranda.io.netcdf import NcFile


class NcTest(unittest.TestCase):

    """
    Testing read and write for NetCDF files.
    """

    def setUp(self):
        """
        Define test file.
        """
        test_dir = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), 'temp')
        if not os.path.isdir(test_dir):
            os.makedirs(test_dir)
        self.filename = os.path.join(test_dir, 'test.nc')

    def test_read_write(self):
        """
        Test a simple write and read operation.
        """
        data = np.ones((100, 100, 100), dtype=np.float32)
        dims = ['time', 'x', 'y']
        coords = {'time': pd.date_range('2000-01-01', periods=data.shape[0])}
        attr1 = {'unit': 'dB'}
        attr2 = {'unit': 'degree', 'fill_value': -9999}

        self.ds = xr.Dataset({'sig': (dims, data, attr1),
                              'inc': (dims, data, attr2),
                              'azi': (dims, data, attr2)}, coords=coords)

        with NcFile(self.filename, mode='w') as nc:
            nc.write(self.ds)

        with NcFile(self.filename) as nc:
            ds = nc.read()
            np.testing.assert_array_equal(ds['sig'][:], self.ds['sig'][:])
            np.testing.assert_array_equal(ds['inc'][:], self.ds['inc'][:])
            np.testing.assert_array_equal(ds['azi'][:], self.ds['azi'][:])

    def test_auto_decoding(self):
        """
        Test automatic decoding of data variables.
        """
        data = np.ones((100, 100, 100), dtype=np.float32)
        dims = ['time', 'x', 'y']
        coords = {'time': pd.date_range('2000-01-01', periods=data.shape[0])}
        attr1 = {'unit': 'dB', 'scale_factor': 2, 'add_offset': 3, 'fill_value': -9999}
        attr2 = {'unit': 'degree', 'fill_value': -9999, 'scale_factor': 2, 'add_offset': 0}
        attr3 = {'unit': 'degree', 'fill_value': -9999}

        self.ds = xr.Dataset({'sig': (dims, data, attr1),
                              'inc': (dims, data, attr2),
                              'azi': (dims, data, attr3)}, coords=coords)

        with NcFile(self.filename, mode='w') as nc:
            nc.write(self.ds)

        with NcFile(self.filename, mode='r_xarray', auto_decode=False) as nc:
            ds = nc.read()
            np.testing.assert_array_equal(ds['sig'][:], self.ds['sig'][:])
            np.testing.assert_array_equal(ds['inc'][:], self.ds['inc'][:])
            np.testing.assert_array_equal(ds['azi'][:], self.ds['azi'][:])

        with NcFile(self.filename, mode='r_xarray', auto_decode=True) as nc:
            ds = nc.read()
            np.testing.assert_array_equal(ds['sig'][:], self.ds['sig'][:]*2 + 3)
            np.testing.assert_array_equal(ds['inc'][:], self.ds['inc'][:]*2)
            np.testing.assert_array_equal(ds['azi'][:], self.ds['azi'][:])

        with NcFile(self.filename, mode='r_netcdf', auto_decode=False) as nc:
            ds = nc.read()
            np.testing.assert_array_equal(ds['sig'][:], self.ds['sig'][:])
            np.testing.assert_array_equal(ds['inc'][:], self.ds['inc'][:])
            np.testing.assert_array_equal(ds['azi'][:], self.ds['azi'][:])

        with NcFile(self.filename, mode='r_netcdf', auto_decode=True) as nc:
            ds = nc.read()
            np.testing.assert_array_equal(ds['sig'][:], self.ds['sig'][:] * 2 + 3)
            np.testing.assert_array_equal(ds['inc'][:], self.ds['inc'][:] * 2)
            np.testing.assert_array_equal(ds['azi'][:], self.ds['azi'][:])

    def test_append(self):
        """
        Test appending to existing NetCDF file.
        """
        data = np.ones((100, 100, 100), dtype=np.float32)
        dims = ['time', 'x', 'y']
        coords = {'time': pd.date_range('2000-01-01', periods=data.shape[0])}

        self.ds = xr.Dataset({'sig': (dims, data),
                              'inc': (dims, data)}, coords=coords)

        with NcFile(self.filename, mode='w') as nc:
            nc.write(self.ds)

        with NcFile(self.filename, mode='a') as nc:
            nc.write(self.ds)

        with NcFile(self.filename, mode='a') as nc:
            nc.write(self.ds)

        with NcFile(self.filename) as nc:
            ds = nc.read()

            np.testing.assert_array_equal(
                ds['sig'][:], np.repeat(self.ds['sig'][:], 3, axis=0))

    def test_chunksizes(self):
        """
        Test setting chunksize.
        """
        data = np.ones((100, 100, 100), dtype=np.float32)
        dims = ['time', 'x', 'y']
        coords = {'time': pd.date_range('2000-01-01', periods=data.shape[0])}

        self.ds = xr.Dataset({'sig': (dims, data),
                              'inc': (dims, data)}, coords=coords)

        chunksizes = (100, 10, 10)
        with NcFile(self.filename, mode='w', chunksizes=chunksizes) as nc:
            nc.write(self.ds)

        with NcFile(self.filename, mode='r_netcdf') as nc:
            ds = nc.read()
            self.assertEqual(ds['sig'].chunking(), list(chunksizes))
            self.assertEqual(ds['inc'].chunking(), list(chunksizes))

    def test_chunk_cache(self):
        """
        Test setting chunk cache.
        """
        data = np.ones((100, 100, 100), dtype=np.float32)
        dims = ['time', 'x', 'y']
        coords = {'time': pd.date_range('2000-01-01', periods=data.shape[0])}

        self.ds = xr.Dataset({'sig': (dims, data),
                              'inc': (dims, data)}, coords=coords)

        size = 1024 * 64
        nelems = 500
        preemption = 0.75

        var_chunk_cache = (size, nelems, preemption)
        with NcFile(self.filename, mode='w',
                    var_chunk_cache=var_chunk_cache) as nc:
            nc.write(self.ds)
            self.assertEqual(var_chunk_cache,
                             nc.src['sig'].get_var_chunk_cache())

        with NcFile(self.filename, mode='r_netcdf',
                    var_chunk_cache=var_chunk_cache) as nc:
            self.assertEqual(var_chunk_cache,
                             nc.src['sig'].get_var_chunk_cache())

    def test_time_units(self):
        """
        Test time series and time units.
        """
        data = np.ones((100, 100, 100), dtype=np.float32)
        dims = ['time', 'x', 'y']
        coords = {'time': pd.date_range('2000-01-01', periods=data.shape[0])}

        self.ds = xr.Dataset({'sig': (dims, data),
                              'inc': (dims, data)}, coords=coords)

        time_units = 'days since 2000-01-01 00:00:00'
        with NcFile(self.filename, mode='w', time_units=time_units) as nc:
            nc.write(self.ds)

        with NcFile(self.filename, time_units=time_units) as nc:
            ds = nc.read()
            time_stamps = netCDF4.num2date(ds['time'][:], nc.time_units)
            np.testing.assert_array_equal(pd.DatetimeIndex(time_stamps),
                                          coords['time'])

    def test_geotransform(self):
        """
        Test computation of x and y coordinates.
        """
        xdim = 100
        ydim = 200
        data = np.ones((100, xdim, ydim), dtype=np.float32)
        dims = ['time', 'x', 'y']
        coords = {'time': pd.date_range('2000-01-01', periods=data.shape[0])}

        self.ds = xr.Dataset({'sig': (dims, data),
                              'inc': (dims, data)}, coords=coords)

        geotransform = (3000000.0, 500.0, 0.0, 1800000.0, 0.0, -500.0)
        with NcFile(self.filename, mode='w', geotransform=geotransform) as nc:
            nc.write(self.ds)

        with NcFile(self.filename) as nc:
            ds = nc.read()

        x = geotransform[0] + (0.5 + np.arange(xdim)) * geotransform[1] + \
            (0.5 + np.arange(xdim)) * geotransform[2]
        y = geotransform[3] + (0.5 + np.arange(ydim)) * geotransform[4] + \
            (0.5 + np.arange(ydim)) * geotransform[5]

        np.testing.assert_array_equal(ds['x'].values, x)
        np.testing.assert_array_equal(ds['y'].values, y)

    def tearDown(self):
        """
        Remove test file.
        """
        os.remove(self.filename)


if __name__ == '__main__':
    unittest.main()
