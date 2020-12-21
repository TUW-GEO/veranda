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
Test timestack functions.
"""

import os
import shutil
import unittest
from tempfile import mkdtemp
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr

from veranda.io.stack import GeoTiffRasterStack
from veranda.io.stack import NcRasterStack


class GeoTiffRasterStackTest(unittest.TestCase):

    """
    Testing Geotiff image stack.
    """

    def setUp(self):
        """
        Set up dummy data set.
        """
        self.path = test_dir = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), 'temp')
        if not os.path.isdir(test_dir):
            os.makedirs(self.path)

    def test_write_read_image_stack(self):
        """
        Test writing and reading an image stack.
        """
        num_files = 50
        xsize = 60
        ysize = 50

        dims = ['time', 'y', 'x']
        coords = {'time': pd.date_range('2000-01-01', periods=num_files)}
        data = np.random.randn(num_files, ysize, xsize)
        ds = xr.Dataset({'data': (dims, data)}, coords=coords)

        with GeoTiffRasterStack(mode='w') as stack:
            inventory_dict = stack.write_from_xr(ds, self.path)

        with GeoTiffRasterStack(inventory=inventory_dict['data']) as stack:
            ts1 = stack.read(col=0, row=0, n_cols=10, n_rows=10)
            ts2 = stack.read(col=12, row=10, n_cols=5, n_rows=5)
            img1 = stack.read(idx=datetime(2000,1,2))
            img2 = stack.read(idx=datetime(2000,2,8))

        self.assertEqual(ts1.shape, (num_files, 10, 10))
        np.testing.assert_equal(ts1, data[:, :10, :10])

        self.assertEqual(ts2.shape, (num_files, 5, 5))
        np.testing.assert_equal(ts2, data[:, 10:15, 12:17])

        self.assertEqual(img1.shape, (ysize, xsize))
        np.testing.assert_equal(
            img1, ds.sel(time=datetime(2000,1,2))['data'].values)

        self.assertEqual(img2.shape, (ysize, xsize))
        np.testing.assert_equal(
            img2, ds.sel(time=datetime(2000,2,8))['data'].values)

    def test_export_nc(self):
        """
        Test exporting image stack to NetCDF.
        """
        num_files = 50
        xsize = 60
        ysize = 50

        dims = ['time', 'y', 'x']
        coords = {'time': pd.date_range('2000-01-01', periods=num_files)}
        data = np.random.randn(num_files, ysize, xsize)
        ds = xr.Dataset({'data': (dims, data)}, coords=coords)

        # write .tif stack
        with GeoTiffRasterStack(mode='w') as stack:
            inventory = stack.write_from_xr(ds, self.path)

        # read and export .tif stack to .nc stack
        with GeoTiffRasterStack(inventory=inventory['data'], mode='r') as stack:
            inventory = stack.export_to_nc(self.path)

        # read .nc stack
        with NcRasterStack(inventory=inventory.drop_duplicates()) as stack:
            nc_ds = stack.read(band='data')
            np.testing.assert_equal(nc_ds['data'].values, ds['data'].values)

    def tearDown(self):
        """
        Remove test file.
        """
        shutil.rmtree(self.path)


class NcRasterStackTest(unittest.TestCase):

    """
    Testing NetCDF image stack.
    """

    def setUp(self):
        """
        Set up dummy data set.
        """
        self.path = mkdtemp()

    def test_write_read_image_stack(self):
        """
        Test writing and reading an image stack.
        """
        num_files = 50
        xsize = 60
        ysize = 50

        dims = ['time', 'y', 'x']
        coords = {'time': pd.date_range('2000-01-01', periods=num_files)}
        data = np.random.randn(num_files, ysize, xsize)
        ds = xr.Dataset({'data': (dims, data)}, coords=coords)

        with NcRasterStack(mode='w') as stack:
            inventory = stack.write_netcdfs(ds, self.path)

        with NcRasterStack(inventory=inventory) as stack:
            ts = stack.read(band='data')

            np.testing.assert_equal(ts['data'][:, :10, :10].values,
                                    data[:, :10, :10])

            np.testing.assert_equal(ts['data'][:, 10:15, 12:17].values,
                                    data[:, 10:15, 12:17])

            np.testing.assert_equal(
                ts['data'][1, :, :],
                ds.sel(time='2000-01-02')['data'].values)

            np.testing.assert_equal(
                ts['data'][38, :, :],
                ds.sel(time='2000-02-08')['data'].values)
    
    def test_write_read_image_stack_small_timestamp(self):
        """
        Test writing and reading an image stack.
        """
        num_files = 100
        xsize = 60
        ysize = 50
        #TODO: global attr?
        dims = ['time', 'y', 'x']
        coords = {'time': pd.date_range('1999-12-31-23-30-00',
                                        periods=num_files,
                                        freq='18min').astype('datetime64[ns]')}
        data = np.random.randn(num_files, ysize, xsize)
        ds = xr.Dataset({'data': (dims, data)}, coords=coords)
        with NcRasterStack(mode='w') as stack:
            inventory = stack.write_netcdfs(ds, self.path, stack_size="12H")
        filepaths = inventory['filepath']
        
        for i in range(len(inventory)):

            with NcRasterStack(inventory=inventory.iloc[[i],:]) as stack:
                ts = stack.read(band='data')
                np.testing.assert_array_less(np.full(len(ts['time']),
                                             inventory.index[i]),ts['time'])
                np.testing.assert_equal(ts['data'][:,:,:].values,
                                        data[np.isin(coords['time'].values,
                                             ts['time'].astype('datetime64[ns]').values)])

    def test_write_read_image_stack_single_nc(self):
        """
        Test writing and reading an image stack.
        """
        num_files = 50
        xsize = 60
        ysize = 50

        dims = ['time', 'y', 'x']
        coords = {'time': pd.date_range('2000-01-01', periods=num_files)}
        data = np.random.randn(num_files, ysize, xsize)
        ds = xr.Dataset({'data': (dims, data)}, coords=coords)

        with NcRasterStack(mode='w') as stack:
            filepath = os.path.join(self.path, "single.nc")
            stack.write(ds, filepath)

        inventory = pd.DataFrame({'filepath': [filepath]})
        with NcRasterStack(inventory=inventory) as stack:
            ts = stack.read(band='data')

            np.testing.assert_equal(ts['data'][:, 0:10, 0:10].values,
                                    data[:, :10, :10])

            np.testing.assert_equal(ts['data'][:, 10:15, 12:17],
                                    data[:, 10:15, 12:17])

            np.testing.assert_equal(
                ts['data'][1, :, :].values,
                ds.sel(time='2000-01-02')['data'].values)

            np.testing.assert_equal(
                ts['data'][38, :, :].values,
                ds.sel(time='2000-02-08')['data'].values)

    def test_export_tif(self):
        """
        Test exporting image stack to Geotiff.
        """
        num_files = 50
        xsize = 60
        ysize = 50

        dims = ['time', 'y', 'x']
        coords = {'time': pd.date_range('2000-01-01', periods=num_files)}
        data = np.random.randn(num_files, ysize, xsize)
        ds = xr.Dataset({'data': (dims, data)}, coords=coords)

        # write .nc stack
        with NcRasterStack(mode='w') as stack:
            inventory = stack.write_netcdfs(ds, self.path)

        # read and export .nc stack to .tif stack
        with NcRasterStack(inventory=inventory) as stack:
            inventory = stack.export_to_tif(self.path, 'data')

        # read .tif stack
        with GeoTiffRasterStack(inventory=inventory) as stack:
            img1 = stack.read(idx=datetime(2000,1,2))
            img2 = stack.read(idx=datetime(2000,2,8))

        np.testing.assert_equal(img1,
                                ds.sel(time='2000-01-02')['data'].values)
        np.testing.assert_equal(img2,
                                ds.sel(time='2000-02-08')['data'].values)

    def tearDown(self):
        """
        Remove test file.
        """
        shutil.rmtree(self.path)


if __name__ == '__main__':
    unittest.main()
