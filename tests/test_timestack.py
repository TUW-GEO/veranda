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

import numpy as np
import pandas as pd
import xarray as xr

from pyraster.timestack import GeoTiffRasterTimeStack
from pyraster.timestack import NcRasterTimeStack


class GeoTiffRasterTimeStackTest(unittest.TestCase):

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

        dims = ['time', 'x', 'y']
        coords = {'time': pd.date_range('2000-01-01', periods=num_files)}
        data = np.random.randn(num_files, xsize, ysize)
        ds = xr.Dataset({'data': (dims, data)}, coords=coords)

        with GeoTiffRasterTimeStack(mode='w', out_path=self.path) as stack:
            file_ts = stack.write(ds)

        with GeoTiffRasterTimeStack(file_ts=file_ts['data']) as stack:
            ts1 = stack.read_ts(0, 0, 10, 10)
            ts2 = stack.read_ts(12, 10, 5, 5)
            img1 = stack.read_img('2000-01-02')
            img2 = stack.read_img('2000-02-08')

        self.assertEqual(ts1.shape, (num_files, 10, 10))
        np.testing.assert_equal(ts1, data[:, :10, :10])

        self.assertEqual(ts2.shape, (num_files, 5, 5))
        np.testing.assert_equal(ts2, data[:, 10:15, 12:17])

        self.assertEqual(img1.shape, (xsize, ysize))
        np.testing.assert_equal(
            img1, ds.sel(time='2000-01-02')['data'].values)

        self.assertEqual(img2.shape, (xsize, ysize))
        np.testing.assert_equal(
            img2, ds.sel(time='2000-02-08')['data'].values)

    def test_export_nc(self):
        """
        Test exporting image stack to NetCDF.
        """
        num_files = 50
        xsize = 60
        ysize = 50

        dims = ['time', 'x', 'y']
        coords = {'time': pd.date_range('2000-01-01', periods=num_files)}
        data = np.random.randn(num_files, xsize, ysize)
        ds = xr.Dataset({'data': (dims, data)}, coords=coords)

        # write .tif stack
        with GeoTiffRasterTimeStack(mode='w', out_path=self.path) as stack:
            file_ts = stack.write(ds)

        # read and export .tif stack to .nc stack
        with GeoTiffRasterTimeStack(file_ts=file_ts['data']) as stack:
            file_ts = stack.export_to_nc(self.path)

        # read .nc stack
        with NcRasterTimeStack(file_ts=file_ts.drop_duplicates()) as stack:
            nc_ds = stack.read()
            np.testing.assert_equal(nc_ds['data'].values, ds['data'].values)

    def tearDown(self):
        """
        Remove test file.
        """
        shutil.rmtree(self.path)


class NcRasterTimeStackTest(unittest.TestCase):

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

        dims = ['time', 'x', 'y']
        coords = {'time': pd.date_range('2000-01-01', periods=num_files)}
        data = np.random.randn(num_files, xsize, ysize)
        ds = xr.Dataset({'data': (dims, data)}, coords=coords)

        with NcRasterTimeStack(mode='w', out_path=self.path) as stack:
            file_ts = stack.write(ds)

        with NcRasterTimeStack(file_ts=file_ts) as stack:
            ts = stack.read()

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

    def test_write_read_image_stack_single_nc(self):
        """
        Test writing and reading an image stack.
        """
        num_files = 50
        xsize = 60
        ysize = 50

        dims = ['time', 'x', 'y']
        coords = {'time': pd.date_range('2000-01-01', periods=num_files)}
        data = np.random.randn(num_files, xsize, ysize)
        ds = xr.Dataset({'data': (dims, data)}, coords=coords)

        with NcRasterTimeStack(
                mode='w', out_path=self.path, stack_size='single',
                fn_prefix='single') as stack:
            file_ts = stack.write(ds)

        with NcRasterTimeStack(file_ts=file_ts) as stack:
            ts = stack.read()

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

        dims = ['time', 'x', 'y']
        coords = {'time': pd.date_range('2000-01-01', periods=num_files)}
        data = np.random.randn(num_files, xsize, ysize)
        ds = xr.Dataset({'data': (dims, data)}, coords=coords)

        # write .nc stack
        with NcRasterTimeStack(mode='w', out_path=self.path) as stack:
            file_ts = stack.write(ds)

        # read and export .nc stack to .tif stack
        with NcRasterTimeStack(file_ts=file_ts) as stack:
            file_ts = stack.export_to_tif(self.path, 'data')

        # read .tif stack
        with GeoTiffRasterTimeStack(file_ts=file_ts) as stack:
            img1 = stack.read_img('2000-01-02')
            img2 = stack.read_img('2000-02-08')

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
