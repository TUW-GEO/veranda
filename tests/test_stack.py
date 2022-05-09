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

# import os
# import shutil
# import unittest
# from tempfile import mkdtemp
# from datetime import datetime
#
# import numpy as np
# import pandas as pd
# import xarray as xr
#
# from veranda.io.stack import GeoTiffRasterStack
# from veranda.io.stack import NcRasterStack
#
#
#
#
#
# class NcRasterStackTest(unittest.TestCase):
#
#     """
#     Testing NetCDF image stack.
#     """
#
#     def setUp(self):
#         """
#         Set up dummy mosaic set.
#         """
#         self.path = mkdtemp()
#
#     def test_write_read_image_stack(self):
#         """
#         Test writing and reading an image stack.
#         """
#         num_files = 50
#         xsize = 60
#         ysize = 50
#
#         dims = ['time', 'y', 'x']
#         coords = {'time': pd.date_range('2000-01-01', periods=num_files)}
#         data = np.random.randn(num_files, ysize, xsize)
#         ds = xr.Dataset({'mosaic': (dims, data)}, coords=coords)
#
#         with NcRasterStack(mode='w') as stack:
#             inventory = stack.write_netcdfs(ds, self.path)
#
#         with NcRasterStack(inventory=inventory) as stack:
#             ts = stack.read(band='mosaic')
#
#             np.testing.assert_equal(ts['mosaic'][:, :10, :10].values,
#                                     data[:, :10, :10])
#
#             np.testing.assert_equal(ts['mosaic'][:, 10:15, 12:17].values,
#                                     data[:, 10:15, 12:17])
#
#             np.testing.assert_equal(
#                 ts['mosaic'][1, :, :],
#                 ds.sel(time='2000-01-02')['mosaic'].values)
#
#             np.testing.assert_equal(
#                 ts['mosaic'][38, :, :],
#                 ds.sel(time='2000-02-08')['mosaic'].values)
#
#     def test_write_read_image_stack_small_timestamp(self):
#         """
#         Test writing and reading an image stack.
#         """
#         num_files = 100
#         xsize = 60
#         ysize = 50
#         #TODO: global attr?
#         dims = ['time', 'y', 'x']
#         coords = {'time': pd.date_range('1999-12-31-23-30-00',
#                                         periods=num_files,
#                                         freq='18min').astype('datetime64[ns]')}
#         data = np.random.randn(num_files, ysize, xsize)
#         ds = xr.Dataset({'mosaic': (dims, data)}, coords=coords)
#         with NcRasterStack(mode='w') as stack:
#             inventory = stack.write_netcdfs(ds, self.path, stack_size="12H")
#         filepaths = inventory['filepath']
#
#         for i in range(len(inventory)):
#             with NcRasterStack(inventory=inventory.iloc[[i],:]) as stack:
#                 ts = stack.read(band='mosaic')
#                 np.testing.assert_array_less(np.full(len(ts['time']),
#                                              inventory.index[i]).astype('datetime64[ns]'),
#                                              ts['time'].data)
#                 np.testing.assert_equal(ts['mosaic'].data[:, :, :],
#                                         data[np.isin(coords['time'].values,
#                                              ts['time'].data.astype('datetime64[ns]'))])
#
#     def test_read_decoded_image_stack(self):
#         """
#         Test decoding when reading an image stack.
#         """
#         num_files = 50
#         xsize = 60
#         ysize = 50
#
#         dims = ['time', 'x', 'y']
#         coords = {'time': pd.date_range('2000-01-01', periods=num_files)}
#         attr1 = {'unit': 'dB', 'scale_factor': 2, 'add_offset': 3, 'fill_value': -9999}
#         attr2 = {'unit': 'dB', 'fill_value': -9999}
#         data = np.ones((num_files, xsize, ysize))
#         ds = xr.Dataset({'data1': (dims, data,  attr1), 'data2': (dims, data, attr2)}, coords=coords)
#         filepath = os.path.join(self.path, "test.nc")
#         with NcRasterStack(mode='w') as stack:
#             stack.write(ds, filepath)
#
#         inventory = pd.DataFrame({'filepath': [filepath]})
#         with NcRasterStack(inventory=inventory, auto_decode=False) as stack:
#             data1 = stack.read(band='data1')['data1'].data
#             data2 = stack.read(band='data2')['data2'].data
#
#             np.testing.assert_equal(data1[:, 0:10, 0:10],
#                                     data[:, :10, :10])
#             np.testing.assert_equal(data2[:, 0:10, 0:10],
#                                     data[:, :10, :10])
#
#         with NcRasterStack(inventory=inventory, auto_decode=True) as stack:
#             data1 = stack.read(band='data1')['data1'].data
#             data2 = stack.read(band='data2')['data2'].data
#
#             np.testing.assert_equal(data1[:, 0:10, 0:10],
#                                     data[:, :10, :10]*2 + 3)
#             np.testing.assert_equal(data2[:, 0:10, 0:10],
#                                     data[:, :10, :10])
#
#     def test_write_read_image_stack_single_nc(self):
#         """
#         Test writing and reading an image stack.
#         """
#         num_files = 50
#         xsize = 60
#         ysize = 50
#
#         dims = ['time', 'y', 'x']
#         coords = {'time': pd.date_range('2000-01-01', periods=num_files)}
#         data = np.random.randn(num_files, ysize, xsize)
#         ds = xr.Dataset({'mosaic': (dims, data)}, coords=coords)
#
#         with NcRasterStack(mode='w') as stack:
#             filepath = os.path.join(self.path, "single.nc")
#             stack.write(ds, filepath)
#
#         inventory = pd.DataFrame({'filepath': [filepath]})
#         with NcRasterStack(inventory=inventory) as stack:
#             ts = stack.read(band='mosaic')
#
#             np.testing.assert_equal(ts['mosaic'][:, 0:10, 0:10].values,
#                                     data[:, :10, :10])
#
#             np.testing.assert_equal(ts['mosaic'][:, 10:15, 12:17],
#                                     data[:, 10:15, 12:17])
#
#             np.testing.assert_equal(
#                 ts['mosaic'][1, :, :].values,
#                 ds.sel(time='2000-01-02')['mosaic'].values)
#
#             np.testing.assert_equal(
#                 ts['mosaic'][38, :, :].values,
#                 ds.sel(time='2000-02-08')['mosaic'].values)
#
#     def test_export_tif(self):
#         """
#         Test exporting image stack to Geotiff.
#         """
#         num_files = 50
#         xsize = 60
#         ysize = 50
#
#         dims = ['time', 'y', 'x']
#         coords = {'time': pd.date_range('2000-01-01', periods=num_files)}
#         data = np.random.randn(num_files, ysize, xsize)
#         ds = xr.Dataset({'mosaic': (dims, data)}, coords=coords)
#
#         # write .nc stack
#         with NcRasterStack(mode='w') as stack:
#             inventory = stack.write_netcdfs(ds, self.path)
#
#         # read and export .nc stack to .tif stack
#         with NcRasterStack(inventory=inventory) as stack:
#             inventory = stack.export_to_tif(self.path, 'mosaic')
#
#         # read .tif stack
#         with GeoTiffRasterStack(inventory=inventory) as stack:
#             img1 = stack.read(idx=datetime(2000, 1, 2))
#             img2 = stack.read(idx=datetime(2000, 2, 8))
#
#         np.testing.assert_equal(img1,
#                                 ds.sel(time='2000-01-02')['mosaic'].values)
#         np.testing.assert_equal(img2,
#                                 ds.sel(time='2000-02-08')['mosaic'].values)
#
#     def tearDown(self):
#         """
#         Remove test file.
#         """
#         shutil.rmtree(self.path)
#
#
# if __name__ == '__main__':
#     unittest.main()
