import os
import shutil
import unittest
import numpy as np
import pandas as pd
import xarray as xr
from geospade.crs import SpatialRef
from geospade.raster import Tile
from geospade.raster import MosaicGeometry

from veranda.raster.data.geotiff import GeoTiffDataReader, GeoTiffDataWriter


class GeoTiffDataTest(unittest.TestCase):

    """
    Testing a GeoTIFF image stack.
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
        band_name = 'data'

        ds_tile = Tile(ysize, xsize, SpatialRef(4326))
        ds_mosaic = MosaicGeometry.from_tile_list([ds_tile])

        dims = ['time', 'y', 'x']
        coords = {'time': pd.date_range('2000-01-01', periods=num_files),
                  'y': ds_tile.y_coords,
                  'x': ds_tile.x_coords}
        data = np.random.randn(num_files, ysize, xsize)
        ds = xr.Dataset({band_name: (dims, data)}, coords=coords)

        with GeoTiffDataWriter(ds_mosaic, data=ds, file_dimension='time', dirpath=self.path) as gt_data:
            gt_data.export()
            filepaths = gt_data.file_register['filepath']

        with GeoTiffDataReader.from_filepaths(filepaths) as gt_data:
            ts1 = gt_data.select_px_window(0, 0, width=10, height=10, inplace=False)
            ts1.read(band_names=(band_name,))
            ts2 = gt_data.select_px_window(10, 12, width=5, height=5, inplace=False)
            ts2.read(band_names=(band_name,))

        self.assertEqual(ts1.data[band_name].shape, (num_files, 10, 10))
        np.testing.assert_equal(ts1.data[band_name], data[:, :10, :10])

        self.assertEqual(ts2.data[band_name].shape, (num_files, 5, 5))
        np.testing.assert_equal(ts2.data[band_name], data[:, 10:15, 12:17])

    def test_read_decoded_image_stack(self):
        """
        Tests reading and decoding of an image stack.

        """
        num_files = 25
        xsize = 60
        ysize = 50

        ds_tile = Tile(ysize, xsize, SpatialRef(4326))
        ds_mosaic = MosaicGeometry.from_tile_list([ds_tile])

        dims = ['time', 'y', 'x']
        coords = {'time': pd.date_range('2000-01-01', periods=num_files),
                  'y': ds_tile.y_coords,
                  'x': ds_tile.x_coords}
        data = np.ones((num_files, ysize, xsize))
        attr1 = {'unit': 'dB', 'scale_factor': 2, 'add_offset': 3, 'fill_value': -9999}
        attr2 = {'unit': 'dB', 'fill_value': -9999}
        ds = xr.Dataset({'data1': (dims, data, attr1), 'data2': (dims, data, attr2)}, coords=coords)

        with GeoTiffDataWriter(ds_mosaic, data=ds, file_dimension='time', dirpath=self.path) as gt_data:
            gt_data.export()
            filepaths = gt_data.file_register['filepath']

        with GeoTiffDataReader.from_filepaths(filepaths) as gt_data:
            ts1 = gt_data.select_px_window(0, 0, width=10, height=10, inplace=False)
            ts1.read(bands=(1, 2), band_names=('data1', 'data2'), auto_decode=True)
            ts2 = gt_data.select_px_window(10, 12, width=5, height=5, inplace=False)
            ts2.read(bands=(1, 2), band_names=('data1', 'data2'), auto_decode=True)
            np.testing.assert_equal(ts1.data['data2'], data[:, :10, :10])
            np.testing.assert_equal(ts2.data['data2'], data[:, 10:15, 12:17])

        with GeoTiffDataReader.from_filepaths(filepaths) as gt_data:
            ts1 = gt_data.select_px_window(0, 0, width=10, height=10, inplace=False)
            ts1.read(bands=(1, 2), band_names=('data1', 'data2'), auto_decode=False)
            ts2 = gt_data.select_px_window(10, 12, width=5, height=5, inplace=False)
            ts2.read(bands=(1, 2), band_names=('data1', 'data2'), auto_decode=False)
            np.testing.assert_equal(ts1.data['data1'], data[:, :10, :10])
            np.testing.assert_equal(ts2.data['data1'], data[:, 10:15, 12:17])

        data = data * 2. + 3.
        with GeoTiffDataReader.from_filepaths(filepaths) as gt_data:
            ts1 = gt_data.select_px_window(0, 0, width=10, height=10, inplace=False)
            ts1.read(bands=(1, 2), band_names=('data1', 'data2'), auto_decode=True)
            ts2 = gt_data.select_px_window(10, 12, width=5, height=5, inplace=False)
            ts2.read(bands=(1, 2), band_names=('data1', 'data2'), auto_decode=True)
            np.testing.assert_equal(ts1.data['data1'], data[:, :10, :10])
            np.testing.assert_equal(ts2.data['data1'], data[:, 10:15, 12:17])

    def test_write_selections(self):
        """ Tests writing data after some select operations have been applied. """
        num_files = 10
        xsize = 60
        ysize = 50
        layer_ids = [0, 5, 9]

        ds_tile = Tile(ysize, xsize, SpatialRef(4326), name=0)
        ds_mosaic = MosaicGeometry.from_tile_list([ds_tile])

        dims = ['time', 'y', 'x']
        coords = {'time': pd.date_range('2000-01-01', periods=num_files),
                  'y': ds_tile.y_coords,
                  'x': ds_tile.x_coords}
        data = np.random.randn(num_files, ysize, xsize)
        ds = xr.Dataset({'data': (dims, data)}, coords=coords)

        with GeoTiffDataWriter(ds_mosaic, data=ds, file_dimension='time', dirpath=self.path) as gt_data:
            gt_data.select_layers(layer_ids, inplace=True)
            ul_data = gt_data.select_px_window(0, 0, height=25, width=30)
            ur_data = gt_data.select_px_window(0, 30, height=25, width=30)
            ll_data = gt_data.select_px_window(25, 0, height=25, width=30)
            lr_data = gt_data.select_px_window(25, 30, height=25, width=30)
            gt_data.write(ul_data.data)
            gt_data.write(ur_data.data)
            gt_data.write(ll_data.data)
            gt_data.write(lr_data.data)
            filepaths = gt_data.file_register['filepath']

        with GeoTiffDataReader.from_filepaths(filepaths) as gt_data:
            ts1 = gt_data.select_px_window(0, 0, width=10, height=10, inplace=False)
            ts1.read(band_names=('data',))
            ts2 = gt_data.select_px_window(45, 55, width=5, height=5, inplace=False)
            ts2.read(band_names=('data',))
            np.testing.assert_equal(ts1.data['data'], data[layer_ids, :10, :10])
            np.testing.assert_equal(ts2.data['data'], data[layer_ids, 45:, 55:])

    def tearDown(self):
        """
        Remove test file.
        """
        shutil.rmtree(self.path)


if __name__ == '__main__':
    pass