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

from veranda.raster.native.geotiff import GeoTiffFile
from veranda.raster.mosaic.geotiff import GeoTiffReader, GeoTiffWriter


class GeoTiffFileTest(unittest.TestCase):

    def setUp(self):
        """
        Set up dummy mosaic set.
        """
        self.filepath = os.path.join(mkdtemp(), 'test.tif')

    def test_read_write_1_band(self):
        """
        Test a write and read of a single band mosaic.
        """
        data = np.ones((1, 100, 100), dtype=np.float32)

        with GeoTiffFile(self.filepath, mode='w') as src:
            src.write(data)

        with GeoTiffFile(self.filepath) as src:
            ds = src.read()

        np.testing.assert_array_equal(ds[1], data[0, :, :])

    def test_read_write_multi_band(self):
        """
        Test write and read of a multi band mosaic.
        """
        data = np.ones((5, 100, 100), dtype=np.float32)

        with GeoTiffFile(self.filepath, mode='w', n_bands=5) as src:
            src.write(data)

        with GeoTiffFile(self.filepath) as src:
            for band in np.arange(data.shape[0]):
                ds = src.read(bands=band + 1)
                np.testing.assert_array_equal(
                    ds[band + 1], data[band, :, :])

    def test_decoding_multi_band(self):
        """
        Test decoding of multi band mosaic.

        """
        data = np.ones((5, 100, 100), dtype=np.float32)
        scale_factors = {1: 1, 2: 2, 3: 1, 4: 1, 5: 3}
        offsets = {2: 3}
        with GeoTiffFile(self.filepath, mode='w', n_bands=5,
                         scale_factors=scale_factors, offsets=offsets) as src:
            src.write(data)

        with GeoTiffFile(self.filepath, auto_decode=False) as src:
            ds = src.read(bands=1)
            np.testing.assert_array_equal(ds[1], data[0, :, :])
            ds = src.read(bands=2)
            np.testing.assert_array_equal(ds[2], data[1, :, :])
            ds = src.read(bands=5)
            np.testing.assert_array_equal(ds[5], data[4, :, :])

        data[1, :, :] = data[1, :, :] * 2 + 3
        data[4, :, :] = data[4, :, :] * 3

        with GeoTiffFile(self.filepath, auto_decode=True) as src:
            ds = src.read(bands=1)
            np.testing.assert_array_equal(ds[1], data[0, :, :])
            ds = src.read(bands=2)
            np.testing.assert_array_equal(ds[2], data[1, :, :])
            ds = src.read(bands=5)
            np.testing.assert_array_equal(ds[5], data[4, :, :])

    def test_read_write_specific_band(self):
        """
        Test a write and read of a specific mosaic.
        """
        data = np.ones((100, 100), dtype=np.float32)

        with GeoTiffFile(self.filepath, mode='w', n_bands=10) as src:
            src.write({5: data})
            src.write({10: data})

        with GeoTiffFile(self.filepath) as src:
            ds = src.read(bands=5)
            np.testing.assert_array_equal(ds[5], data)
            ds = src.read(bands=10)
            np.testing.assert_array_equal(ds[10], data)

    def test_metadata(self):
        """
        Test meta mosaic tags.
        """
        data = np.ones((1, 100, 100), dtype=np.float32)

        metadata = {'attr1': '123', 'attr2': 'test'}

        with GeoTiffFile(self.filepath, mode='w', metadata=metadata) as src:
            src.write(data)

        with GeoTiffFile(self.filepath) as src:
            ds_md = src.metadata

        self.assertEqual(ds_md, metadata)

    def test_geotransform(self):
        """
        Test geotransform and spatialref keywords.

        Notes
        -----
        Spatial reference cannot be compared due to different versions of WKT1 strings.

        """
        data = np.ones((1, 100, 100), dtype=np.float32)

        geotrans_ref = (3000000.0, 500.0, 0.0, 1800000.0, 0.0, -500.0)
        sref = ('PROJCS["Azimuthal_Equidistant",GEOGCS["WGS 84",'
                     'DATUM["WGS_1984",SPHEROID["WGS 84",6378137,'
                     '298.257223563,AUTHORITY["EPSG","7030"]],'
                     'AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],'
                     'UNIT["degree",0.0174532925199433],'
                     'AUTHORITY["EPSG","4326"]],'
                     'PROJECTION["Azimuthal_Equidistant"],'
                     'PARAMETER["latitude_of_center",53],'
                     'PARAMETER["longitude_of_center",24],'
                     'PARAMETER["false_easting",5837287.81977],'
                     'PARAMETER["false_northing",2121415.69617],'
                     'UNIT["metre",1,AUTHORITY["EPSG","9001"]]]')

        with GeoTiffFile(self.filepath, mode='w', geotrans=geotrans_ref, sref_wkt=sref) as src:
            src.write(data)

        with GeoTiffFile(self.filepath) as src:
            geotrans_val = src.geotrans

        self.assertEqual(geotrans_val, geotrans_ref)

    def tearDown(self):
        """
        Remove test file.
        """
        os.remove(self.filepath)


class GeoTiffDataTest(unittest.TestCase):
    """
    Testing a GeoTIFF image stack.
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
        band_name = 'mosaic'

        ds_tile = Tile(ysize, xsize, SpatialRef(4326))
        ds_mosaic = MosaicGeometry.from_tile_list([ds_tile])

        dims = ['time', 'y', 'x']
        coords = {'time': pd.date_range('2000-01-01', periods=num_files),
                  'y': ds_tile.y_coords,
                  'x': ds_tile.x_coords}
        data = np.random.randn(num_files, ysize, xsize)
        ds = xr.Dataset({band_name: (dims, data)}, coords=coords)

        with GeoTiffWriter(ds_mosaic, data=ds, file_dimension='time', dirpath=self.path,
                           fn_pattern='{time}.tif', fn_formatter={'time': lambda x: x.strftime('%Y%m%d')}) as gt_writer:
            gt_writer.export()
            filepaths = list(gt_writer.file_register['filepath'])

        with GeoTiffReader.from_filepaths(filepaths) as gt_reader:
            ts1 = gt_reader.select_px_window(0, 0, width=10, height=10, inplace=False)
            ts1.read(band_names=(band_name,))
            ts2 = gt_reader.select_px_window(10, 12, width=5, height=5, inplace=False)
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
        band_names = ['data1', 'data2']

        ds_tile = Tile(ysize, xsize, SpatialRef(4326))
        ds_mosaic = MosaicGeometry.from_tile_list([ds_tile])

        dims = ['time', 'y', 'x']
        coords = {'time': pd.date_range('2000-01-01', periods=num_files),
                  'y': ds_tile.y_coords,
                  'x': ds_tile.x_coords}
        data = np.ones((num_files, ysize, xsize))
        attr1 = {'unit': 'dB', 'scale_factor': 2, 'add_offset': 3, 'fill_value': -9999}
        attr2 = {'unit': 'dB', 'fill_value': -9999}
        ds = xr.Dataset({band_names[0]: (dims, data, attr1), band_names[1]: (dims, data, attr2)}, coords=coords)

        with GeoTiffWriter(ds_mosaic, data=ds, file_dimension='time', dirpath=self.path,
                           fn_pattern='{time}.tif', fn_formatter={'time': lambda x: x.strftime('%Y%m%d')}) as gt_writer:
            gt_writer.export()
            filepaths = list(gt_writer.file_register['filepath'])

        with GeoTiffReader.from_filepaths(filepaths) as gt_reader:
            ts1 = gt_reader.select_px_window(0, 0, width=10, height=10, inplace=False)
            ts1.read(bands=(1, 2), band_names=band_names, auto_decode=True)
            ts2 = gt_reader.select_px_window(10, 12, width=5, height=5, inplace=False)
            ts2.read(bands=(1, 2), band_names=band_names, auto_decode=True)
            np.testing.assert_equal(ts1.data[band_names[1]], data[:, :10, :10])
            np.testing.assert_equal(ts2.data[band_names[1]], data[:, 10:15, 12:17])

        with GeoTiffReader.from_filepaths(filepaths) as gt_reader:
            ts1 = gt_reader.select_px_window(0, 0, width=10, height=10, inplace=False)
            ts1.read(bands=(1, 2), band_names=band_names, auto_decode=False)
            ts2 = gt_reader.select_px_window(10, 12, width=5, height=5, inplace=False)
            ts2.read(bands=(1, 2), band_names=band_names, auto_decode=False)
            np.testing.assert_equal(ts1.data[band_names[0]], data[:, :10, :10])
            np.testing.assert_equal(ts2.data[band_names[0]], data[:, 10:15, 12:17])

        data = data * 2. + 3.
        with GeoTiffReader.from_filepaths(filepaths) as gt_reader:
            ts1 = gt_reader.select_px_window(0, 0, width=10, height=10, inplace=False)
            ts1.read(bands=(1, 2), band_names=band_names, auto_decode=True)
            ts2 = gt_reader.select_px_window(10, 12, width=5, height=5, inplace=False)
            ts2.read(bands=(1, 2), band_names=band_names, auto_decode=True)
            np.testing.assert_equal(ts1.data[band_names[0]], data[:, :10, :10])
            np.testing.assert_equal(ts2.data[band_names[0]], data[:, 10:15, 12:17])

    def test_write_selections(self):
        """ Tests writing mosaic after some select operations have been applied. """
        num_files = 10
        xsize = 60
        ysize = 50
        layer_ids = [0, 5, 9]
        band_name = 'mosaic'

        ds_tile = Tile(ysize, xsize, SpatialRef(4326), name='0')
        ds_mosaic = MosaicGeometry.from_tile_list([ds_tile])

        dims = ['time', 'y', 'x']
        dates = pd.date_range('2000-01-01', periods=num_files)
        coords = {'time': dates,
                  'y': ds_tile.y_coords,
                  'x': ds_tile.x_coords}
        data = np.random.randn(num_files, ysize, xsize)
        ds = xr.Dataset({band_name: (dims, data)}, coords=coords)
        layers = dates[layer_ids]

        with GeoTiffWriter(ds_mosaic, data=ds, file_dimension='time', dirpath=self.path,
                           fn_pattern='{time}.tif', fn_formatter={'time': lambda x: x.strftime('%Y%m%d')}) as gt_writer:
            gt_writer.select_layers(layers, inplace=True)
            ul_data = gt_writer.select_px_window(0, 0, height=25, width=30)
            ur_data = gt_writer.select_px_window(0, 30, height=25, width=30)
            ll_data = gt_writer.select_px_window(25, 0, height=25, width=30)
            lr_data = gt_writer.select_px_window(25, 30, height=25, width=30)
            gt_writer.write(ul_data.data)
            gt_writer.write(ur_data.data)
            gt_writer.write(ll_data.data)
            gt_writer.write(lr_data.data)
            filepaths = list(gt_writer.file_register['filepath'])

        with GeoTiffReader.from_filepaths(filepaths) as gt_reader:
            ts1 = gt_reader.select_px_window(0, 0, width=10, height=10, inplace=False)
            ts1.read(band_names=band_name)
            ts2 = gt_reader.select_px_window(45, 55, width=5, height=5, inplace=False)
            ts2.read(band_names=band_name)
            np.testing.assert_equal(ts1.data[band_name], data[layer_ids, :10, :10])
            np.testing.assert_equal(ts2.data[band_name], data[layer_ids, 45:, 55:])

    def tearDown(self):
        """
        Remove test file.
        """
        shutil.rmtree(self.path)


if __name__ == '__main__':
    unittest.main()