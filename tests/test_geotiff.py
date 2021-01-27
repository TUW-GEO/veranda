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
Test geotiff functions.
"""

import os
import unittest
from tempfile import mkdtemp

import numpy as np

from veranda.io.geotiff import read_tiff
from veranda.io.geotiff import write_tiff
from veranda.io.geotiff import GeoTiffFile


class GeotiffTest(unittest.TestCase):

    """
    Testing i/o of geotiff.
    """

    def setUp(self):
        """
        Set up dummy data set.
        """
        self.data = np.ones((100, 100))
        test_dir = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), 'temp')
        if not os.path.isdir(test_dir):
            os.makedirs(test_dir)
        self.filename = os.path.join(test_dir, 'test.tif')

    def test_read_write_arr(self):
        """
        Test writing 2d data.
        """
        write_tiff(self.filename, src_arr=self.data)
        src_arr, tags_dict = read_tiff(self.filename)
        np.testing.assert_equal(self.data, src_arr)

    def test_read_write_rgb(self):
        """
        Test writing RGB data.
        """
        write_tiff(self.filename, red=self.data, green=self.data,
                   blue=self.data)

        # read_tiff can only handl 1-band tif files
        try:
            src_arr, tags_dict = read_tiff(self.filename)
        except IOError:
            pass

    def test_description(self):
        """
        Test description.
        """
        tags_dict = {'description': 'Hello World'}
        write_tiff(self.filename, src_arr=self.data, tags_dict=tags_dict)
        src_arr, tags = read_tiff(self.filename)

        self.assertNotEqual(tags_dict['description'], tags['description'])

    def tearDown(self):
        """
        Remove test file.
        """
        os.remove(self.filename)


class GeoTiffFileTest(unittest.TestCase):

    def setUp(self):
        """
        Set up dummy data set.
        """
        self.filepath = os.path.join(mkdtemp(), 'test.tif')

    def test_read_write_1_band(self):
        """
        Test a write and read of a single band data.
        """
        data = np.ones((1, 100, 100), dtype=np.float32)

        with GeoTiffFile(self.filepath, mode='w') as src:
            src.write(data)

        with GeoTiffFile(self.filepath) as src:
            ds = src.read()

        np.testing.assert_array_equal(ds, data[0, :, :])

    def test_read_write_multi_band(self):
        """
        Test a write and read of a multi band data.
        """
        data = np.ones((5, 100, 100), dtype=np.float32)

        with GeoTiffFile(self.filepath, mode='w') as src:
            src.write(data)

        with GeoTiffFile(self.filepath) as src:
            for band in np.arange(data.shape[0]):
                ds = src.read(band=band + 1)
                np.testing.assert_array_equal(
                    ds, data[band, :, :])

    def test_decoding_multi_band(self):
        """
        Test decoding of multi band data.

        """
        data = np.ones((5, 100, 100), dtype=np.float32)

        with GeoTiffFile(self.filename, mode='w') as src:
            src.write(data, scale_factor=[1, 2, 1, 1, 3], add_offset=[0, 3, 0, 0, 0])

        with GeoTiffFile(self.filename, auto_decode=False) as src:
            ds, tags = src.read(1)
            np.testing.assert_array_equal(ds, data[0, :, :])
            ds, tags = src.read(2)
            np.testing.assert_array_equal(ds, data[1, :, :])
            ds, tags = src.read(5)
            np.testing.assert_array_equal(ds, data[4, :, :])

        data[1, :, :] = data[1, :, :] * 2 + 3
        data[4, :, :] = data[4, :, :] * 3

        with GeoTiffFile(self.filename, auto_decode=True) as src:
            ds, tags = src.read(1)
            np.testing.assert_array_equal(ds, data[0, :, :])
            ds, tags = src.read(2)
            np.testing.assert_array_equal(ds, data[1, :, :])
            ds, tags = src.read(5)
            np.testing.assert_array_equal(ds, data[4, :, :])

    def test_read_write_specific_band(self):
        """
        Test a write and read of a specific data.
        """
        data = np.ones((100, 100), dtype=np.float32)

        with GeoTiffFile(self.filepath, mode='w', n_bands=10) as src:
            src.write(data, band=5)
            src.write(data, band=10)

        with GeoTiffFile(self.filepath) as src:
            ds = src.read(band=5)
            np.testing.assert_array_equal(ds, data)
            ds = src.read(band=10)
            np.testing.assert_array_equal(ds, data)

    def test_tags(self):
        """
        Test meta data tags.
        """
        data = np.ones((1, 100, 100), dtype=np.float32)

        metadata = {'attr1': '123', 'attr2': 'test'}
        tags = {'description': 'helloworld', 'metadata': metadata}

        with GeoTiffFile(self.filepath, mode='w', tags=tags) as src:
            src.write(data)

        with GeoTiffFile(self.filepath) as src:
            ds_tags = src.read_tags(1)

        self.assertEqual(ds_tags['metadata'], tags['metadata'])
        self.assertNotEqual(ds_tags['description'], tags['description'])

    def test_geotransform(self):
        """
        Test geotransform and spatialref keywords.

        Notes
        -----
        Spatial reference cannot be compared due to different versions of WKT1 strings.
        """
        data = np.ones((1, 100, 100), dtype=np.float32)

        geotrans = (3000000.0, 500.0, 0.0, 1800000.0, 0.0, -500.0)
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

        with GeoTiffFile(self.filepath, mode='w', geotrans=geotrans, sref=sref) as src:
            src.write(data)

        with GeoTiffFile(self.filepath) as src:
            ds_tags = src.read_tags(1)

        self.assertEqual(ds_tags['geotransform'], geotrans)

    def tearDown(self):
        """
        Remove test file.
        """
        os.remove(self.filepath)


if __name__ == '__main__':
    unittest.main()