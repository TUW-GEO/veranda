"""
Tests all functions of a GeoTIFF driver.
"""

import os
import unittest
import numpy as np
from tempfile import mkdtemp

from veranda.raster.driver.geotiff import GeoTiffDriver


class GeoTiffDriverTest(unittest.TestCase):

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

        with GeoTiffDriver(self.filepath, mode='w') as src:
            src.write(data)

        with GeoTiffDriver(self.filepath) as src:
            ds = src.read()

        np.testing.assert_array_equal(ds[1], data[0, :, :])

    def test_read_write_multi_band(self):
        """
        Test write and read of a multi band data.
        """
        data = np.ones((5, 100, 100), dtype=np.float32)

        with GeoTiffDriver(self.filepath, mode='w', n_bands=5) as src:
            src.write(data)

        with GeoTiffDriver(self.filepath) as src:
            for band in np.arange(data.shape[0]):
                ds = src.read(bands=band + 1)
                np.testing.assert_array_equal(
                    ds[band + 1], data[band, :, :])

    def test_decoding_multi_band(self):
        """
        Test decoding of multi band data.

        """
        data = np.ones((5, 100, 100), dtype=np.float32)
        scale_factors = {1: 1, 2: 2, 3: 1, 4: 1, 5: 3}
        offsets = {2: 3}
        with GeoTiffDriver(self.filepath, mode='w', n_bands=5,
                           scale_factors=scale_factors, offsets=offsets) as src:
            src.write(data)

        with GeoTiffDriver(self.filepath, auto_decode=False) as src:
            ds = src.read(bands=1)
            np.testing.assert_array_equal(ds[1], data[0, :, :])
            ds = src.read(bands=2)
            np.testing.assert_array_equal(ds[2], data[1, :, :])
            ds = src.read(bands=5)
            np.testing.assert_array_equal(ds[5], data[4, :, :])

        data[1, :, :] = data[1, :, :] * 2 + 3
        data[4, :, :] = data[4, :, :] * 3

        with GeoTiffDriver(self.filepath, auto_decode=True) as src:
            ds = src.read(bands=1)
            np.testing.assert_array_equal(ds[1], data[0, :, :])
            ds = src.read(bands=2)
            np.testing.assert_array_equal(ds[2], data[1, :, :])
            ds = src.read(bands=5)
            np.testing.assert_array_equal(ds[5], data[4, :, :])

    def test_read_write_specific_band(self):
        """
        Test a write and read of a specific data.
        """
        data = np.ones((100, 100), dtype=np.float32)

        with GeoTiffDriver(self.filepath, mode='w', n_bands=10) as src:
            src.write({5: data})
            src.write({10: data})

        with GeoTiffDriver(self.filepath) as src:
            ds = src.read(bands=5)
            np.testing.assert_array_equal(ds[5], data)
            ds = src.read(bands=10)
            np.testing.assert_array_equal(ds[10], data)

    def test_metadata(self):
        """
        Test meta data tags.
        """
        data = np.ones((1, 100, 100), dtype=np.float32)

        metadata = {'attr1': '123', 'attr2': 'test'}

        with GeoTiffDriver(self.filepath, mode='w', metadata=metadata) as src:
            src.write(data)

        with GeoTiffDriver(self.filepath) as src:
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

        with GeoTiffDriver(self.filepath, mode='w', geotrans=geotrans_ref, sref_wkt=sref) as src:
            src.write(data)

        with GeoTiffDriver(self.filepath) as src:
            geotrans_val = src.geotrans

        self.assertEqual(geotrans_val, geotrans_ref)

    def tearDown(self):
        """
        Remove test file.
        """
        os.remove(self.filepath)


if __name__ == '__main__':
    unittest.main()