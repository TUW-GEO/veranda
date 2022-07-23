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
Test operations in the `gdalport` module.
"""

import os
import unittest

import numpy as np

from veranda.raster.gdalport import call_gdal_util
from veranda.raster.native.geotiff import GeoTiffFile


class GdalportTest(unittest.TestCase):

    def setUp(self):

        self.data = np.ones((1, 100, 100))
        test_dir = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), 'temp')
        if not os.path.isdir(test_dir):
            os.makedirs(test_dir)
        self.filename = os.path.join(test_dir, 'test.tif')
        self.filename_gdal_trans = os.path.join(test_dir, 'test_gdal_trans.tif')

    def test_gdal_translate(self):
        with GeoTiffFile(self.filename, mode='w') as gt_file:
            gt_file.write(self.data)
        # configure options
        options = {'-of': 'GTiff', '-co': 'COMPRESS=LZW',
                   '-mo': ['parent_data_file="%s"' % os.path.basename(self.filename)], '-outsize': ('50%', '50%'),
                   '-ot': 'Byte'}
        succeed, output = call_gdal_util("gdal_translate", src_files=self.filename, dst_file=self.filename_gdal_trans,
                                         options=options)

        assert succeed

    def tearDown(self):

        if os.path.exists(self.filename):
            os.remove(self.filename)
        if os.path.exists(self.filename_gdal_trans):
            os.remove(self.filename_gdal_trans)


if __name__ == '__main__':
    unittest.main()
