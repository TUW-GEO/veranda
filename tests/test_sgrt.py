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
Test sgrt functions.
"""

import unittest

from pyraster.sgrt.SgrtTile import SgrtTile


class SgrtTileTest(unittest.TestCase):

    """
    Testing i/o of geotiff.
    """

    def setUp(self):
        pass

    def test_sgrt_tile(self):
        e7tile = 'EU500M_E048N012T6'
        tl = SgrtTile('', product_id='SSSGIOGL_T020', wflow_id='C0201',
                      ftile=e7tile, ptop_name='asdf')

if __name__ == '__main__':
    unittest.main()
