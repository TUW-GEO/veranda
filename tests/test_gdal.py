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
Test read/write of gdalport.
"""

import os
import unittest
import ogr, osr


class GdalTest(unittest.TestCase):

    def setUp(self):
        """
        Set up dummy data set.
        """
        self.wkt = ('PROJCS["Azimuthal_Equidistant",GEOGCS["WGS 84",'
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

    def test_biproj_point(self):
        """
        Test write and read.
        """
        lon = 16.3695
        lat = 48.2058

        # transform point from LonLat system to other system
        point_ori = ogr.Geometry(ogr.wkbPoint)
        point_ori.AddPoint(lon, lat)

        spref_src = osr.SpatialReference()
        spref_src.ImportFromEPSG(4326)
        spref_tar = osr.SpatialReference()
        spref_tar.ImportFromWkt(self.wkt)

        coord_traffo = osr.CoordinateTransformation(spref_src, spref_src)
        point_ori.Transform(coord_traffo)

        # transform projected point from other system to LonLat system
        point_proj = ogr.Geometry(ogr.wkbPoint)
        point_proj.AddPoint(point_ori.GetX(), point_ori.GetY())

        spref_src = osr.SpatialReference()
        spref_src.ImportFromWkt(self.wkt)
        spref_tar = osr.SpatialReference()
        spref_tar.ImportFromEPSG(4326)

        coord_traffo = osr.CoordinateTransformation(spref_src, spref_src)
        point_proj.Transform(coord_traffo)

        assert point_proj.GetX() == lon
        assert point_proj.GetY() == lat

    def tearDown(self):
        """
        Remove dummy files.
        """
        pass


if __name__ == '__main__':
    unittest.main()
