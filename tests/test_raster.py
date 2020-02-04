import os
import shutil
import unittest
import numpy as np
import xarray as xr
import cartopy.crs as ccrs

from geospade.spatial_ref import SpatialRef
from veranda.io.geotiff import GeoTiffFile
from veranda.io.netcdf import NcFile
from veranda.raster import RasterData

tmp_dirpath = "tmp"

class RasterDataTest(unittest.TestCase):
    """ Testing all functionalities of `RasterData`. """

    def setUpClass(cls):
        """ Creates temporary data folder. """
        if not os.path.exists(tmp_dirpath):
            os.makedirs(tmp_dirpath)

    def tearDownClass(cls):
        """ Deletes temporary data folder. """
        if os.path.exists(tmp_dirpath):
            shutil.rmtree(tmp_dirpath)

    def setUp(self):
        """
        Sets up needed test data.
        """

        x_pixel_size = 0.01
        y_pixel_size = -0.01
        self.rows = 800
        self.cols = 800
        ul_x = 0.
        ul_y = 60. + y_pixel_size
        lr_x = ul_x + self.cols * x_pixel_size
        lr_y = ul_y + self.rows * y_pixel_size

        self.sref = SpatialRef(4326)
        self.label = "B01"
        self.gt = (ul_x, x_pixel_size, 0, ul_y, 0, y_pixel_size)
        self.roi = [(ul_x, lr_y), (lr_x, ul_y)]

        # create numpy and xarray data
        self.np_data = np.random.rand(self.rows, self.cols)
        xs = np.arange(ul_x, lr_x, x_pixel_size).tolist()
        ys = np.arange(ul_y, lr_y, y_pixel_size).tolist()
        # TODO: ask Shahn about variables and dimensions. Allow to put coordinates from outside
        xr_dar = xr.DataArray(data=self.np_data, dims=['y', 'x'])
        self.xr_data = xr.Dataset({self.label: xr_dar})

        # create data file paths
        self.gt_filepath_read = os.path.join(tmp_dirpath, "test_read.tiff")
        self.gt_filepath_write = os.path.join(tmp_dirpath, "test_write.tiff")
        self.nc_filepath_read = os.path.join(tmp_dirpath, "test_read.nc")
        self.nc_filepath_write = os.path.join(tmp_dirpath, "test_write.nc")

        # create files on disk
        if not os.path.exists(self.gt_filepath_read):
            gt_file = GeoTiffFile(self.gt_filepath_read, mode='w', geotransform=self.gt, spatialref=self.sref.wkt,
                                  count=1)
            gt_file.write(self.np_data, band=1)
        if not os.path.exists(self.nc_filepath_read):
            nc_file = NcFile(self.nc_filepath_read, mode='w', geotransform=self.gt, spatialref=self.sref.wkt)
            nc_file.write(self.xr_data)

        # create raster data object
        #raster_data = RasterData(self.rows, self.cols, self.sref, self.gt, data=self.np_data, label=self.label)

    def test_create_raster_data(self):
        """ Tests creation of a raster data object from general input parameters. """
        raster_data = RasterData(self.rows, self.cols, self.sref, self.gt, data=self.np_data, label=self.label)

    def test_from_array(self):
        """ Tests creation of a raster data object from an array. """

        # Test creation from a NumPy array
        raster_data = RasterData.from_array(self.sref, self.gt, self.np_data)

        # Test creation from an Xarray
        raster_data = RasterData.from_array(self.sref, self.gt, self.xr_data)

    def test_from_file(self):
        """ Tests creation of a raster data object from a file. """

        # Test creation from a GeoTIFF file
        raster_data = RasterData.from_file(self.gt_filepath_read)

        # Test creation from a NetCDF file
        raster_data = RasterData.from_file(self.nc_filepath_read)

    def test_data(self):
        """ Tests data property and data type conversions. """

        # create the raster data object with the data as a numpy array from an xarray
        raster_data_np = RasterData(self.rows, self.cols, self.sref, self.gt, data=self.xr_data, label=self.label,
                                    data_type="numpy")

        # create the raster data object with the data as an xarray from a numpy array
        raster_data_xr = RasterData(self.rows, self.cols, self.sref, self.gt, data=self.np_data, label=self.label,
                                    data_type="xarray")

        assert raster_data_np.data == self.np_data
        assert raster_data_xr.data == self.xr_data


    def test_crop(self):
        """ Tests cropping of the raster data instance. """

    def test_load(self):
        """ Tests loading of full data. """

    def test_read_by_coords(self):
        """ Tests reading data by coordinates. """

        # test reading by coordinates
        # test by reading coordinates of data, which is already loaded
        # test reading data by coordinates from different spatial reference system


    def test_read_by_geom(self):
        """ Tests reading data by a given geometry. """

        # test reading by a geometry
        # test by reading data by a geometry with data being already loaded
        # test reading data by a geometry from different spatial reference system

    def test_read_by_pixels(self):
        """ Tests reading data by pixels. """

        # test reading by pixels
        # test by reading data by pixels, which is already loaded

    def test_write(self):
        """ Tests writing data. """

    def test_plot(self):
        """ Tests plotting the data. """

        # test plotting xarray data
        raster_data = RasterData.from_array(self.sref, self.gt, self.xr_data)
        raster_data.plot(proj=ccrs.PlateCarree())

        # test plotting numpy data
        raster_data = RasterData.from_array(self.sref, self.gt, self.np_data)
        raster_data.plot(proj=ccrs.PlateCarree())

        # change extent


    def test_wrong_data(self):
        """ Tests `RasterData` class initialisation with a wrong data object. """

        # test wrong data type
        try:
            _ = RasterData.from_array(self.sref, self.gt, 1)
            assert False
        except:
            assert True

        # test wrong array dimension
        np_data = self.np_data[None, :, :]
        try:
            _ = RasterData.from_array(self.sref, self.gt, np_data)
            assert False
        except:
            assert True

if __name__ == '__main__':
    #unittest.main()
    tester = RasterDataTest()
    tester.setUpClass()
    tester.setUp()
    tester.test_create_raster_data()
    tester.setUp()
    tester.test_from_array()