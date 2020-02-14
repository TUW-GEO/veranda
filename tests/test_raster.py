import os
import shutil
import unittest
import numpy as np
import xarray as xr
import cartopy.crs as ccrs

from geospade.spatial_ref import SpatialRef
from geospade.definition import RasterGeometry
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

        self.tearDownClass()
        self.setUpClass()

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

        self.extent = (ul_x, lr_y, lr_x, ul_y)

        # create numpy and xarray data
        self.np_data = np.random.rand(self.rows, self.cols)
        xs = np.arange(ul_x, lr_x, x_pixel_size).tolist()
        ys = np.arange(ul_y, lr_y, y_pixel_size).tolist()
        # TODO: ask Shahn about variables and dimensions. Allow to put coordinates from outside
        xr_dar = xr.DataArray(data=self.np_data, coords={'x': xs, 'y': ys}, dims=['y', 'x'])
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
            gt_file.close()
        if not os.path.exists(self.nc_filepath_read):
            nc_file = NcFile(self.nc_filepath_read, mode='w', geotransform=self.gt, spatialref=self.sref.wkt)
            nc_file.write(self.xr_data)
            nc_file.close()

        # create raster data object
        self.raster_data = RasterData(self.rows, self.cols, self.sref, self.gt, data=self.np_data, label=self.label)

    def test_from_array(self):
        """ Tests creation of a raster data object from an array. """

        # Test creation from a NumPy array
        raster_data = RasterData.from_array(self.sref, self.gt, self.np_data)

        assert np.all(raster_data._data == self.np_data)

        # Test creation from an Xarray
        raster_data = RasterData.from_array(self.sref, self.gt, self.xr_data, label=self.label)

        assert np.all(raster_data._data == self.xr_data)

    def test_from_filepath(self):
        """ Tests creation of a raster data object from a file path. """

        # Test creation from a GeoTIFF file
        raster_data = RasterData.from_file(self.gt_filepath_read, read=True)
        assert np.all(raster_data._data == self.np_data)

        # Test creation from a NetCDF file
        raster_data = RasterData.from_file(self.nc_filepath_read, read=True, label=self.label)
        xr_data = raster_data._data.drop('proj_unknown')  # drop projection info variable
        assert xr_data.equals(self.xr_data)

    def test_from_io(self):
        """ Tests creation of a raster data object from an io instance. """
        pass

    def test_data(self):
        """ Tests data property and data type conversions. """

        # create the raster data object with the data as a numpy array from an xarray
        raster_data_np = RasterData(self.rows, self.cols, self.sref, self.gt, data=self.xr_data, label=self.label,
                                    data_type="numpy")

        # create the raster data object with the data as an xarray from a numpy array
        raster_data_xr = RasterData(self.rows, self.cols, self.sref, self.gt, data=self.np_data, label=self.label,
                                    data_type="xarray")

        assert np.all(raster_data_np.data == self.np_data)
        assert raster_data_xr.data.equals(self.xr_data)

    def test_crop(self):
        """
        Tests cropping of the raster data instance.
        Implicitly, the `_read_array` function is tested here for both array types.
        """

        # define crop region
        extent_width = self.extent[2] - self.extent[0]
        extent_height = self.extent[3] - self.extent[1]
        extent_shrnkd = (self.extent[0] + extent_width / 4.,
                         self.extent[1] + extent_height / 4.,
                         self.extent[2] - extent_width / 4.,
                         self.extent[3] - extent_height / 4.)

        min_row = int(self.rows / 4.)
        max_row = int(self.rows - self.rows / 4.)
        min_col = int(self.cols / 4.)
        max_col = int(self.cols - self.cols / 4.)

        # test with numpy array
        raster_data_crpd = self.raster_data.crop(extent_shrnkd, inplace=False)
        assert np.all(raster_data_crpd.data == self.raster_data.data[min_row:max_row, min_col:max_col])

        # test with xarray
        raster_data = RasterData.from_array(self.sref, self.gt, self.xr_data, label=self.label, data_type='xarray')
        raster_data_crpd = raster_data.crop(extent_shrnkd, inplace=False)
        assert raster_data_crpd.data.equals(raster_data.data[self.label][min_row:max_row, min_col:max_col].to_dataset())

    def test_apply_mask(self):
        """ . """
        pass

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
        raster_data.plot(proj=ccrs.PlateCarree(), proj_extent=(0, 0, 180, 90))

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
    tester.test_from_array()
    tester.setUp()
    tester.test_from_filepath()
    tester.setUp()
    tester.test_data()
    tester.setUp()
    tester.test_crop()
    tester.setUp()
    tester.test_load()
    tester.setUp()
    tester.test_read_by_coords()
    tester.setUp()
    tester.test_read_by_geom()
    tester.setUp()
    tester.test_read_by_pixels()