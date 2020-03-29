import os
import ogr
import shutil
import unittest
import random
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs

from shapely.geometry import Polygon

from geospade.spatial_ref import SpatialRef
from geospade.operation import xy2ij
from geospade.operation import coordinate_traffo
from geospade.operation import rel_extent

from veranda.io.geotiff import GeoTiffFile
from veranda.io.netcdf import NcFile
from veranda.io.timestack import GeoTiffRasterTimeStack
from veranda.io.timestack import NcRasterTimeStack
from veranda.raster import RasterLayer
from veranda.raster import RasterStack

# TODO: join common parts of RasterLayer and RasterStack

tmp_dirpath = "tmp"

class RasterLayerTest(unittest.TestCase):
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

        self.x_pixel_size = 0.01
        self.y_pixel_size = -0.01
        self.rows = 800
        self.cols = 800
        ul_x = 0.
        ul_y = 60. + self.y_pixel_size
        lr_x = ul_x + (self.cols-1) * self.x_pixel_size
        lr_y = ul_y + (self.rows-1) * self.y_pixel_size

        self.sref = SpatialRef(4326)
        self.label = "B01"
        self.gt = (ul_x, self.x_pixel_size, 0, ul_y, 0, self.y_pixel_size)

        self.extent = (ul_x, lr_y, lr_x, ul_y)
        self.coords = (random.uniform(ul_x, lr_x), random.uniform(ul_y, lr_y))
        self.pixels = xy2ij(self.coords[0], self.coords[1], self.gt)

        # create numpy and xarray data
        self.np_data = np.random.rand(self.rows, self.cols)
        xs = np.arange(ul_x, lr_x + self.x_pixel_size, self.x_pixel_size).tolist()
        ys = np.arange(ul_y, lr_y + self.y_pixel_size, self.y_pixel_size).tolist()
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
            gt_file = GeoTiffFile(self.gt_filepath_read, mode='w', geotrans=self.gt, sref=self.sref.wkt,
                                  count=1)
            gt_file.write(self.np_data, band=1)
            gt_file.close()
        if not os.path.exists(self.nc_filepath_read):
            nc_file = NcFile(self.nc_filepath_read, mode='w', geotrans=self.gt, sref=self.sref.wkt)
            nc_file.write(self.xr_data)
            nc_file.close()

        # create raster layer object
        self.np_raster_layer = RasterLayer(self.rows, self.cols, self.sref, self.gt, data=self.np_data,
                                          data_type="numpy", label=self.label)
        self.xr_raster_layer = RasterLayer(self.rows, self.cols, self.sref, self.gt, data=self.xr_data,
                                          data_type="xarray", label=self.label)

    def test_from_array(self):
        """ Tests creation of a raster data object from an array. """

        # Test creation from a NumPy array
        raster_layer = RasterLayer.from_array(self.sref, self.gt, self.np_data)

        assert np.all(raster_layer._data == self.np_data)

        # Test creation from an xarray
        raster_layer = RasterLayer.from_array(self.sref, self.gt, self.xr_data, label=self.label)

        assert np.all(raster_layer._data == self.xr_data)

    def test_from_filepath(self):
        """ Tests creation of a raster data object from a file path. """

        # Test creation from a GeoTIFF file
        raster_layer = RasterLayer.from_file(self.gt_filepath_read, read=True)
        assert np.all(raster_layer._data == self.np_data)

        # Test creation from a NetCDF file
        raster_layer = RasterLayer.from_file(self.nc_filepath_read, read=True, label=self.label)
        xr_data = raster_layer._data.drop('proj_unknown')  # drop projection info variable
        assert xr_data.equals(self.xr_data)

    def test_from_io(self):
        """ Tests creation of a raster data object from an io instance. """

        # Test creation from GeoTiFF class instance
        io_instance = GeoTiffFile(self.gt_filepath_read, mode='r')
        raster_layer = RasterLayer.from_io(io_instance, read=True)
        assert np.all(raster_layer._data == self.np_data)

        # Test creation from NetCDF class instance
        io_instance = NcFile(self.nc_filepath_read, mode='r')
        raster_layer = RasterLayer.from_io(io_instance, read=True)
        assert np.all(raster_layer._data == self.xr_data)

    def test_data(self):
        """ Tests data property and data type conversions. """

        # create the raster data object with the data as a numpy array from an xarray
        raster_layer_np = RasterLayer(self.rows, self.cols, self.sref, self.gt, data=self.xr_data, label=self.label,
                                    data_type="numpy")

        # create the raster data object with the data as an xarray from a numpy array
        raster_layer_xr = RasterLayer(self.rows, self.cols, self.sref, self.gt, data=self.np_data, label=self.label,
                                    data_type="xarray")

        assert np.all(raster_layer_np.data == self.np_data)
        assert raster_layer_xr.data.equals(self.xr_data)

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

        min_col, min_row = xy2ij(extent_shrnkd[0], extent_shrnkd[3], self.gt)
        max_col, max_row = xy2ij(extent_shrnkd[2], extent_shrnkd[1], self.gt)

        # test with numpy array
        raster_layer_crpd = self.np_raster_layer.crop(extent_shrnkd, inplace=False)
        assert np.all(raster_layer_crpd.data == self.np_raster_layer.data[min_row:max_row, min_col:max_col])

        # test with xarray
        raster_layer_crpd = self.xr_raster_layer.crop(extent_shrnkd, inplace=False)
        assert raster_layer_crpd.data.equals(self.xr_raster_layer.data[self.label][min_row:max_row, min_col:max_col].to_dataset())

    def test_load(self):
        """ Tests loading of full data. """

        # test load function for GeoTiff files
        raster_layer = RasterLayer.from_filepath(self.gt_filepath_read)
        raster_layer.load(data_type="numpy", inplace=True)
        assert np.all(raster_layer.data == self.np_data)

        # test load function for NetCDF files
        raster_layer = RasterLayer.from_filepath(self.nc_filepath_read, label="B01")
        raster_layer.load(data_type="xarray", inplace=True)
        assert raster_layer.data.equals(self.xr_data)

    def test_read_by_coords(self):
        """ Tests reading data by coordinates. """

        pixel_value_ref = self.np_data[self.pixels[1], self.pixels[0]]

        # test reading by coordinates
        raster_layer = RasterLayer.from_filepath(self.gt_filepath_read)
        pixel_value = raster_layer.read_by_coords(self.coords[0], self.coords[1], data_type="numpy")
        assert pixel_value == pixel_value_ref

        # test by reading coordinates of data, which is already loaded
        pixel_value = self.np_raster_layer.read_by_coords(self.coords[0], self.coords[1], data_type="numpy")
        assert pixel_value == pixel_value_ref

        # test reading by coordinates with inplace=True
        raster_layer = RasterLayer.from_filepath(self.gt_filepath_read, read=True)
        _ = raster_layer.read_by_coords(self.coords[0], self.coords[1], data_type="numpy", inplace=True)
        assert raster_layer.data == pixel_value_ref

        # test reading data by coordinates from different spatial reference system
        x_merc, y_merc = coordinate_traffo(self.coords[0], self.coords[1], SpatialRef(4326), SpatialRef(3857))
        pixel_value = self.np_raster_layer.read_by_coords(x_merc, y_merc, sref=SpatialRef(3857), data_type="numpy")
        assert pixel_value == pixel_value_ref

        # test reading coordinates with different data type
        raster_layer = RasterLayer.from_filepath(self.nc_filepath_read, label='B01')
        pixel_value = raster_layer.read_by_coords(self.coords[0], self.coords[1], data_type="numpy")
        assert pixel_value == pixel_value_ref

    def test_read_by_geom(self):
        """ Tests reading data by a given geometry. """

        # prepare random geometry (only extents are limited to a certain, disjunct region)
        ul_x, lr_y, lr_x, ul_y = self.extent
        ul_point_x = random.uniform(lr_x - self.x_pixel_size * self.cols / 2, lr_x)
        ur_point_x = random.uniform(lr_x, lr_x + self.x_pixel_size * self.cols / 2)
        lr_point_x = random.uniform(lr_x, lr_x + self.x_pixel_size * self.cols / 2)
        ll_point_x = random.uniform(lr_x - self.x_pixel_size * self.cols / 2, lr_x)
        ul_point_y = random.uniform(lr_y + abs(self.y_pixel_size) * self.rows / 2, lr_y)
        ur_point_y = random.uniform(lr_y + abs(self.y_pixel_size) * self.rows / 2, lr_y)
        lr_point_y = random.uniform(lr_y, lr_y - abs(self.y_pixel_size) * self.rows / 2)
        ll_point_y = random.uniform(lr_y, lr_y - abs(self.y_pixel_size) * self.rows / 2)
        geom = Polygon([(ul_point_x, ul_point_y), (ur_point_x, ur_point_y), (lr_point_x, lr_point_y),
                        (ll_point_x, ll_point_y), (ul_point_x, ul_point_y)])
        geom_ogr = ogr.CreateGeometryFromWkt(geom.wkt)
        geom_ogr.AssignSpatialReference(self.sref.osr_sref)

        # use RasterGeometry functionalities for setting up the mask and get the intersection boundaries
        new_geom = self.np_raster_layer.geometry & geom_ogr

        min_col, min_row, _, _ = rel_extent((self.np_raster_layer.geometry.parent_root.ul_x,
                                             self.np_raster_layer.geometry.parent_root.ul_y),
                                            new_geom.inner_extent,
                                            x_pixel_size=self.x_pixel_size,
                                            y_pixel_size=self.y_pixel_size)
        mask = self.np_raster_layer.geometry.create_mask(geom_ogr)
        pixel_values_ref = self.np_data[min_row:self.rows, min_col:self.cols]
        pixel_values_masked_ref = np.ma.array(pixel_values_ref, mask=mask[min_row:self.rows, min_col:self.cols])

        # test reading by a geometry
        raster_layer = RasterLayer.from_filepath(self.gt_filepath_read)
        pixel_values = raster_layer.read_by_geom(geom, sref=SpatialRef(4326), data_type="numpy")
        assert np.all(pixel_values == pixel_values_ref)

        # test by reading data by a geometry with data being already loaded
        pixel_values = self.np_raster_layer.read_by_geom(geom, sref=SpatialRef(4326), data_type="numpy")
        assert np.all(pixel_values == pixel_values_ref)

        # test reading data by a geometry from different spatial reference system
        geom_ogr = ogr.CreateGeometryFromWkt(geom.wkt)
        geom_ogr.AssignSpatialReference(SpatialRef(4326).osr_sref)
        geom_ogr.TransformTo(SpatialRef(3857).osr_sref)
        pixel_values = self.np_raster_layer.read_by_geom(geom_ogr, data_type="numpy")
        assert np.all(pixel_values == pixel_values_ref)

        # test reading data by a geometry with a masked applied
        pixel_values = self.np_raster_layer.read_by_geom(geom, sref=SpatialRef(4326), data_type="numpy", apply_mask=True)
        assert np.all(pixel_values == pixel_values_masked_ref)

        # test reading from a geometry with different data type
        raster_layer = RasterLayer.from_filepath(self.nc_filepath_read, label='B01')
        pixel_values = raster_layer.read_by_geom(geom, sref=SpatialRef(4326), data_type="numpy")
        assert np.all(pixel_values == pixel_values_ref)

    def test_read_by_pixels(self):
        """ Tests reading data by pixels. """

        pixel_value_ref = self.np_data[self.pixels[1], self.pixels[0]]

        # test reading by pixel coordinates
        raster_layer = RasterLayer.from_filepath(self.gt_filepath_read)
        pixel_value = raster_layer.read_by_pixel(self.pixels[1], self.pixels[0], data_type="numpy")
        assert pixel_value == pixel_value_ref

        # test by reading coordinates of data, which is already loaded
        pixel_value = self.np_raster_layer.read_by_pixel(self.pixels[1], self.pixels[0], data_type="numpy")
        assert pixel_value == pixel_value_ref

        # test reading by coordinates with inplace=True
        raster_layer = RasterLayer.from_filepath(self.gt_filepath_read, read=True)
        _ = raster_layer.read_by_pixel(self.pixels[1], self.pixels[0], data_type="numpy", inplace=True)
        assert raster_layer.data == pixel_value_ref

        # test reading data by a pixel window
        pixel_values_ref = self.np_data[self.pixels[1]:(self.pixels[1]+10), self.pixels[0]:(self.pixels[0]+10)]
        pixel_values = self.np_raster_layer.read_by_pixel(self.pixels[1], self.pixels[0], row_size=10, col_size=10,
                                                          data_type="numpy")
        assert np.all(pixel_values == pixel_values_ref)

        # test reading coordinates with different data type
        raster_layer = RasterLayer.from_filepath(self.nc_filepath_read, label='B01')
        pixel_value = raster_layer.read_by_pixel(self.pixels[1], self.pixels[0], data_type="numpy")
        assert pixel_value == pixel_value_ref

    def test_write(self):
        """ Tests writing data. """

        # test writing GeoTIFF file with a specific band name
        self.np_raster_layer.write(self.gt_filepath_write)

        # test writing NcFile file
        self.xr_raster_layer.write(self.nc_filepath_write)

        # test writing NcFile from numpy/GeoTIFF raster data
        os.remove(self.nc_filepath_write)
        self.np_raster_layer.write(self.nc_filepath_write)

    def test_plot(self):
        """ Tests plotting the data. """

        # test plotting xarray data
        self.xr_raster_layer.plot(proj=ccrs.PlateCarree())

        # test plotting numpy data
        self.np_raster_layer.plot(proj=ccrs.PlateCarree())

        # change extent
        self.np_raster_layer.plot(proj=ccrs.PlateCarree(), proj_extent=(-10, 40, 20, 70))

    def test_wrong_data(self):
        """ Tests `RasterLayer` class initialisation with a wrong data object. """

        # test wrong data type
        try:
            _ = RasterLayer.from_array(self.sref, self.gt, 1)
            assert False
        except:
            assert True

        # test wrong array dimension
        np_data = self.np_data[None, :, :]
        try:
            _ = RasterLayer.from_array(self.sref, self.gt, np_data)
            assert False
        except:
            assert True


class RasterStackTest(unittest.TestCase):
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

        self.x_pixel_size = 0.01
        self.y_pixel_size = -0.01
        self.rows = 800
        self.cols = 800
        self.layers = 5
        ul_x = 0.
        ul_y = 60. + self.y_pixel_size
        lr_x = ul_x + (self.cols-1) * self.x_pixel_size
        lr_y = ul_y + (self.rows-1) * self.y_pixel_size

        self.sref = SpatialRef(4326)
        self.label = "B01"
        self.gt = (ul_x, self.x_pixel_size, 0, ul_y, 0, self.y_pixel_size)

        self.extent = (ul_x, lr_y, lr_x, ul_y)
        self.coords = (random.uniform(ul_x, lr_x), random.uniform(ul_y, lr_y))
        self.pixels = xy2ij(self.coords[0], self.coords[1], self.gt)

        # create 3D numpy and xarray data
        self.np_data = np.zeros((self.layers, self.rows, self.cols))
        for i in range(self.layers):
            self.np_data[i, :, :] = np.random.rand(self.rows, self.cols)
        xs = np.arange(ul_x, lr_x + self.x_pixel_size, self.x_pixel_size).tolist()
        ys = np.arange(ul_y, lr_y + self.y_pixel_size, self.y_pixel_size).tolist()

        xr_dar = xr.DataArray(data=self.np_data, coords={'time': list(range(self.layers)), 'y': ys, 'x': xs},
                              dims=['time', 'y', 'x'])
        self.xr_data = xr.Dataset({self.label: xr_dar})

        # create data file paths
        self.gt_filepaths_read = [os.path.join(tmp_dirpath, "test_read_{}.tiff".format(i)) for i in range(self.layers)]
        self.gt_filepaths_write = [os.path.join(tmp_dirpath, "test_write_{}.tiff".format(i)) for i in range(self.layers)]
        self.nc_filepaths_read = [os.path.join(tmp_dirpath, "test_read_{}.nc".format(i)) for i in range(self.layers)]
        self.nc_filepaths_write = [os.path.join(tmp_dirpath, "test_write_{}.nc".format(i)) for i in range(self.layers)]

        # create files on disk
        gt_raster_layers = []
        for i, filepath in enumerate(self.gt_filepaths_read):
            if not os.path.exists(filepath):
                gt_file = GeoTiffFile(filepath, mode='w', geotrans=self.gt, sref=self.sref.wkt, count=1)
                gt_file.write(self.np_data[i], band=1)
                gt_file.close()

            raster_layer = RasterLayer.from_filepath(filepath)
            gt_raster_layers.append(raster_layer)

        nc_raster_layers = []
        for i, filepath in enumerate(self.nc_filepaths_read):
            if not os.path.exists(filepath):
                nc_file = NcFile(filepath, mode='w', geotrans=self.gt, sref=self.sref.wkt)
                nc_file.write(self.xr_data[self.label][i, :, :].to_dataset().drop('time'))
                nc_file.close()

            raster_layer = RasterLayer.from_filepath(filepath)
            nc_raster_layers.append(raster_layer)

        # create raster layer object
        self.np_raster_stack = RasterStack(gt_raster_layers, data=self.np_data, data_type="numpy", label=self.label)
        self.xr_raster_stack = RasterStack(gt_raster_layers, data=self.xr_data, data_type="xarray", label=self.label)

    def test_from_filepaths(self):
        """ Test creating a raster stack object from a list of filepaths. """

        # Test creation from a GeoTIFF files
        raster_stack = RasterStack.from_filepath(self.gt_filepaths_read, read=True)
        assert np.all(raster_stack._data == self.np_data)

        # Test creation from a NetCDF file
        raster_stack = RasterLayer.from_file(self.nc_filepaths_read, read=True, label=self.label)
        xr_data = raster_stack._data.drop('proj_unknown')  # drop projection info variable
        assert xr_data.equals(self.xr_data)

    def test_from_array(self):
        """ Tests creation of a raster stack object from an array. """

        # Test creation from a NumPy array
        raster_stack = RasterStack.from_array(self.sref, self.gt, self.np_data)

        assert np.all(raster_stack._data == self.np_data)

        # Test creation from an xarray
        raster_stack = RasterStack.from_array(self.sref, self.gt, self.xr_data, label=self.label)

        assert np.all(raster_stack._data == self.xr_data)

    def test_from_io(self):
        """ Tests creation of a raster stack object from an io instance. """

        # Test creation from GeoTiFF class instance
        filepath_df = pd.DataFrame({'filepath': self.gt_filepaths_read})
        io_instance = GeoTiffRasterTimeStack(file_ts=filepath_df, mode='r')
        raster_stack = RasterStack.from_io(io_instance, read=True)
        assert np.all(raster_stack._data == self.np_data)

        # Test creation from NetCDF class instance
        filepath_df = pd.DataFrame({'filepath': self.nc_filepaths_read})
        io_instance = NcRasterTimeStack(file_ts=filepath_df, mode='r')
        raster_stack = RasterLayer.from_io(io_instance, read=True)
        assert np.all(raster_stack._data == self.xr_data)

    def test_write(self):
        """ Tests writing data. """

        # test writing GeoTIFF files with a specific band name
        self.np_raster_stack.write(self.gt_filepaths_write)

        # test writing NcFile files
        self.xr_raster_stack.write(self.xr_filepath_write)

        # test writing NetCDF files from numpy/GeoTIFF raster data
        self.np_raster_stack.write(self.nc_filepaths_write, write_kwargs={})

    def test_plot(self):
        """ Tests plotting a raster stack on an interactive map plot. """

        # test common plot with specific layer id
        self.np_raster_stack.plot(layer_id=2, proj=ccrs.PlateCarree(), proj_extent=(-10, 40, 20, 70))

        # test interactive
        self.np_raster_stack.plot(proj=ccrs.PlateCarree(), proj_extent=(-10, 40, 20, 70), interactive=True)
        pass

    def test_wrong_data(self):
        """ Tests `RasterStack` class initialisation with a wrong data object. """

        # test wrong data type
        try:
            _ = RasterStack.from_array(self.sref, self.gt, 1)
            assert False
        except:
            assert True

        # test wrong array dimension
        np_data = self.np_data[0, :, :]
        try:
            _ = RasterStack.from_array(self.sref, self.gt, np_data)
            assert False
        except:
            assert True



if __name__ == '__main__':
    #unittest.main()
    # tester = RasterLayerTest()
    # tester.setUpClass()
    # tester.setUp()
    # tester.test_from_array()
    # tester.setUp()
    # tester.test_from_filepath()
    # tester.setUp()
    # tester.test_from_io()
    # tester.setUp()
    # tester.test_data()
    # tester.setUp()
    # tester.test_crop()
    # tester.setUp()
    # tester.test_load()
    # tester.setUp()
    # tester.test_read_by_coords()
    # tester.setUp()
    # tester.test_read_by_geom()
    # tester.setUp()
    # tester.test_read_by_pixels()
    # tester.setUp()
    # tester.test_plot()
    # tester.setUp()
    # tester.test_write()
    # tester.setUp()
    # tester.test_wrong_data()
    tester = RasterStackTest()
    tester.setUp()
    tester.test_plot()
