import os
import ogr
import shutil
import unittest
import random
import numpy as np
import xarray as xr
import cartopy.crs as ccrs

from shapely.geometry import Polygon

from geospade.spatial_ref import SpatialRef
from geospade.operation import xy2ij
from geospade.operation import coordinate_traffo
from geospade.operation import rel_extent

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

        self.x_pixel_size = 0.01
        self.y_pixel_size = -0.01
        self.rows = 800
        self.cols = 800
        ul_x = 0.
        ul_y = 60. + self.y_pixel_size
        lr_x = ul_x + (self.cols-1) * self.x_pixel_size  # TODO: check -1
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
            gt_file = GeoTiffFile(self.gt_filepath_read, mode='w', geotransform=self.gt, spatialref=self.sref.wkt,
                                  count=1)
            gt_file.write(self.np_data, band=1)
            gt_file.close()
        if not os.path.exists(self.nc_filepath_read):
            nc_file = NcFile(self.nc_filepath_read, mode='w', geotransform=self.gt, spatialref=self.sref.wkt)
            nc_file.write(self.xr_data)
            nc_file.close()

        # create raster data object
        self.np_raster_data = RasterData(self.rows, self.cols, self.sref, self.gt, data=self.np_data,
                                         data_type="numpy", label=self.label)
        self.xr_raster_data = RasterData(self.rows, self.cols, self.sref, self.gt, data=self.xr_data,
                                         data_type="xarray", label=self.label)

    def test_from_array(self):
        """ Tests creation of a raster data object from an array. """

        # Test creation from a NumPy array
        raster_data = RasterData.from_array(self.sref, self.gt, self.np_data)

        assert np.all(raster_data._data == self.np_data)

        # Test creation from an xarray
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

        # Test creation from GeoTiFF class instance
        io_instance = GeoTiffFile(self.gt_filepath_read, mode='r')
        raster_data = RasterData.from_io(io_instance, read=True)
        assert np.all(raster_data._data == self.np_data)

        # Test creation from NetCDF class instance
        io_instance = NcFile(self.nc_filepath_read, mode='r')
        raster_data = RasterData.from_io(io_instance, read=True)
        assert np.all(raster_data._data == self.xr_data)

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

        min_col, min_row = xy2ij(extent_shrnkd[0], extent_shrnkd[3], self.gt)
        max_col, max_row = xy2ij(extent_shrnkd[2], extent_shrnkd[1], self.gt)
        # -1 because because extent goes from the ul to the lr pixel corner
        max_col, max_row = max_col - 1, max_row - 1

        # test with numpy array
        raster_data_crpd = self.np_raster_data.crop(extent_shrnkd, inplace=False)
        assert np.all(raster_data_crpd.data == self.np_raster_data.data[min_row:(max_row+1), min_col:(max_col+1)])

        # test with xarray
        raster_data_crpd = self.xr_raster_data.crop(extent_shrnkd, inplace=False)
        assert raster_data_crpd.data.equals(self.xr_raster_data.data[self.label][min_row:(max_row+1), min_col:(max_col+1)].to_dataset())

    def test_load(self):
        """ Tests loading of full data. """

        # test load function for GeoTiff files
        raster_data = RasterData.from_filepath(self.gt_filepath_read)
        raster_data.load(data_type="numpy")
        assert np.all(raster_data.data == self.np_data)

        # test load function for NetCDF files
        raster_data = RasterData.from_filepath(self.nc_filepath_read, label="B01")
        raster_data.load(data_type="xarray")
        assert raster_data.data.equals(self.xr_data)

    def test_read_by_coords(self):
        """ Tests reading data by coordinates. """

        pixel_value_ref = self.np_data[self.pixels[1], self.pixels[0]]

        # test reading by coordinates
        raster_data = RasterData.from_filepath(self.gt_filepath_read)
        pixel_value = raster_data.read_by_coords(self.coords[0], self.coords[1], data_type="numpy")
        assert pixel_value == pixel_value_ref

        # test by reading coordinates of data, which is already loaded
        pixel_value = self.np_raster_data.read_by_coords(self.coords[0], self.coords[1], data_type="numpy")
        assert pixel_value == pixel_value_ref

        # test reading by coordinates with inplace=True
        raster_data = RasterData.from_filepath(self.gt_filepath_read, read=True)
        _ = raster_data.read_by_coords(self.coords[0], self.coords[1], data_type="numpy", inplace=True)
        assert raster_data.data == pixel_value_ref

        # test reading data by coordinates from different spatial reference system
        x_merc, y_merc = coordinate_traffo(self.coords[0], self.coords[1], SpatialRef(4326), SpatialRef(3857))
        pixel_value = self.np_raster_data.read_by_coords(x_merc, y_merc, sref=SpatialRef(3857), data_type="numpy")
        assert pixel_value == pixel_value_ref

        # test reading coordinates with different data type
        raster_data = RasterData.from_filepath(self.nc_filepath_read, label='B01')
        pixel_value = raster_data.read_by_coords(self.coords[0], self.coords[1], data_type="numpy")
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
        new_geom = self.np_raster_data.geometry & geom_ogr
        min_col, _, _, min_row = rel_extent(self.np_raster_data.geometry.parent_root.extent, new_geom.extent,
                                            x_pixel_size=self.x_pixel_size, y_pixel_size=self.y_pixel_size)
        mask = self.np_raster_data.geometry.create_mask(geom_ogr)
        pixel_values_ref = self.np_data[min_row:self.rows, min_col:self.cols]
        pixel_values_masked_ref = np.ma.array(pixel_values_ref, mask=mask[min_row:self.rows, min_col:self.cols])

        # test reading by a geometry
        raster_data = RasterData.from_filepath(self.gt_filepath_read)
        pixel_values = raster_data.read_by_geom(geom, sref=SpatialRef(4326), data_type="numpy")
        assert np.all(pixel_values == pixel_values_ref)

        # test by reading data by a geometry with data being already loaded
        pixel_values = self.np_raster_data.read_by_geom(geom, sref=SpatialRef(4326), data_type="numpy")
        assert np.all(pixel_values == pixel_values_ref)

        # test reading data by a geometry from different spatial reference system
        geom_ogr = ogr.CreateGeometryFromWkt(geom.wkt)
        geom_ogr.AssignSpatialReference(SpatialRef(4326).osr_sref)
        geom_ogr.TransformTo(SpatialRef(3857).osr_sref)
        pixel_values = self.np_raster_data.read_by_geom(geom_ogr, data_type="numpy")
        assert np.all(pixel_values == pixel_values_ref)

        # test reading data by a geometry with a masked applied
        pixel_values = self.np_raster_data.read_by_geom(geom, sref=SpatialRef(4326), data_type="numpy", apply_mask=True)
        assert np.all(pixel_values == pixel_values_masked_ref)

        # test reading from a geometry with different data type
        raster_data = RasterData.from_filepath(self.nc_filepath_read, label='B01')
        pixel_values = raster_data.read_by_geom(geom, sref=SpatialRef(4326), data_type="numpy")
        assert np.all(pixel_values == pixel_values_ref)

    def test_read_by_pixels(self):
        """ Tests reading data by pixels. """

        pixel_value_ref = self.np_data[self.pixels[1], self.pixels[0]]

        # test reading by pixel coordinates
        raster_data = RasterData.from_filepath(self.gt_filepath_read)
        pixel_value = raster_data.read_by_pixel(self.pixels[1], self.pixels[0], data_type="numpy")
        assert pixel_value == pixel_value_ref

        # test by reading coordinates of data, which is already loaded
        pixel_value = self.np_raster_data.read_by_pixel(self.pixels[1], self.pixels[0], data_type="numpy")
        assert pixel_value == pixel_value_ref

        # test reading by coordinates with inplace=True
        raster_data = RasterData.from_filepath(self.gt_filepath_read, read=True)
        _ = raster_data.read_by_pixel(self.pixels[1], self.pixels[0], data_type="numpy", inplace=True)
        assert raster_data.data == pixel_value_ref

        # test reading data by a pixel window
        pixel_values_ref = self.np_data[self.pixels[1]:(self.pixels[1]+10), self.pixels[0]:(self.pixels[0]+10)]
        pixel_values = self.np_raster_data.read_by_pixel(self.pixels[1], self.pixels[0], row_size=10, col_size=10,
                                                        data_type="numpy")
        assert np.all(pixel_values == pixel_values_ref)

        # test reading coordinates with different data type
        raster_data = RasterData.from_filepath(self.nc_filepath_read, label='B01')
        pixel_value = raster_data.read_by_pixel(self.pixels[1], self.pixels[0], data_type="numpy")
        assert pixel_value == pixel_value_ref

    def test_write(self):
        """ Tests writing data. """

        # test writing GeoTIFF file with a specific band name
        #write_kwargs = {"band": self.label}
        self.np_raster_data.write(self.gt_filepath_write)

        # test writing NcFile file
        self.xr_raster_data.write(self.nc_filepath_write)

        # test writing NcFile from numpy/GeoTIFF raster data
        os.remove(self.nc_filepath_write)
        self.np_raster_data.write(self.nc_filepath_write)

    def test_plot(self):
        """ Tests plotting the data. """

        # test plotting xarray data
        self.xr_raster_data.plot(proj=ccrs.PlateCarree())

        # test plotting numpy data
        self.np_raster_data.plot(proj=ccrs.PlateCarree())

        # change extent
        self.np_raster_data.plot(proj=ccrs.PlateCarree(), proj_extent=(-10, 40, 20, 70))

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
    unittest.main()