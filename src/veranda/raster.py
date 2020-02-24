import os
import sys
import copy
import ogr
import osr
import inspect
import warnings
import shapely
import shapely.wkt
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import cm
from collections import OrderedDict
from shapely.geometry import Polygon
from shapely.geometry import Point

from veranda.io.netcdf import NcFile
from veranda.io.geotiff import GeoTiffFile
from veranda.io.timestack import GeoTiffRasterTimeStack
from veranda.io.timestack import NcRasterTimeStack

from veranda.errors import DataTypeUnknown
from veranda.errors import DataTypeMismatch

from geospade.definition import RasterGeometry
from geospade.definition import RasterGrid
from geospade.spatial_ref import SpatialRef
from geospade.operation import rasterise_polygon
from geospade.definition import _any_geom2ogr_geom
from geospade.operation import ij2xy
from geospade.operation import rel_extent
from geospade.operation import coordinate_traffo

# TODO: can we represent a rotated array with xarray?
# ToDO: how to handle Bands with Geotiff files?


# TODO: where should we put this?
def convert_data_coding(data, coder, code_kwargs=None, band=None):
    """
    Converts data values via a given coding function.
    A band needs to be given if one works with xarray data sets.

    Parameters
    ----------
    data : numpy.ndarray or xarray.Dataset, optional
        Array-like object containing image pixel values.
    coder : function
        Coding function, which expects (NumPy) arrays.
    code_kwargs : dict, optional
        Keyword arguments for the coding function.
    band : None
        Band/data variable of xarray data set.

    Returns
    -------
    numpy.ndarray or xarray.Dataset
        Array-like object containing coded image pixel values.
    """

    code_kwargs = {} if code_kwargs is None else code_kwargs

    if isinstance(data, xr.Dataset):
        if band is None:
            err_msg = "A band name has to be specified for coding the data."
            raise KeyError(err_msg)
        data[band].data = coder(data[band].data, **code_kwargs)
        return data
    elif isinstance(data, np.ndarray):
        return coder(data, **code_kwargs)
    else:
        err_msg = "Data type is not supported for coding the data."
        raise Exception(err_msg)


# TODO: where should we put this?
def convert_data_type(data, data_type, xs=None, ys=None, zs=None, band=None, x_dim='x', y_dim='y', z_dim='time'):
    """
    Converts `data` into an array-like object defined by `dtype`. It supports NumPy arrays, Xarray arrays and
    Pandas data frames.

    Parameters
    ----------
    data : numpy.ndarray or xarray.Dataset
        Array-like object containing image pixel values.
    data_type : str
        Data type of the returned array-like structure (default is 'xarray'). It can be:
            - 'xarray': loads data as an xarray.DataSet
            - 'numpy': loads data as a numpy.ndarray
            - 'dataframe': loads data as a pandas.DataFrame
    xs : list, optional
        List of world system coordinates in X direction.
    ys : list, optional
        List of world system coordinates in Y direction.
    zs : list, optional
        List of z values, preferably time stamps.
    band : int or str, optional
        Band number or name.
    x_dim : str, optional
        Name of the X dimension (needed for xarray data sets, default is 'x').
    y_dim : str, optional
        Name of the Y dimension (needed for xarray data sets, default is 'y').
    z_dim : str, optional
        Name of the Z dimension (needed for xarray data sets, default is 'time').

    Returns
    -------
    numpy.ndarray or xarray.Dataset
        Array-like object containing image pixel values.
    """

    if data_type == "xarray":
        if isinstance(data, np.ndarray):
            coords = OrderedDict()
            if zs is not None:
                coords[z_dim] = zs
            if ys is not None:
                coords[y_dim] = ys
            if xs is not None:
                coords[x_dim] = xs
            xr_ar = xr.DataArray(data, coords=coords, dims=list(coords.keys()))
            converted_data = xr.Dataset(data_vars={band: xr_ar})
        elif isinstance(data, xr.Dataset):
            converted_data = data
        else:
            raise DataTypeMismatch(type(data), data_type)
    elif data_type == "numpy":
        if isinstance(data, xr.Dataset):
            if band is None:
                err_msg = "Band/label argument is not specified."
                raise Exception(err_msg)
            converted_data = np.array(data[band].data)
        elif isinstance(data, np.ndarray):
            converted_data = data
        else:
            raise DataTypeMismatch(type(data), data_type)
    elif data_type == "dataframe":
        xr_ds = convert_data_type(data, 'xarray', xs=xs, ys=ys, zs=zs, band=band, x_dim=x_dim, y_dim=y_dim, z_dim=z_dim)
        converted_data = xr_ds.to_dataframe()
    else:
        raise DataTypeMismatch(type(data), data_type)

    return converted_data


class RasterData:
    """
    This class represents geo-referenced raster data. Its two main components are a geometry and data.
    The geometry defines all spatial properties of the data like extent, pixel size,
    location and orientation in a spatial reference system (class `RasterGeometry`).
    The other component is data, which is a 2D array-like object that contains the actual values of the raster file.
    Every `RasterData` object has stores an instance of some IO class (e.g., `GeoTiffFile`, `NcFile`), which is used for
    IO operations.
    """

    def __init__(self, rows, cols, sref, gt, data=None, data_type="numpy", is_decoded=True, io=None,
                 label=None, parent=None):
        """
        Basic constructor of class `RasterData`.

        Parameters
        ----------
        rows : int
            Number of pixel rows.
        cols : int
            Number of pixel columns.
        sref : geospade.spatial_ref.SpatialRef or osr.SpatialReference
            Instance representing the spatial reference of the geometry.
        gt : 6-tuple, optional
            GDAL geotransform tuple.
        data : numpy.ndarray or xarray.Dataset, optional
            2D array-like object containing image pixel values.
        data_type : str, optional
            Data type of the returned array-like structure (default is 'numpy'). It can be:
                - 'xarray': loads data as an xarray.DataSet
                - 'numpy': loads data as a numpy.ndarray
        is_decoded : bool, optional
            Defines if the given data is decoded (unit values) or not (default is True).
        io : pyraster.io.geotiff.GeoTiffFile or pyraster.io.netcdf.NcFile, optional
            Instance of a IO Class that is associated with a file that contains the data.
        label : str or int, optional
            Defines a band or a data variable name.
        parent : geospade.definition.RasterGeometry, optional
            Parent `RasterGeometry` instance.
        """

        self.geometry = RasterGeometry(rows, cols, sref, gt, parent=parent)
        if data is not None:
            self._check_data(data)

        # data properties
        self._data = data
        self._is_decoded = is_decoded
        self.data_type = data_type

        self.io = io
        self.label = label
        # TODO: Shahn: change this behaviour?
        if self.io is None and self._data is None:
            raise Exception('Either data or an IO instance has to be given!')

    # TODO: add if statement for checking data type?
    @property
    def data(self):
        """ numpy.ndarray or xarray.Dataset : Retrieves array in the requested data format. """
        if self._data is not None:
            return self._convert_data_type()

    @data.setter
    def data(self, data):
        """
        Sets internal data to given decoded 2D data.

        Parameters
        ----------
        data : numpy.ndarray or xarray.Dataset
            2D array-like object containing image pixel values.
        """

        self._check_data(data)
        rows, cols = convert_data_type(data, "numpy", band=self.label).shape
        if (rows, cols) == (self.geometry.rows, self.geometry.rows):
            self._data = data
            self._is_decoded = True
        else:
            wrn_msg = "Data dimension ({}, {}) mismatches RasterData dimension ({}, {})."
            warnings.warn(wrn_msg.format(rows, cols, self.geometry.rows, self.geometry.rows))

    @classmethod
    def from_array(cls, sref, gt, data, is_decoded=True, **kwargs):
        """
        Creates a `RasterData` object from a 2D array-like object.

        Parameters
        ----------
        sref : geospade.spatial_ref.SpatialRef or osr.SpatialReference
            Instance representing the spatial reference of the geometry.
        gt : 6-tuple, optional
            GDAL geotransform tuple.
        data : numpy.ndarray or xarray.Dataset, optional
            2D array-like object containing image pixel values.
        is_decoded : bool, optional
            Defines if the given data is decoded (unit values) or not (default is True).
        **kwargs
            Keyword arguments for `RasterData` constructor, i.e. `data_type`, `io`, `label` or `parent`.

        Returns
        -------
        RasterData
        """

        rows, cols = None, None
        if isinstance(data, np.ndarray):
            n = len(data.shape)
            if n == 2:
                rows, cols = data.shape
        elif isinstance(data, xr.Dataset):
            n = len(data.dims)
            rows = len(data.coords['y'])
            cols = len(data.coords['x'])
        else:
            err_msg = "Data type is not supported for this class."
            raise Exception(err_msg)

        if n != 2:
            err_msg = "Data has {} dimensions, but 2 dimensions are required."
            raise Exception(err_msg.format(n))

        return cls(rows, cols, sref, gt, data=data, is_decoded=is_decoded, **kwargs)

    @classmethod
    def from_file(cls, arg, read=False, read_kwargs=None, io_class=None, io_kwargs=None, **kwargs):
        """
        Creates a `RasterData` object from a filepath or an IO class instance.

        Parameters
        ----------
        arg : str or GeoTiffFile or NcFile or object
            Full file path to the raster file.
        read : bool, optional
            If true, data is read and assigned to the `RasterData` class.
        read_kwargs : dict, optional
            Keyword arguments for the reading function of the IO class.
        io_class : class, optional
            IO class.
        io_kwargs : dict, optional
            Keyword arguments for IO class initialisation.
        **kwargs
            Keyword arguments for `RasterData` constructor, i.e. `data_type`, `io`, `label`, `is_decoded` or `parent`.

        Returns
        -------
        RasterData
        """

        if isinstance(arg, str):
            return cls.from_filepath(arg, read=read, read_kwargs=read_kwargs,
                                     io_class=io_class, io_kwargs=io_kwargs, **kwargs)
        else:  # if it is not a string it is assumed that it is an IO class instance
            return cls.from_io(arg, read=read, read_kwargs=read_kwargs, **kwargs)

    @classmethod
    def from_filepath(cls, filepath, read=False, read_kwargs=None, io_class=None, io_kwargs=None, **kwargs):
        """
        Creates a `RasterData` object from a filepath.

        Parameters
        ----------
        filepath : str
            Full file path to the raster file.
        read : bool, optional
            If true, data is read and assigned to the `RasterData` class.
        read_kwargs : dict, optional
            Keyword arguments for the reading function of the IO class.
        io_class : class, optional
            IO class.
        io_kwargs : dict, optional
            Keyword arguments for IO class initialisation.
        **kwargs
            Keyword arguments for `RasterData` constructor, i.e. `data_type`, `io`, `label` or `parent`.

        Returns
        -------
        RasterData
        """

        io_class, _ = cls._io_class_from_filepath(filepath) if io_class is None else io_class

        io_kwargs = io_kwargs if io_kwargs is not None else {}
        io = io_class(filepath, mode='r', **io_kwargs)

        return cls.from_io(io, read=read, read_kwargs=read_kwargs, **kwargs)

    @classmethod
    def from_io(cls, io, read=False, read_kwargs=None, **kwargs):
        """
        Creates a `RasterData` object from an IO class instance.

        Parameters
        ----------
        io : GeoTiffFile or NcFile or object
            IO class instance.
        read : bool, optional
            If true, data is read and assigned to the `RasterData` class.
        read_kwargs : dict, optional
            Keyword arguments for the reading function of the IO class.
        **kwargs
            Keyword arguments for `RasterData` constructor, i.e. `data_type`, `io`, `label` or `parent`.

        Returns
        -------
        RasterData
        """

        sref = io.spatialref
        gt = io.geotransform
        sref = SpatialRef(sref)

        if read:
            read_kwargs = read_kwargs if read_kwargs is not None else {}
            col = read_kwargs.pop("col", None)
            row = read_kwargs.pop("row", None)
            data = io.read(col, row, **read_kwargs)
            return cls.from_array(sref, gt, data, io=io, is_decoded=False, **kwargs)
        else:
            rows, cols = io.shape
            return cls(rows, cols, sref, gt, io=io, is_decoded=False, **kwargs)

    @_any_geom2ogr_geom
    def crop(self, geometry, sref=None, apply_mask=False, buffer=0, inplace=False):
        """
        Crops the image by a geometry. Inplace determines, whether a new object
        is returned or the cropping happens on this current object.

        Parameters
        ----------
        geometry : geospade.definition.RasterGeometry or ogr.Geometry or shapely.geometry or list or tuple
            Other geometry used for cropping the data.
        sref : geospade.spatial_ref.SpatialRef or osr.SpatialReference, optional
            Spatial reference of the coordinates.
            Has to be given if the spatial reference is different than the spatial reference of the raster data.
        apply_mask : bool, optional
            If true, a mask is applied for data points being not inside the given geometry (default is False).
        buffer : int, optional
            Pixel buffer for crop geometry (default is 0).
        inplace : bool, optional
            If true, the current instance will be modified.
            If false, a new `RasterData` instance will be created (default).

        Returns
        -------
        RasterData or None
            `RasterData` object only containing data within the intersection.
            If the `RasterData` and the given geometry do not intersect, None is returned.
        """

        # create new geometry
        new_geom = self.geometry & geometry

        # create new data
        if self._data is not None:
            min_col, max_row, max_col, min_row = rel_extent(self.geometry.extent, new_geom.extent,
                                                            x_pixel_size=self.geometry.x_pixel_size,
                                                            y_pixel_size=self.geometry.y_pixel_size)
            row_size = max_row - min_row + 1
            col_size = max_col - min_col + 1
            data = self._read_array(min_row, min_col, row_size=row_size, col_size=col_size)
            if apply_mask:
                mask = self.geometry.create_mask(geometry, buffer=buffer)
                data = self.apply_mask(mask[min_row:(max_row+1), min_col:(max_col+1)], data=data, inplace=inplace)
            else:
                mask = None
        else:
            data = None
            mask = None

        if inplace:
            self.parent = self.geometry
            self.geometry = new_geom
            self._data = data
            if apply_mask:
                self.apply_mask(mask, inplace=True)
            return self
        else:
            raster_data = RasterData.from_array(self.geometry.sref, new_geom.gt, data, io=self.io, parent=self.geometry,
                                                data_type=self.data_type, is_decoded=self._is_decoded, label=self.label)
            raster_data.apply_mask(mask, inplace=True)
            return raster_data

    # TODO: should a RasterData object be returned?
    def load(self, io=None, read_kwargs=None, data_type=None, decode=True, decode_kwargs=None, inplace=False):
        """
        Reads data from disk and assigns the resulting array to the
        `self.data` attribute.

        Parameters
        ----------
        io : GeoTiffFile or NcFile or object, optional
            IO class instance.
        read_kwargs : dict, optional
            Keyword arguments for reading function of IO class.
        data_type : str, optional
            Data type of the returned array-like structure (default is None -> class variable `data_type` is used).
            It can be:
                - 'xarray': loads data as an xarray.DataSet
                - 'numpy': loads data as a numpy.ndarray
        decode : bool, optional
            If true, data is decoded according to the class method `decode`.
        decode_kwargs: dict, optional
            Keyword arguments for the decoder.
        inplace : bool, optional
            If true, the current `RasterData` instance will be modified.
            If false, the loaded data will be returned (default).

        Returns
        -------
        array-like
        """

        io = self.io if io is None else io
        if io is None:
            err_msg = "An IO instance has to be given to load data."
            raise Exception(err_msg)

        read_kwargs = read_kwargs if read_kwargs is not None else {}
        data_type = data_type if data_type is not None else self.data_type

        col = read_kwargs.pop("col", None)
        row = read_kwargs.pop("row", None)
        col_size = read_kwargs.get("col_size", 1)
        row_size = read_kwargs.get("row_size", 1)
        if self.label is not None:
            read_kwargs.update({"band": self.label})
        data = io.read(col, row, **read_kwargs)
        if data is None:
            err_msg = "Could not read data."
            raise Exception(err_msg)

        self._check_data(data)
        data = convert_data_coding(data, self.decode, code_kwargs=decode_kwargs, band=self.label) if decode else data

        # cut geometry according to loaded data
        xs = self.geometry.x_coords
        ys = self.geometry.y_coords
        col_size = len(xs) if col is None else col_size
        row_size = len(ys) if row is None else row_size
        col = 0 if col is None else col
        row = 0 if row is None else row
        x_min = xs[col]
        x_max = xs[col + col_size - 1]
        y_min = ys[row + row_size - 1]
        y_max = ys[row]
        geometry = self.geometry[x_min:(x_max + self.geometry.x_pixel_size),
                   y_min + self.geometry.y_pixel_size:y_max]

        data = self._convert_data_type(data=data, data_type=data_type, geometry=geometry)

        if inplace:
            self._is_decoded = decode
            self.data_type = data_type
            self._data = data
            self.parent = self.geometry
            self.geometry = geometry

        return data

    def read_by_coords(self, x, y, sref=None, data_type=None, px_origin="ul", decode=True, decode_kwargs=None,
                       inplace=False, **kwargs):
        """
        Reads data/one pixel according to the given coordinates.

        Parameters
        ----------
        x : float
            World system coordinate in x direction.
        y : float
            World system coordinate in y direction.
        sref : geospade.spatial_ref.SpatialRef or osr.SpatialReference, optional
            Spatial reference of the coordinates.
            Has to be given if the spatial reference is different than the spatial reference of the raster data.
        data_type : str, optional
            Data type of the returned array-like structure (default is None -> class variable `data_type` is used).
            It can be:
                - 'xarray': loads data as an xarray.DataSet
                - 'numpy': loads data as a numpy.ndarray
        px_origin : str, optional
            Defines the world system origin of the pixel. It can be:
                - upper left ("ul", default)
                - upper right ("ur")
                - lower right ("lr")
                - lower left ("ll")
                - center ("c")
        decode : bool, optional
            If true, data is decoded according to the class method `decode` (default is True).
        decode_kwargs: dict, optional
            Keyword arguments for the decoder.
        inplace : bool, optional
            If true, the current `RasterData` instance will be modified.
            If false, the loaded data will be returned (default).
        ** kwargs
            Keyword arguments for `load` function, i.e. `io`.

        Returns
        -------
        array-like
        """

        read_kwargs = kwargs.get("read_kwargs", {})
        data_type = data_type if data_type is not None else self.data_type

        poi = ogr.Geometry(ogr.wkbPoint)
        poi.AddPoint(x, y)
        row, col = self.geometry.xy2rc(x, y, px_origin=px_origin, sref=sref)
        data = None
        if self._data is None or not self.geometry.intersects(poi, sref=sref): # maybe it does not intersect because part of data is not loaded
            read_kwargs.update({"row": row})
            read_kwargs.update({"col": col})
            data = self.load(read_kwargs=read_kwargs, data_type=data_type, inplace=inplace, decode=decode,
                             decode_kwargs=decode_kwargs, **kwargs)
        else:
            data = self._read_array(row, col, inplace=inplace, decode=decode, decode_kwargs=decode_kwargs,
                                    data_type=data_type)

        return data

    @_any_geom2ogr_geom
    def read_by_geom(self, geometry, sref=None, data_type=None, apply_mask=False, decode=True, decode_kwargs=None,
                     buffer=0, inplace=False, **kwargs):
        """
        Reads data according to the given geometry/region of interest.

        Parameters
        ----------
        geometry : geospade.definition.RasterGeometry or ogr.Geometry or shapely.geometry or list or tuple
            Other geometry used for cropping the data.
        sref : geospade.spatial_ref.SpatialRef or osr.SpatialReference, optional
            Spatial reference of the coordinates.
            Has to be given if the spatial reference is different than the spatial reference of the raster data.
        data_type : str, optional
            Data type of the returned array-like structure (default is None -> class variable `data_type` is used).
            It can be:
                - 'xarray': loads data as an xarray.DataSet
                - 'numpy': loads data as a numpy.ndarray
        apply_mask : bool, optional
            If true, a mask is applied for data points being not inside the given geometry (default is False).
        decode : bool, optional
            If true, data is decoded according to the class method `decode` (default is True).
        decode_kwargs: dict, optional
            Keyword arguments for the decoder.
        buffer : int, optional
            Pixel buffer for crop geometry (default is 0).
        inplace : bool, optional
            If true, the current `RasterData` instance will be modified.
            If false, the loaded data will be returned (default).
        ** kwargs
            Keyword arguments for `load` function, i.e. `io`.

        Returns
        -------
        array-like
        """

        read_kwargs = kwargs.get("read_kwargs", {})
        data_type = data_type if data_type is not None else self.data_type

        new_geom = self.geometry & geometry
        min_col, max_row, max_col, min_row = rel_extent(self.geometry.parent_root.extent, new_geom.extent,
                                                        x_pixel_size=self.geometry.x_pixel_size,
                                                        y_pixel_size=self.geometry.y_pixel_size)
        row_size = max_row - min_row + 1
        col_size = max_col - min_col + 1

        data = None
        if self._data is None or not self.geometry.intersects(geometry, sref=sref):  # maybe it does not intersect because part of data is not loaded
            read_kwargs.update({"row": min_row})
            read_kwargs.update({"col": min_col})
            read_kwargs.update({"row_size": row_size})
            read_kwargs.update({"col_size": col_size})
            data = self.load(read_kwargs=read_kwargs, data_type=data_type, inplace=inplace, decode=decode,
                             decode_kwargs=decode_kwargs, **kwargs)
        else:
            data = self._read_array(min_row, min_col, row_size=row_size, col_size=col_size, inplace=inplace,
                                    decode=decode, decode_kwargs=decode_kwargs, data_type=data_type)

        if apply_mask:
            mask = self.geometry.create_mask(geometry, buffer=buffer)
            data = self.apply_mask(mask[min_row:(max_row+1), min_col:(max_col+1)], data=data, inplace=inplace)

        return data

    def read_by_pixel(self, row, col, row_size=1, col_size=1, px_origin="ul", data_type=None, decode=True,
                      decode_kwargs=None, inplace=False, **kwargs):
        """
        Reads data according to the given pixel extent.

        Parameters
        ----------
        row : int
            Pixel row number.
        col : int
            Pixel column number.
        row_size : int, optional
            Number of rows to read (default is 1).
        col_size : int, optional
            Number of cols to read (default is 1).
        px_origin : str, optional
            Defines the world system origin of the pixel. It can be:
            - upper left ("ul", default)
            - upper right ("ur")
            - lower right ("lr")
            - lower left ("ll")
            - center ("c")
        data_type : str, optional
            Data type of the returned array-like structure (default is None -> class variable `data_type` is used).
            It can be:
                - 'xarray': loads data as an xarray.Dataset
                - 'numpy': loads data as a numpy.ndarray
        decode : bool, optional
            If true, data is decoded according to the class method `decode` (default is True).
        decode_kwargs: dict, optional
            Keyword arguments for the decoder.
        inplace : bool, optional
            If true, the current `RasterData` instance will be modified.
            If false, the loaded data will be returned (default).
        ** kwargs
            Keyword arguments for `load` function, i.e. `io`.

        Returns
        -------
        array-like
        """

        read_kwargs = kwargs.get("read_kwargs", {})
        data_type = data_type if data_type is not None else self.data_type

        # get bounding box of world system coordinates defined by the pixel window
        x_min, y_min = self.geometry.rc2xy(row + row_size, col, px_origin=px_origin)
        x_max, y_max = self.geometry.rc2xy(row, col + col_size, px_origin=px_origin)
        bbox = [(x_min, y_min), (x_max, y_max)]

        data = None
        if self._data is None or not self.geometry.intersects(bbox):
            read_kwargs.update({"row": row})
            read_kwargs.update({"col": col})
            read_kwargs.update({"row_size": row_size})
            read_kwargs.update({"col_size": col_size})
            data = self.load(read_kwargs=read_kwargs, data_type=data_type, inplace=inplace, decode=decode,
                             decode_kwargs=decode_kwargs, **kwargs)
        else:
            data = self._read_array(row, col, row_size=row_size, col_size=col_size, inplace=inplace,
                                    decode=decode, decode_kwargs=decode_kwargs, data_type=data_type)

        return data

    def _read_array(self, row, col, row_size=1, col_size=1, data_type=None, decode=False, decode_kwargs=None,
                    inplace=False):
        """
        Reads/indexes array data from memory.

        Parameters
        ----------
        row : int
            Pixel row number.
        col : int
            Pixel column number.
        row_size : int, optional
            Number of rows to read (default is 1).
        col_size : int, optional
            Number of cols to read (default is 1).
        data_type : str, optional
            Data type of the returned array-like structure (default is None -> class variable `data_type` is used).
            It can be:
                - 'xarray': loads data as an xarray.Dataset
                - 'numpy': loads data as a numpy.ndarray
        decode : bool, optional
            If true, data is decoded according to the class method `decode` (default is False).
        decode_kwargs: dict, optional
            Keyword arguments for the decoder.
        inplace : bool, optional
            If true, the current `RasterData` instance will be modified.
            If false, the data in memory will be returned (default).

        Returns
        -------
        array-like
        """

        decode_kwargs = {} if decode_kwargs is None else decode_kwargs
        data_type = self.data_type if data_type is None else data_type
        row_min = row
        col_min = col
        row_max = row_min + row_size
        col_max = col_min + col_size
        row_min = max(0, row_min)
        col_min = max(0, col_min)
        row_max = min(self.geometry.rows, row_max)
        col_max = min(self.geometry.cols, col_max)
        if row_min >= row_max:
            err_msg = "Row bounds [{};{}] exceed range of possible row numbers {}."
            raise ValueError(err_msg.format(row_min, row_max, self.geometry.rows))

        if col_min >= col_max:
            err_msg = "Column bounds [{};{}] exceed range of possible column numbers {}."
            raise ValueError(err_msg.format(col_min, col_max, self.geometry.cols))

        if isinstance(self._data, np.ndarray):
            data = self._data[row_min:row_max, col_min:col_max]
            if decode and not self._is_decoded:
                data = self.decode(data, **decode_kwargs)
        elif isinstance(self._data, xr.Dataset):
            data_ar = self._data[self.label][row_min:row_max, col_min:col_max]
            if decode and not self._is_decoded:
                data_ar.data = self.decode(data_ar.data, **decode_kwargs)
            data = data_ar.to_dataset()
        else:
            err_msg = "Data type is not supported for accessing and decoding the data."
            raise Exception(err_msg)

        xs = self.geometry.x_coords
        ys = self.geometry.y_coords
        x_min = xs[col]
        x_max = xs[col + col_size - 1]
        y_min = ys[row + row_size - 1]
        y_max = ys[row]
        geometry = self.geometry[x_min:(x_max + self.geometry.x_pixel_size),
                   y_min + self.geometry.y_pixel_size:y_max]

        data = self._convert_data_type(data=data, data_type=data_type, geometry=geometry)

        if inplace:
            self._is_decoded = decode
            self.data_type = data_type
            self._data = data
            self.parent = self.geometry
            self.geometry = geometry

        return data

    def _convert_data_type(self, data=None, data_type=None, geometry=None):
        """
        Class wrapper for `convert_data_type` function.

        Parameters
        ----------
        data : numpy.ndarray or xarray.Dataset, optional
            2D array-like object containing image pixel values.
        data_type : str, optional
            Data type of the returned array-like structure (default is None -> class variable `data_type` is used).
            It can be:
                - 'xarray': loads data as an xarray.Dataset
                - 'numpy': loads data as a numpy.ndarray
        geometry : `RasterGeometry`, optional
            Geometry in synergy with `data`. Needed to select the coordinate values along the data axis.

        Returns
        -------
        numpy.ndarray or xarray.Dataset, optional
            2D array-like object containing image pixel values.
        """

        data = self._data if data is None else data
        data_type = self.data_type if data_type is None else data_type
        geometry = self.geometry if geometry is None else geometry

        xs = geometry.x_coords
        ys = geometry.y_coords
        zs = None

        return convert_data_type(data, data_type, xs=xs, ys=ys, zs=zs, band=self.label)

    def apply_mask(self, mask, data=None, inplace=False):
        """
        Applies a 2D mask to given or internal data.

        Parameters
        ----------
        mask : numpy.ndarray
            2D mask.
        data : numpy.ndarray or xarray.Dataset, optional
            2D array-like object containing image pixel values.
        inplace : bool, optional
            If true, the current `RasterData` instance will be modified.
            If false, the data in memory will be returned (default).

        Returns
        -------
        numpy.ndarray or xarray.Dataset, optional
            2D array-like object containing image pixel values and a numpy masked array.
        """

        self._check_data(mask)
        self._check_data(data)
        data = data if data is not None else self._data

        if isinstance(data, np.ndarray):
            data = np.ma.array(data, mask=mask)
        elif isinstance(data, xr.Dataset):
            data_ar = data[self.label]
            data_ar.data = np.ma.array(data_ar.data, mask=mask)
            data = data_ar.to_dataset()
        else:
            err_msg = "Data type is not supported for accessing and decoding the data."
            raise Exception(err_msg)

        if inplace:
            self._data = data

        return data

    def write(self, filepath, io_class=None, io_kwargs=None, write_kwargs=None, encode=True, encode_kwargs=None):
        """
        Writes data to disk into a target file or into a file associated
        with this object.

        Parameters
        ----------
        filepath : str
            Full file path of the output file.
        io_class : class, optional
            IO class.
        io_kwargs : dict, optional
            Keyword arguments for IO class initialisation.
        write_kwargs : dict, optional
            Keyword arguments for writing function of IO class.
        encode : bool, optional
            If true, encoding function of `RasterData` class is applied (default is True).
        encode_kwargs : dict, optional
            Keyword arguments for the encoder.
        """

        write_kwargs = write_kwargs if write_kwargs is not None else {}
        io_kwargs = io_kwargs if io_kwargs is not None else {}

        io_class, file_type = self._io_class_from_filepath(filepath) if io_class is None else io_class
        io = io_class(filepath, mode='w', **io_kwargs)

        if file_type == "GeoTIFF":
            data_type = "numpy"
        elif file_type == "NetCDF":
            data_type = "xarray"
        else:
            data_type = self.data_type

        data = self._convert_data_type(data_type=data_type)
        data = convert_data_coding(data, self.encode, code_kwargs=encode_kwargs, band=self.label) if encode else data
        io.write(data, **write_kwargs)
        io.close()

    def encode(self, data, **kwargs):
        """
        Encodes data.

        Parameters
        ----------
        data : numpy.ndarray or xarray.DataArray, optional
            2D array-like object containing image pixel values.
        **kwargs : Keyword arguments for encoding function.

        Returns
        -------
        data : numpy.ndarray or xarray.DataArray, optional
            Encoded array.
        """

        return data

    def decode(self, data, **kwargs):
        """
        Decodes data.

        Parameters
        ----------
        data : numpy.ndarray or xarray.DataArray, optional
            2D array-like object containing image pixel values.
        **kwargs : Keyword arguments for decoding function.

        Returns
        -------
        data : numpy.ndarray or xarray.DataArray, optional
            Decoded array.
        """

        return data

    # TODO: decoding alse here?
    def plot(self, ax=None, proj=None, proj_extent=None, extent_sref=None, cmap='viridis'):
        """
        Plots the data on a map that uses a projection if provided.
        If not, the map projection defaults to the spatial reference
        in which the data are provided. The extent of the data is specified by `extent`.
        If an extent is not provided, it defaults to the bbox of the data's geometry.
        If provided, one can also specify the spatial reference of the extent that is being parsed, otherwise it is
        assumed that the spatial reference of the extent is the same as the spatial reference of the data.

        Parameters
        ----------
        ax :  matplotlib.pyplot.axes
            Pre-defined Matplotlib axis.
        proj :  cartopy.crs.Projection or its subclass, optional
            Projection of the map. The figure will be drawn in
            this spatial reference. If omitted, the spatial reference in which
            the data are present is used.
        proj_extent : 4 tuple, optional
            Extent of the projection (x_min, x_max, y_min, y_max). If omitted, the bbox of the data is used.
        extent_sref : geospade.spatial_ref.SpatialRef or osr.SpatialReference, optional
            Spatial reference of the coordinates.
            Has to be given if the spatial reference is different than the spatial reference of the raster data.
        cmap : matplotlib.colors.Colormap or string, optional
            Colormap for displaying the data (default is 'viridis').
        """

        if 'matplotlib' in sys.modules:
            import matplotlib.pyplot as plt
        else:
            err_msg = "Module 'matplotlib' is mandatory for plotting a RasterGeometry object."
            raise ImportError(err_msg)

        if proj is None:
            proj = self.geometry.to_cartopy_crs()

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection=proj)

        ll_x, ll_y, ur_x, ur_y = self.geometry.extent
        img_extent = ll_x, ur_x, ll_y, ur_y

        if proj_extent:
            x_min, y_min, x_max, y_max = proj_extent
            if extent_sref:
                x_min, y_min = coordinate_traffo(x_min, y_min, extent_sref, self.geometry.sref.osr_sref)
                x_max, y_max = coordinate_traffo(x_max, y_max, extent_sref, self.geometry.sref.osr_sref)
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])

        if isinstance(cmap, str):
            cmap = cm.get_cmap(cmap)

        ax.coastlines()

        # plot data
        ax.imshow(self._convert_data_type(data_type="numpy"), extent=img_extent, origin='upper', transform=proj, cmap=cmap)
        ax.set_aspect('equal', 'box')

        return ax

    @staticmethod
    def _check_data(data):
        """
        Checks array type and structure of data.

        Parameters
        ----------
        data : numpy.ndarray or xarray.DataArray, optional
            2D array-like object containing image pixel values.

        Returns
        -------
        bool
            If true, the given data fulfills all requirements for a `RasterData` object.
        """

        if data is not None:
            if isinstance(data, np.ndarray):
                n = len(data.shape)
            elif isinstance(data, xr.Dataset):
                n = len(data.dims)
            else:
                err_msg = "Data type is not supported for this class."
                raise Exception(err_msg)

            if n != 2:
                err_msg = "Data has {} dimensions, but 2 dimensions are required."
                raise Exception(err_msg.format(n))
            return True
        else:
            return False

    @staticmethod
    def _io_class_from_filepath(filepath):
        """
        Selects an IO class depending on the filepath/file ending.

        Parameters
        ----------
        filepath : str
            Full file path of the output file.

        Returns
        -------
        io_class : class
            IO class.
        file_type : str
            Type of file: can be "GeoTIFF" or "NetCDF".
        """

        tiff_ext = ('.tiff', '.tif', '.geotiff')
        netcdf_ext = ('.nc', '.netcdf', '.ncdf')

        # determine IO class
        file_ext = os.path.splitext(filepath)[1].lower()
        if file_ext in tiff_ext:
            io_class = GeoTiffFile
            file_type = "GeoTIFF"
        elif file_ext in netcdf_ext:
            io_class = NcFile
            file_type = "NetCDF"
        else:
            raise IOError('File format not supported.')

        return io_class, file_type

    @staticmethod
    def _data_type_from_data(data):
        """

        Parameters
        ----------
        data : numpy.ndarray or xarray.DataArray, optional
            Array-like object containing image pixel values.

        Returns
        -------
        str :
            Data type of the given array. It can be "numpy" or "xarray".
        """

        if isinstance(data, np.ndarray):
            data_type = "numpy"
        elif isinstance(data, xr.Dataset):
            data_type = "xarray"
        else:
            err_msg = "Data type is not supported for this class."
            raise Exception(err_msg)

        return data_type

    # ToDo: needs to be tested
    def __getitem__(self, item):
        """
        Handles indexing of a raster data object,
        which is herein defined as a 2D spatial indexing via x and y coordinates.

        Parameters
        ----------
        item : 2-tuple
            Tuple containing coordinate slices (e.g., (10:100,20:200)) or coordinate values.

        Returns
        -------
        veranda.raster.RasterData
            Raster data defined by the intersection.
        """

        intsctd_geom = self.geometry[item]
        return self.crop(intsctd_geom, inplace=False)


class RasterStack(object):
    """
    Class representing a collection of `RasterData` objects. A raster stack can be described as a stack of
    geospatial files covering a certain extent in space, given by a geometry object.
    """
    def __init__(self, raster_datas, data=None, geometry=None, parent=None, io=None):
        """
        Constructor of class `RasterStack`.

        Parameters
        ----------

        Returns
        -------
        """
        self.raster_datas = raster_datas
        self.data = data
        self.io = io
        self.parent = parent

        self.geometry = geometry
        if self.geometry is None:
            if self.is_congruent:
                self.geometry = raster_datas.values()[0].geometry

    @classmethod
    def from_filepaths(cls, filepaths, label, **kwargs):
        pass

    @classmethod
    def from_array(cls, data, sref, gt, labels=None, parent=None, **kwargs):

        (n, rows, cols) = data.shape
        raster_data = RasterData(rows, cols, sref, gt)
        raster_datas = [copy.deepcopy(raster_data) for _ in range(n)]
        if labels is None:
            labels = list(range(n))
        ds = pd.Series(raster_datas, index=labels)

        return cls(ds, data=data, geometry=raster_data.geometry, parent=parent, **kwargs)

    @classmethod
    def from_others(cls, others, dtype=None, geometry=None, **kwargs):
        if geometry is None:
            geometry = RasterGeometry.get_common_geometry([other.geometry for other in others])

        if dtype == "numpy":
            combined_data = np.zeros((len(others[0].raster_datas.index), geometry.rows, geometry.cols))
            for other in others:
                if other is not None:
                    min_col, max_row, max_col, min_row = other.geometry.get_rel_extent(geometry, unit='px')
                    combined_data[:, min_row:(max_row+1), min_col:(max_col+1)] = other.data

            return RasterStack.from_array(combined_data, labels=others[0].index)
        elif dtype == "xarray":
            datasets = [other.data for other in others if other is not None]
            return RasterStack.from_array(xr.combine_by_coords(datasets), labels=others[0].index)
        else:
            raise DataTypeUnknown(dtype)

    def crop(self, geometry, inplace=True):
        """
        Crops the raster stack by a geometry. ˋinplaceˋ determines, whether a new object
        is returned or the cropping happens on the current raster stack.
        The resulting ˋRasterStackˋ object contains no-data value in pixels that
        are not contained in the original RasterData object

        Parameters
        ----------
        geometry : RasterGeometry or shapely geometry
            geometry to which current RasterData should be cropped
        inplace : bool
            if true, current object gets modified, else new object is returned

        Returns
        -------
        RasterData or None
            Depending on inplace argument

        """

        # create new geometry
        new_geom = self.geometry & geometry

        # create new data
        data = None
        if self.__check_data():
            min_col, max_row, max_col, min_row = new_geom.get_rel_extent(self.geometry, unit="px")
            data = self.data[:, min_row:(max_row + 1), min_col:(max_col + 1)]
            mask = rasterise_polygon(new_geom.geometry)
            data = np.ma.array(data, mask=np.stack([mask] * data.shape[0], axis=0))

        if inplace:
            self.geometry = new_geom
            self.parent = self
            self.data = data
        else:
            return RasterStack(self.raster_datas, geometry=new_geom, parent=self, data=data)

    # TODO: in progress
    @_stack_io(mode='w')
    def write(self, data=None, **kwargs):
        if data is None:
            if self.data is not None:
                data = self.data
            else:
                raise Exception('Please specify data to write.')

        self.io.write(data, **kwargs)

    @_any_geom2ogr_geom
    def load_by_geom(self, geom, **kwargs):
        raster_stack = self.crop(geom, inplace=False)
        min_col, _, _, min_row = raster_stack.geometry.get_rel_extent(self.geometry, unit='px')
        return raster_stack.load(min_col, min_row,
                                 col_size=raster_stack.geometry.cols,
                                 row_size=raster_stack.geometry.rows,
                                 **kwargs)

    def load_by_coord(self, x, y, osr_spref=None, **kwargs):
        return self.load_by_geom((x, y), osr_spref=osr_spref, **kwargs)

    @_stack_io(mode='r')
    def load(self, col, row, col_size=1, row_size=1, band=1, dtype="numpy", x_dim="x", y_dim="y", z_dim="time",
             inplace=True):

        min_col = col
        min_row = row
        max_col = col + col_size
        max_row = row + row_size
        pixel_polygon = Polygon(((0, 0), (self.geometry.cols, 0), (self.geometry.cols, self.geometry.rows),
                                 (0, self.geometry.rows), (0, 0)))
        ll_point = Point((min_col, max_row))
        ur_point = Point((max_col, min_row))
        if ll_point.intersects(pixel_polygon) and ur_point.intersects(pixel_polygon):
            col = 0 if col < 0 else col
            row = 0 if row < 0 else row
            max_col = self.geometry.cols-1 if max_col >= self.geometry.cols else max_col
            max_row = self.geometry.rows-1 if max_row >= self.geometry.rows else max_row
            col_size = max_col - col
            row_size = max_row - row

            data = self._ds.read(col, row, col_size=col_size, row_size=row_size, band=band)
            xs = None
            ys = None
            zs = None
            if dtype != "numpy":  # data should be referenced with coordinates
                max_col = col + col_size
                max_row = row + row_size
                if self.geometry.is_axis_parallel:  # if the raster is axis parallel
                    xs, _ = ij2xy(np.arange(col, max_col), np.array([row] * col_size), self.geometry.gt)
                    _, ys = ij2xy(np.array([col] * row_size), np.arange(row, max_row), self.geometry.gt)
                else:  # raster is not axis-parallel, each point/pixel needs to be projected
                    cols, rows = np.meshgrid(np.arange(col, max_col), np.arange(row, max_row))
                    xs, ys = ij2xy(cols, rows)

                zs = self.raster_datas.index

            data = convert_dtype(data, dtype, xs, ys, zs, band=band, x_dim=x_dim, y_dim=y_dim, z_dim=z_dim)

            if inplace:
                self.data = data
                return self
            else:
                return RasterStack.from_array(data, self.geometry.sref, self.geometry.gt, labels=self.raster_datas.index)
        else:
            return None

    def plot(self, id=None, axis_unit='px', cm=None, **kwargs):
        if id is None:
            if self.is_congruent:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                cols, rows = np.meshgrid(np.arange(0, self.geometry.width, 1), np.arange(0, self.geometry.height, 1))
                for band, rd in enumerate(self.raster_datas.values()):
                    data = rd.data
                    bands = np.ones(cols.shape)*band
                    ax.plot_surface(cols, -rows, bands, rstride=1, cstride=1, facecolors=cm(data), shade=False)
        else:
            if id not in self.raster_datas.keys():
                err_msg = "'{}' is not a valid index.".format(id)
                raise KeyError(err_msg)
            else:
                rd = self.raster_datas[id]
                rd.plot(**kwargs)

    @property
    def is_congruent(self):
        ref_geom = self.raster_datas[0].geometry
        return all([ref_geom == rd.geometry for rd in self.raster_datas.values()])

    # TODO: data should be checked depending on the data type, i.e. if it is a numpy array or a xarray
    def __check_data(self):
        if self.data is not None:
            n = len(self.data.shape)
            if n != 3:
                raise Exception('Data has {} dimensions, but 3 dimensions are required.'.format(n))
            return True
        else:
            return False


class RasterMosaic(object):

    def __init__(self, raster_stacks, grid=None, aggregator=None):

        self.aggregator = aggregator
        self.raster_stacks = raster_stacks

        raster_geoms = [raster_stack.geometry for raster_stack in self.raster_stacks
                        if raster_stack.geometry is not None]

        if grid is None:
            self.grid = RasterGrid(raster_geoms)
        else:
            self.grid = grid

        self.geometry = RasterGeometry.get_common_geometry(raster_geoms)

    @classmethod
    def from_df(cls, arg, **kwargs):
        pass

    @classmethod
    def from_list(cls, arg, **kwargs):
        arg_dict = dict()
        for i, layer in enumerate(arg):
            arg_dict[i] = layer

        return cls.from_dict(arg_dict, **kwargs)

    @classmethod
    def from_dict(cls, arg, **kwargs):
        structure = RasterMosaic._dict2structure(arg)
        raster_stacks = cls._build_raster_stacks(structure)

        return cls(raster_stacks)

    @staticmethod
    def _dict2structure(struct_dict):
        structure = dict()
        structure["raster_data"] = []
        structure["spatial_id"] = []
        structure["layer_id"] = []
        geoms = []
        for layer_id, layer in struct_dict.items():
            for elem in layer:
                structure['layer_id'].append(layer_id)
                if isinstance(elem, RasterData):
                    rd = elem
                elif isinstance(elem, str) and os.path.exists(elem):
                    rd = RasterData.from_file(elem, mode=None)
                else:
                    raise Exception("Data type '{}' is not understood.".format(type()))

                structure['raster_data'].append(rd)
                if rd.geometry.id is not None:
                    structure['spatial_id'].append(rd.geometry.id)
                else:
                    spatial_id = None
                    for i, geom in enumerate(geoms):
                        if geom == rd.geometry:
                            spatial_id = i
                            break
                    if spatial_id is None:
                        spatial_id = len(geoms)
                        geoms.append(rd.geometry)
                    structure['spatial_id'].append(spatial_id)

        return pd.DataFrame(structure)

    @staticmethod
    def _build_raster_stacks(structure):
        struct_groups = structure.groupby(by="spatial_id")
        raster_stacks = {'raster_stack': []}
        spatial_ids = []
        for struct_group in struct_groups:
            raster_datas = struct_group['raster_data']
            raster_datas.set_index(struct_group['layer_id'], inplace=True)
            raster_stacks['raster_stack'].append(RasterStack(raster_datas))
            spatial_ids.append(struct_group['spatial_id'])

        return pd.Series(raster_stacks, index=spatial_ids)

    @_any_geom2ogr_geom
    def read_by_geom(self, geom, osr_sref=None, band=1, dtype='numpy', **kwargs):
        raster_grid = self.grid.crop(geom)
        geometry = self.geometry.crop(geom)
        spatial_ids = raster_grid.geom_ids
        raster_stacks = [raster_stack.load_by_geom(geom, osr_sref=osr_sref, dtype=dtype, band=band, inplace=False, **kwargs)
                         for raster_stack in self.raster_stacks[spatial_ids]]

        return RasterStack.from_others(raster_stacks, dtype=dtype, geometry=geometry)

    def read_by_coord(self, x, y, osr_sref=None, band=1, dtype='numpy', **kwargs):
        point = Point((x, y))
        return self.read_by_geom(point, osr_sref=osr_sref, band=band, dtype=dtype, **kwargs)

    def read(self, col, row, col_size=1, row_size=1, band=1, dtype="numpy", **kwargs):
        if col_size == 1 and row_size == 1:
            x, y = ij2xy(col, row, self.geometry.gt)
            return self.read_by_coord(x, y, band=band, dtype=dtype, **kwargs)
        else:
            max_col = col + col_size
            max_row = row + row_size
            min_x, min_y = ij2xy(col, max_row, self.geometry.gt)
            max_x, max_y = ij2xy(max_col, row, self.geometry.gt)
            bbox = [(min_x, min_y), (max_x, max_y)]
            return self.read_by_geom(bbox, band=band, dtype=dtype, **kwargs)





