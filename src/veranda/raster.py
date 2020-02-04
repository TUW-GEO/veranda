import os
import sys
import copy
import ogr
import osr
import inspect
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import cm
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
from geospade.operation import xy2ij
from geospade.operation import ij2xy
from geospade.operation import rel_extent

# TODO: can we represent a rotated array with xarray?
def _file_io(mode='r'):
    """
    Decorator which checks whether parsed file arguments are actual IO classes (e.g., `NcFile`, `GeoTiffFile`, ...)
    or strings. If a string/filename is given, then a corresponding IO class is initiated with the filename.

    mode : str, optional
        File opening mode:
            - 'r': read operation (default).
            - 'w': write operation (default).
    """

    def _inner_file_io(func):
        tiff_ext = ('.tiff', '.tif', '.geotiff')
        netcdf_ext = ('.nc', '.netcdf', '.ncdf')

        def wrapper(self, *args, **kwargs):
            io_kwarg = kwargs.get('io', None)
            io_class_kwarg = kwargs.get('io_class', None)
            io_class_kwargs = kwargs.get('io_kwargs', None)
            if args is None:
                arg = io_kwarg
                args = tuple()
            else:
                arg = args[0]  # get first argument
                args = args[1:]  # remove first argument

            if isinstance(arg, str):  # argument seems to be a filename
                io = None
                file_ext = os.path.splitext(arg)[1].lower()
                if io_class_kwarg is not None:
                    io_class = io_class_kwarg
                elif file_ext in tiff_ext:
                    io_class = GeoTiffFile
                elif file_ext in netcdf_ext:
                    io_class = NcFile
                else:
                    raise IOError('File format not supported.')
            else:  # argument seems to be a self defined io class
                io_class = None
                io = arg

            io_class_kwargs = dict() if io_class_kwargs is None else io_class_kwargs
            geotransform = io_class_kwargs.get('geotransform', self.geometry.gt if hasattr(self, "geometry") else None)
            spatialref = io_class_kwargs.get('spatialref', self.geometry.sref.wkt if hasattr(self, "geometry") else None)
            add_kwargs = {'geotransform': geotransform,
                          'spatialref': spatialref}
            io_class_kwargs.update(add_kwargs)

            if self.io is not None:
                return func(self, self.io, *args, **kwargs)
            elif io is not None:
                self.io = io
                return func(self, io, *args, **kwargs)
            else:
                io = io_class(arg, mode=mode, **io_class_kwargs)
                self.io = io
                return func(self, io, *args, **kwargs)

        return wrapper
    return _inner_file_io


def _stack_io(mode='r'):
    def _inner_stack_io(func):
        """
        Decorator which checks whether parsed file arguments are acutual IO-Classes
        such as NcFile or GeoTiffFile. If string is parsed as file, then
        a corresponding IO-Class object is parsed into the original function
        """

        def wrapper(self, *args, **kwargs):
            io = kwargs.get('io', None)
            ref_io_class = self.rds[0].io
            if isinstance(ref_io_class, GeoTiffFile):
                io_class = GeoTiffRasterTimeStack
            elif isinstance(ref_io_class, NcFile):
                io_class = NcRasterTimeStack
            else:
                raise IOError('IO class not supported.')

            create_io = False
            if self.io is None:
                if io is not None:
                    self.io = io
                else:
                    create_io = True
            else:
                if self.io.mode != mode:
                    create_io = True

            if create_io:
                filepaths = [rd.io.filename for rd in self.rds]
                df = pd.DataFrame({'filenames': filepaths})
                geotransform = kwargs.get('geotransform', self.geometry.gt)
                spatialref = kwargs.get('spatialref', self.geometry.wkt)
                add_kwargs = {'geotransform': geotransform,
                              'spatialref': spatialref}
                kwargs.update(add_kwargs)
                self.io = io_class(file_ts=df, mode=mode, **kwargs)

            return func(self, *args, **kwargs)

        return wrapper
    return _inner_stack_io

# TODO: where should we put this?
def convert_data(data, data_type, xs=None, ys=None, zs=None, band=1, x_dim='x', y_dim='y', z_dim='time'):
    """
    Converts `data` into an array-like object defined by `dtype`. It supports NumPy arrays, Xarray arrays and
    Pandas data frames.

    Parameters
    ----------
    data : list of numpy.ndarray or list of xarray.DataSets or numpy.ndarray or xarray.DataArray
    dtype : str
        Data type of the returned array-like structure (default is 'xarray'). It can be:
            - 'xarray': loads data as an xarray.DataSet
            - 'numpy': loads data as a numpy.ndarray
            - 'dataframe': loads data as a pandas.DataFrame
    xs : list, optional
        List of world system coordinates in X direction.
    ys : list, optional
        List of world system coordinates in Y direction.
    temporal_dim_name : str, optional
        Name of the temporal dimension (default: 'time').
    band : int or str, optional
        Band number or name (default is 1).

    Returns
    -------
    list of numpy.ndarray or list of xarray.DataSets or pandas.DataFrame or numpy.ndarray or xarray.DataSet
        Data as an array-like object.
    """

    if data_type == "xarray":
        if isinstance(data, list) and isinstance(data[0], np.ndarray):
            ds = []
            for i, entry in enumerate(data):
                x = xs[i]
                y = ys[i]
                if not isinstance(x, list):
                    x = [x]
                if not isinstance(y, list):
                    y = [y]
                xr_ar = xr.DataArray(entry, coords={z_dim: zs, y_dim: y, x_dim: x},
                                     dims=[z_dim, 'y', 'x'])
                ds.append(xr.Dataset(data_vars={band: xr_ar}))
            converted_data = xr.merge(ds)
        elif isinstance(data, list) and isinstance(data[0], xr.Dataset):
            converted_data = xr.merge(data)
            converted_data.attrs = data[0].attrs
        elif isinstance(data, np.ndarray):
            xr_ar = xr.DataArray(data, coords={z_dim: zs, y_dim: ys, x_dim: xs}, dims=[z_dim, y_dim, x_dim])
            converted_data = xr.Dataset(data_vars={band: xr_ar})
        elif isinstance(data, xr.Dataset):
            converted_data = data
        else:
            raise DataTypeMismatch(type(data), data_type)

    elif data_type == "numpy":
        if isinstance(data, list) and isinstance(data[0], np.ndarray):
            if len(data) == 1:
                converted_data = data[0]
            else:
                converted_data = data
        elif isinstance(data, list) and isinstance(data[0], xr.Dataset):
            converted_data = [np.array(entry[band].data) for entry in data]
            if len(converted_data) == 1:
                converted_data = converted_data[0]
        elif isinstance(data, xr.Dataset):
            converted_data = np.array(data[band].data)
        elif isinstance(data, np.ndarray):
            converted_data = data
        else:
            raise DataTypeMismatch(type(data), data_type)
    elif data_type == "dataframe":
        xr_ds = convert_data(data, 'xarray', xs=xs, ys=ys, zs=zs, band=band, x_dim=x_dim, y_dim=y_dim, z_dim=z_dim)
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

    def __init__(self, rows, cols, sref, gt, data=None, data_type="numpy", io=None, label=None, parent=None):
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
        data : numpy.ndarray or xarray.DataArray, optional
            2D array-like object containing image pixel values.
        data_type : str, optional
            Data type of the returned array-like structure (default is 'numpy'). It can be:
                - 'xarray': loads data as an xarray.DataSet
                - 'numpy': loads data as a numpy.ndarray
                - 'dataframe': loads data as a pandas.DataFrame
        io : pyraster.io.geotiff.GeoTiffFile or pyraster.io.netcdf.NcFile, optional
            Instance of a IO Class that is associated with a file that contains the data.
        label : str or int, optional
            Defines a band or a data variable name.
        parent : geospade.definition.RasterGeometry, optional
            Parent `RasterGeometry` instance.
        """

        self.geometry = RasterGeometry(rows, cols, sref, gt, parent=parent)
        self.__check_data(convert_data(data, "numpy"))
        self._data = data
        self.data_type = data_type
        self.io = io
        self.label = label

    # TODO: add if statement for checking data type?
    @property
    def data(self):
        """ array-like : Retrieves array in the requested data format. """
        xs = self.geometry.x_coords
        ys = self.geometry.y_coords
        zs = None
        return convert_data(self._data, self.data_type, xs=xs, ys=ys, zs=zs, band=self.label)

    @classmethod
    def from_array(cls, sref, gt, data, **kwargs):
        """
        Creates a `RasterData` object from a 2D array-like object.

        Parameters
        ----------
        sref : geospade.spatial_ref.SpatialRef or osr.SpatialReference
            Instance representing the spatial reference of the geometry.
        gt : 6-tuple, optional
            GDAL geotransform tuple.
        data : numpy.ndarray or xarray.DataArray, optional
            2D array-like object containing image pixel values.
        **kwargs
            Keyword arguments for `RasterData` constructor, i.e. `data_type`, `io`, `label` or `parent`.

        Returns
        -------
        RasterData
        """
        # TODO: add data checks or convert it?
        rows, cols = convert_data(data, "numpy").shape
        return cls(rows, cols, sref, gt, data, **kwargs)

    @classmethod
    @_file_io(mode='r')
    def from_file(cls, io, read_kwargs=None, io_kwargs=None):
        """
        Creates a `RasterData` object from a file.

        Parameters
        ----------
        io : string or GeoTiffFile or NcFile
            File path or IO class instance. The decorator deals with the distinction.
        read_kwargs : dict, optional
            Keyword arguments for the reading function of the IO class.
        io_kwargs: dict, optional
            Potential arguments for IO class initialisation (is applied in decorator).

        Returns
        -------
        RasterData
        """

        read_kwargs = read_kwargs if read_kwargs is not None else {}

        spatialref = io_kwargs.get('spatialref', None)
        geotransform = io_kwargs.get('geotransform', None)
        if spatialref is None or geotransform is None:
            spatialref = io.spatialref
            geotransform = io.geotransform

        sref = SpatialRef(spatialref)
        data = io.read(**read_kwargs)
        # TODO: add data checks or convert it?
        rows, cols = convert_data(data, "numpy").shape
        return cls(rows, cols, sref, geotransform, data=data, io=io)

    @_any_geom2ogr_geom
    def crop(self, geometry, sref=None, inplace=True):
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
        inplace : bool, optional
            If true, the current instance will be modified.
            If false, a new `RasterData` instance will be created.

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
            min_col, max_row, max_col, min_row = rel_extent(self.geometry.extent, new_geom.extent)
            row_size = max_row - min_row
            col_size = max_col - min_col
            data = self._read_array(min_row, min_col, row_size=row_size, col_size=col_size)
        else:
            data = None

        if inplace:
            self.parent = self.geometry
            self.geometry = new_geom
            self._data = data
            return self
        else:
            return RasterData.from_array(self.geometry.sref, new_geom.gt, data, io=self.io, parent=self.geometry,
                                         data_type=self.data_type, label=self.label)

    # TODO: define/discuss IO function names
    @_file_io('r')
    def load(self, io=None, read_kwargs=None, data_type=None, decode=True, inplace=True):
        """
        Reads data from disk and assigns the resulting array to the
        `self.data` attribute.

        Parameters
        ----------
        io : string or GeoTiffFile or NcFile
            Filepath to the file or IO class instance. The decorator deals with
            the distinction.
        read_kwargs : dict, optional
            Keyword arguments for reading function of IO class.
        data_type : str, optional
            Data type of the returned array-like structure (default is None -> class variable `data_type` is used).
            It can be:
                - 'xarray': loads data as an xarray.DataSet
                - 'numpy': loads data as a numpy.ndarray
        decode : bool, optional
            If true, data is decoded according to the class method `decode`.
        inplace : bool, optional
            If true, the current `data` instance will be modified.
            If false, the loaded data will be returned.

        Returns
        -------
        array-like
        """

        read_kwargs = read_kwargs if read_kwargs is not None else {}
        data_type = data_type if data_type is not None else self.data_type
        data = self.decode(io.read(**read_kwargs)) if decode else io.read(**read_kwargs)
        data = convert_data(data, data_type)
        self.__check_data(data if data_type == "numpy" else convert_data(data, "numpy"))
        if inplace:
            self._data = data

        return data

    def read_by_coords(self, x, y, sref=None, data_type=None, px_origin="ul", decode=True, **kwargs):
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
        if self._data is None or not self.geometry.intersects(poi, sref=sref): # maybe it does not intersect because part of data is not loaded
            read_kwargs.update({"row": row})
            read_kwargs.update({"col": col})
            self.load(read_kwargs=read_kwargs, data_type=data_type, inplace=True, decode=False, **kwargs)

        xs = [x]
        ys = [y]
        zs = None
        return convert_data(self._read_array(row, col, decode=decode), data_type, xs=xs, ys=ys, zs=zs, band=self.label)

    def read_by_geom(self, geometry, sref=None, data_type=None, apply_mask=True, decode=True, **kwargs):
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
            If true, a mask is applied for data points being not inside the given geometry (default is True).
        decode : bool, optional
            If true, data is decoded according to the class method `decode` (default is True).
        ** kwargs
            Keyword arguments for `load` function, i.e. `io`.

        Returns
        -------
        array-like
        """

        read_kwargs = kwargs.get("read_kwargs", {})
        data_type = data_type if data_type is not None else self.data_type

        new_geom = geometry & self.geometry

        if self._data is None or not self.geometry.intersects(geometry, sref=sref):  # maybe it does not intersect because part of data is not loaded
            min_col, max_row, max_col, min_row = rel_extent(self.geometry.parent_root.extent, new_geom.extent)
            row_size = max_row - min_row + 1
            col_size = max_col - min_col + 1
            read_kwargs.update({"row": min_row})
            read_kwargs.update({"col": min_col})
            read_kwargs.update({"row_size": row_size})
            read_kwargs.update({"col_size": col_size})
            self.load(read_kwargs=read_kwargs, data_type=data_type, inplace=True, **kwargs)

        min_col, max_row, max_col, min_row = rel_extent(self.geometry.extent, new_geom.extent)
        row_size = max_row - min_row + 1
        col_size = max_col - min_col + 1
        if apply_mask:
            mask = rasterise_polygon(list(shapely.wkt.loads(new_geom.boundary.ExportToWkt()).exterior.coords),
                                          sres=self.geometry.x_pixel_size)
        else:
            mask = None
        xs = new_geom.x_coords
        ys = new_geom.y_coords
        zs = None
        return convert_data(self._read_array(min_row, min_col, row_size=row_size, col_size=col_size, mask=mask,
                                             decode=decode), data_type, xs=xs, ys=ys, zs=zs, band=self.label)

    def read_by_pixel(self, row, col, row_size=1, col_size=1, px_origin="ul", data_type=None, decode=True, **kwargs):
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
                - 'xarray': loads data as an xarray.DataSet
                - 'numpy': loads data as a numpy.ndarray
        decode : bool, optional
            If true, data is decoded according to the class method `decode` (default is True).
        ** kwargs
            Keyword arguments for `load` function, i.e. `io`.

        Returns
        -------
        array-like
        """

        read_kwargs = kwargs.get("read_kwargs", {})
        data_type = data_type if data_type is not None else self.data_type

        x_min, y_min = self.geometry.rc2xy(row + row_size, col, px_origin=px_origin)
        x_max, y_max = self.geometry.rc2xy(row, col + col_size, px_origin=px_origin)
        bbox = [(x_min, y_min), (x_max, y_max)]

        if self._data is None or not self.geometry.intersects(bbox):
            read_kwargs.update({"row": row})
            read_kwargs.update({"col": col})
            read_kwargs.update({"row_size": row_size})
            read_kwargs.update({"col_size": col_size})
            self.load(read_kwargs=read_kwargs, data_type=data_type, inplace=True, **kwargs)

        xs = np.arange(x_min, x_max + self.geometry.x_pixel_size, self.geometry.x_pixel_size).tolist()
        ys = np.arange(y_min, y_max + self.geometry.y_pixel_size, self.geometry.y_pixel_size).tolist()
        zs = None
        return convert_data(self._read_array(row, col, row_size=row_size, col_size=col_size, decode=decode),
                            data_type, xs=xs, ys=ys, zs=zs, band=self.label)

    def _read_array(self, row, col, row_size=1, col_size=1, data=None, mask=None, decode=False):
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
        data : numpy.ndarray or xarray.DataArray, optional
            2D array-like object containing image pixel values.
        mask : numpy.ndarray, optional
            2D boolean mask.
        decode : bool, optional
            If true, data is decoded according to the class method `decode` (default is False).

        Returns
        -------
        array-like
        """

        data = self._data if data is None else data
        row_min = row
        col_min = col
        row_max = row_min + row_size + 1
        col_max = col_min + col_size + 1
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

        if isinstance(data, np.ndarray):
            data = data[row_min:row_max, col_min:col_max]
        elif isinstance(data, xr.Dataset):
            data = data[self.label][row_min:row_max, col_min:col_max]
        else:
            raise Exception(".")

        if decode:
            data = self.decode(data)
        if mask is not None:
            data = np.ma.array(data, mask=mask)
        return data

    @_file_io('w')
    def write(self, io=None, write_kwargs=None, encode=True):
        """
        Writes data to disk into a target file or into a file associated
        with this object.

        Parameters
        ----------
        io : string or GeoTiffFile or NcFile
            File path to the file or IO class instance. The decorator deals with the distinction.
        write_kwargs : dict, optional
            Keyword arguments for writing function of IO class.
        """

        write_kwargs = write_kwargs if write_kwargs is not None else {}
        data = self.encode(self._data) if encode else self._data
        io.write(data, **write_kwargs)

    def encode(self, data):
        """
        Encodes data.

        Parameters
        ----------
        data : numpy.ndarray or xarray.DataArray, optional
            2D array-like object containing image pixel values.

        Returns
        -------
        data : numpy.ndarray or xarray.DataArray, optional
            Encoded array.
        """

        return data

    def decode(self, data):
        """
        Decodes data.

        Parameters
        ----------
        data : numpy.ndarray or xarray.DataArray, optional
            2D array-like object containing image pixel values.

        Returns
        -------
        data : numpy.ndarray or xarray.DataArray, optional
            Decoded array.
        """

        return data

    def plot(self, ax=None, proj=None, extent=None, extent_proj=None, cmap='viridis'):
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
        extent : 4 tuple, optional
            Extent of the data (x_min, x_max, y_min, y_max). If omitted, geometry bbox is assumed.
        extent_proj : cartopy.crs.Projection or its subclass, optional
            Spatial reference of the given extent. If omitted, the spatial reference in which
            the data are present is used.
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

        if extent_proj is None:
            extent_proj = proj

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection=proj)

        if extent is None:
            ll_x, ll_y, ur_x, ur_y = self.geometry.extent
            extent = ll_x, ur_x, ll_y, ur_y

        if isinstance(cmap, str):
            cmap = cm.get_cmap(cmap)

        ax.set_extent(extent, crs=extent_proj)
        ax.coastlines()

        ax.imshow(self._data, extent=extent, origin='upper', transform=proj, cmap=cmap)
        ax.set_aspect('equal', 'box')

        return ax

    @staticmethod
    def __check_data(data):
        """
        Checks array type and structure of data.

        Parameters
        ----------
        data : numpy.ndarray
            2D array-like object containing image pixel values.

        Returns
        -------
        bool
            If true, the given data fulfills all requirements for a `RasterData` object.
        """

        if data is not None:
            n = len(data.shape)
            if n != 2:
                err_msg = "Data has {} dimensions, but 2 dimensions are required."
                raise Exception(err_msg.format(n))
            return True
        else:
            return False


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





