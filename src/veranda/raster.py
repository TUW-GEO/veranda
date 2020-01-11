import os
import copy
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
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
from geospade.operation import any_geom2ogr_geom
from geospade.operation import xy2ij
from geospade.operation import ij2xy


def _file_io(mode='r'):
    """
    Decorator which checks whether parsed file arguments are acutual IO-Classes
    such as NcFile or GeoTiffFile. If string is parsed as file, then
    a corresponding IO-Class object is parsed into the original function
    """
    def _inner_file_io(func):
        tiff_ext = ('.tiff', '.tif', '.geotiff')
        netcdf_ext = ('.nc', '.netcdf', '.ncdf')

        def wrapper(self, *args, **kwargs):
            io_kwarg = kwargs.get('io', None)
            io_kwargs = kwargs.get('io_kwargs', None)
            io_arg = args[0]

            io = None
            if isinstance(io_arg, str):
                file_ext = os.path.splitext(io_arg)[1].lower()
                if file_ext in tiff_ext:
                    io_class = GeoTiffFile
                elif file_ext in netcdf_ext:
                    io_class = NcFile
                elif io_kwarg is not None:
                    io_class = io_kwarg
                else:
                    raise IOError('File format not supported.')
            else:
                io_class = None
                io = io_arg  # argument seems to be a self defined io class

            create_io = False
            if hasattr(self, 'io'):
                if self.io is None:
                    if io is None:
                        create_io = True
                else:
                    if self.io.mode != mode:
                        create_io = True
                io = self.io if self.io is not None else io
            else:
                create_io = True

            if create_io:
                io_kwargs = dict() if io_kwargs is None else io_kwargs
                geotransform = io_kwargs.get('geotransform', self.geometry.gt)
                spatialref = io_kwargs.get('spatialref', self.geometry.wkt)
                add_kwargs = {'geotransform': geotransform,
                              'spatialref': spatialref}
                io_kwargs.update(add_kwargs)
                io = io_class(filename=io_arg, mode=mode, **io_kwargs)

            return func(self, io, **kwargs)
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


def convert_dtype(data, dtype, xs=None, ys=None, zs=None, band=1, x_dim='x', y_dim='y', z_dim='time'):
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

    if dtype == "xarray":
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
            raise DataTypeMismatch(type(data), dtype)

    elif dtype == "numpy":
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
            raise DataTypeMismatch(type(data), dtype)
    elif dtype == "dataframe":
        xr_ds = convert_dtype(data, 'xarray', xs=xs, ys=ys, zs=zs, band=band, x_dim=x_dim, y_dim=y_dim, z_dim=z_dim)
        converted_data = xr_ds.to_dataframe()
    else:
        raise DataTypeMismatch(type(data), dtype)

    return converted_data


class RasterData(object):
    """
    This class represents georeferenced raster data. Two main components of this
    data structure is geometry and actual data. The geometry of the raster data,
    defines all geometrical properties of the data like extent, pixel size,
    location and orientation in a spatial reference system. The geometry is a
    RasterGeometry object which implements several geometrical operations.
    The other component is data, which is a simple numpy ndarray, that contains
    the actual values of the raster file. Every RasterData has its own instance
    of some IO-Class such as GeoTiffFile or NcFile which is used for IO
    operations. The advantage of this class is, that we can perform geometrical
    operations without even having to load the pixel values from hard-drive, and
    then later load only data we need.
    """

    def __init__(self, rows, cols, sref, gt,
                 data=None, parent=None, io=None):
        """
        Basic constructor

        Parameters
        ----------
        rows : int
            number of pixel rows
        cols : int
            number pixel columns
        sref: pyraster.spatial_ref.SpatialRef
            spatial reference of the data
        gt : 6-tuple
            GDAL Geotransform 'matrix'
        data : array-like (optional)
            Image pixel values (data)
        parent : RasterData ,optional (default: None)
            This attribute is used to trace back to the actual file for the
            io operations. For example by cropping existing RasterData object,
            that was created from existing file, a new RasterData object is
            created with parent set to the RasterData object we are cropping.
            This way we can access the file, that contains data in our
            particular region of interest, without having to load the complete
            file into memory only to discard most of it later. If parent is
            None, then this object is associated with the file that contains the
            data.
        io : pyraster.geotiff.GeoTiffFile or pyraster.netcdf.NcFile
            Instance of a IO Class that is associated with a file that contains
            the data.

        """
        self.geometry = RasterGeometry(rows, cols, sref, gt)
        self.data = data
        self.parent = parent
        self.io = io

    @classmethod
    def from_array(cls, sref, gt, data, parent=None):
        """
        Creates a RasterData object from an array-like object.

        Parameters
        ----------
        sref : pyraster.spatial_ref.SpatialRef
            Spatial reference of the data geometry
        gt : 6-tuple
            GDAL geotransform 'matrix'
        data : array-like
            pixel values
        parent : RasterData or None, optional
            parent RasterData object


        Returns
        -------
        RasterData

        """
        rows, cols = data.shape
        return cls(rows, cols, sref, gt, data, parent=parent)

    @classmethod
    @_file_io(mode=None)
    def from_file(cls, io, read=False, io_kwargs=None):
        """
        Creates a RasterData object from a file

        Parameters
        ----------
        file : string or GeoTiffFile or NcFile
            path to the file or IO Classes instance. The decorator deals with
            the distinction
        io_kwargs: dict
            potential arguments for IO Class instance. (Happens in decorator)

        Returns
        -------
        RasterData

        """
        spatialref = io_kwargs.get('spatialref', None)
        geotransform = io_kwargs.get('geotransform', None)
        if spatialref is None or geotransform is None:
            io._open('r')
            spatialref = io.spatialref
            geotransform = io.spatialref

        sref = SpatialRef(spatialref)
        data = io.read(return_tags=False) if read else None
        rows, cols = data.shape
        return cls(rows, cols, sref, geotransform, data=data, io=io)

    def crop(self, geometry, inplace=True):
        """
        Crops the image by geometry. Inplace determines, whether a new object
        is returned or the cropping happens on this current object.
        The resulting RasterData object contains no-data value in pixels that
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
            data = self.data[min_row:(max_row + 1), min_col:(max_col + 1)]
            mask = rasterise_polygon(new_geom.geometry)
            data = np.ma.array(data, mask=mask)

        if inplace:
            self.geometry = new_geom
            self.parent = self
            self.data = data
        else:
            return RasterData.from_array(self.geometry.sref,
                                         new_geom.gt,
                                         data,
                                         parent=self)

    @_file_io('r')
    def load(self, io):
        """
        Reads the data from disk and assigns the resulting array to the
        self.data attribute
        Returns
        -------
        None
        """
        self.io = self._get_orig_io()
        g = self.geometry
        sub_rect = (g.ll_x, g.ll_y, g.width, g.height)
        self.data = self.io.read(sub_rect=sub_rect)

    @_file_io
    def write(self, file=None):
        """
        Writes data to on hard-drive into target file or into file associated
        with this object.

        Parameters
        ----------
        file: IO-Class or string
            either initialized instance of one of the IO Classes (GeoTiffFile,
            NcFile ..) or string representing path to the file on filesystem

        Returns
        -------
        None
        """
        # nodata ??? set / externalize ???
        file.write(self.data, band=1)

    def _get_orig_io(self):
        """
        Follows the parent links to until it reaches a RasterData object with
        valid io attribute
        Returns
        -------
        IO Class
            instance of an IO Class of an RasterData object that is associated
            with an file

        """
        parent = self.parent
        while parent.io is None:
            parent = parent.parent
        return parent.io

    def plot(self, fig=None, map_crs=None, extent=None, extent_crs=None,
             cmap='viridis'):
        """
        Plots the raster data to a map that uses map_crs projection
        if provided. When not, the map projection defaults to spatial reference
        in which the data are provided (self.geometry.sref). The extent of the
        data is specified by 'extent'. If extent is not provided, it defaults
        to the bbox of the data's geometry. If provided, one can also specify
        spatial reference of the extent that is beeing parsed, otherwise is is
        assumed that the spatial reference of the extent is same as the spatial
        reference of the data.
        :param fig: pyplot Figure (optional)
            Target figure, into which will the data be plotted
        :param map_crs: cartopy.crs.Projection or its subclass (optional)
            Spatial reference of the map. The Figure is going to be drawn in
            this spatial reference. If omitted, the spatial reference in which
            the data are present is assumed.
        :param extent: 4 - tuple (optional)
            Extent of the data. If omitted, geometry bbox is assumed.
        :param extent_crs: cartopy.crs.Projection or its subclass (optional)
            Spatial reference of parsed extent. If omitted, the spatial
            reference in which the data are present is assumed.
        :param cmap matplotlib.colors.Colormap or string
            Colormap which is to be used to plot the data
        """

        data_crs = self.geometry.to_cartopy_crs()

        if map_crs is None:
            map_crs = data_crs
        if fig is None or len(fig.get_axes()) == 0:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection=map_crs)
        else:
            ax = fig.gca()

        if extent_crs is None:
            extent_crs = data_crs
        if extent is None:
            ll_x, ll_y, ur_x, ur_y = self.geometry.geometry.bounds
            extent = ll_x, ur_x, ll_y, ur_y

        if isinstance(cmap, str):
            cmap = cm.get_cmap(cmap)

        ax.set_extent(extent, crs=extent_crs)
        ax.coastlines()

        ax.imshow(self.data, extent=extent, origin='upper', transform=data_crs,
                  cmap=cmap)
        ax.set_aspect('equal', 'box')

        return fig

    def __check_data(self):
        if self.data is not None:
            n = len(self.data.shape)
            if n != 2:
                raise Exception('Data has {} dimensions, but 2 dimensions are required.'.format(n))
            return True
        else:
            return False


class RasterStack(object):
    def __init__(self, raster_datas, data=None, geometry=None, parent=None, io=None):
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

    @any_geom2ogr_geom
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

    @any_geom2ogr_geom
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





