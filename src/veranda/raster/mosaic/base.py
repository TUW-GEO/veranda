""" Collection of base classes for managing multi-dimensional I/O of raster data. """

import os
import copy
import gc
import abc
import warnings
import rioxarray  # this import is needed as an extension for xarray
import xarray as xr
import pandas as pd
import numpy as np
from affine import Affine
from typing import List, Tuple, Sequence

from geospade.tools import any_geom2ogr_geom
from geospade.tools import rel_extent
from geospade.crs import SpatialRef
from geospade.raster import RasterGeometry
from geospade.raster import MosaicGeometry
from geospade.raster import Tile
from geospade.raster import find_congruent_tile_id_from_tiles


class RasterAccess:
    """
    Helper class to build the link between indexes of the source array (access) and the target array (assignment).

    """
    def __init__(self, src_raster_geom, dst_raster_geom, src_root_raster_geom=None):
        """
        Constructor of `RasterAccess`.

        Parameters
        ----------
        src_raster_geom : geospade.raster.RasterGeometry
            Raster geometry representing the extent and indices of the data to access.
        dst_raster_geom : geospade.raster.RasterGeometry
            Raster geometry representing the extent and indices of the data to assign.
        src_root_raster_geom : geospade.raster.RasterGeometry, optional
            Raster geometry representing the origin to which `src_raster_geom` should be referred to. Defaults to None,
            i.e. the root parent of `src_raster_geom` is used.

        """
        src_root_raster_geom = src_root_raster_geom or src_raster_geom.parent_root
        origin = src_root_raster_geom.ul_x, src_root_raster_geom.ul_y
        min_col, min_row, max_col, max_row = rel_extent(origin, src_raster_geom.coord_extent,
                                                        src_raster_geom.x_pixel_size, src_raster_geom.y_pixel_size)
        self.src_window = (min_row, min_col, max_row, max_col)

        origin = dst_raster_geom.ul_x, dst_raster_geom.ul_y
        min_col, min_row, max_col, max_row = rel_extent(origin, src_raster_geom.coord_extent, src_raster_geom.x_pixel_size,
                                                        src_raster_geom.y_pixel_size)
        self.dst_window = (min_row, min_col, max_row, max_col)

    @property
    def src_row_slice(self) -> slice:
        """ Indices for the rows of the data to access. """
        return slice(self.src_window[0], self.src_window[2] + 1)

    @property
    def src_col_slice(self) -> slice:
        """ Indices for the columns of the data to access. """
        return slice(self.src_window[1], self.src_window[3] + 1)

    @property
    def dst_row_slice(self) -> slice:
        """ Indices for the rows of the data to assign. """
        return slice(self.dst_window[0], self.dst_window[2] + 1)

    @property
    def dst_col_slice(self) -> slice:
        """ Indices for the cols of the data to assign. """
        return slice(self.dst_window[1], self.dst_window[3] + 1)


class RasterData(metaclass=abc.ABCMeta):
    """
    Combines spatial information represented as a mosaic with raster mosaic given as 3D array data in memory or as
    geospatial files on disk.

    """
    def __init__(self, file_register, mosaic, data=None, stack_dimension='layer_id', stack_coords=None,
                 tile_dimension='tile_id', **kwargs):
        """
        Constructor of `RasterData`.

        Parameters
        ----------
        file_register : pd.Dataframe
            Data frame managing a stack/list of files containing the following columns:
                - 'filepath' : str
                    Full file path to a geospatial file.
                - `stack_dimension` : object
                    Specifies an ID to which layer a file belongs to, e.g. a layer counter or a timestamp. Must
                    correspond to `stack_dimension`.
                - `tile_dimension` : str
                    Tile name or ID to which tile a file belongs to.
        mosaic : geospade.raster.MosaicGeometry
            Mosaic representing the spatial allocation of the given files. The tiles of the mosaic have to match the
            ID's/names of the 'tile_id' column.
        data : xr.Dataset, optional
            Raster data stored in memory. It must match the spatial sampling and CRS of the mosaic, but not its spatial
            extent or tiling. Moreover, the dimension of the mosaic along the first dimension (stack dimension), must
            match the entries/filepaths in `file_register`.
        stack_dimension : str, optional
            Dimension/column name of the dimension, where to stack files or data over the same spatial region along
            (first axis), e.g. time, bands etc. Defaults to 'layer_id', i.e. the layer ID's are used as the main
            coordinates to stack the files.
        stack_coords : list, optional
            Additional columns of `file_register` to use as coordinates. Defaults to None, i.e. only coordinates along
            `stack_dimension` are used.
        tile_dimension : str, optional
            Dimension/column name of the dimension containing tile ID's in correspondence with the tiles in `mosaic`.
            Defaults to 'tile_id'.

        """
        self._file_register = file_register
        self._files = dict()
        self._mosaic = mosaic
        self._data = data
        self._data_geom = None if data is None else self.raster_geom_from_data(data, sref=mosaic.sref, name='0')
        self._file_dim = stack_dimension
        self._tile_dim = tile_dimension
        self._file_coords = [self._file_dim] if stack_coords is None else stack_coords

        if 'file_id' not in self._file_register.columns:
            self._file_register['file_id'] = [None] * len(self._file_register)

    @property
    def mosaic(self) -> MosaicGeometry:
        """ Mosaic geometry of the raster mosaic files. """
        return self._mosaic

    @property
    def data_geom(self) -> RasterGeometry:
        """ Raster/tile geometry of the raster mosaic files. """
        return self._data_geom

    @property
    def n_layers(self) -> int:
        """ Maximum number of layers. """
        return self._file_register.groupby([self._tile_dim])[self._tile_dim].count().max()

    @property
    def n_tiles(self) -> int:
        """ Number of tiles. """
        return len(self._mosaic.all_tiles)

    @property
    def file_register(self) -> pd.DataFrame:
        """ File register of the raster data object. """
        return self._file_register.drop(columns=['file_id']) \
            if 'file_id' in self._file_register.columns else self._file_register

    @property
    def filepaths(self) -> List[str]:
        """ Unique list of file paths stored in the file register. """
        return list(set(self._file_register['filepath']))

    @property
    def data_view(self) -> xr.Dataset:
        """ View on internal raster data. """
        return self._view_data()

    @abc.abstractmethod
    def load(self, *args, **kwargs):
        """ An abstract method for loading data either from disk or RAM. """
        pass

    @staticmethod
    def _sdims_from_data(data) -> List[str]:
        """
        Collects spatial dimensions of an xr.Dataset, which are assumed to be the last two.

        Parameters
        ----------
        data : xr.Dataset
            Raster data.

        Returns
        -------
        list :
            Spatial dimensions of the given xr.Dataset.

        """
        return list(data.dims)[-2:]

    @staticmethod
    def _pixel_sizes_from_data(data) -> Tuple[float, float]:
        """
        Collects pixel sizes of an xr.Dataset.

        Parameters
        ----------
        data : xr.Dataset
            Raster data.

        Returns
        -------
         x_pixel_size, y_pixel_size :
            Pixel size in X and Y direction.

        """
        sdims = RasterData._sdims_from_data(data)
        y_pixel_size = data[sdims[0]].data[0] - data[sdims[0]].data[1]
        x_pixel_size = data[sdims[1]].data[1] - data[sdims[1]].data[0]

        return x_pixel_size, y_pixel_size

    @staticmethod
    def _extent_from_data(data) -> Tuple[float, float, float, float]:
        """
        Computes the extent/bounding box of an xr.Dataset.

        Parameters
        ----------
        data : xr.Dataset
            Raster data.

        Returns
        -------
         4-tuple :
            Extent/bounding box of the given data (lower left x, lower left y, upper right x, upper right y).

        """
        sdims = RasterData._sdims_from_data(data)
        x_pixel_size, y_pixel_size = RasterData._pixel_sizes_from_data(data)

        return data[sdims[1]].data[0], data[sdims[0]].data[-1] - y_pixel_size, \
               data[sdims[1]].data[-1] + x_pixel_size, data[sdims[0]].data[0]

    @staticmethod
    def raster_geom_from_data(data, sref=None, **kwargs) -> RasterGeometry:
        """
        Creates a raster geometry from an xarray dataset.

        Parameters
        ----------
        data : xr.Dataset
            Raster data.
        sref : geospade.crs.SpatialRef, optional
            CRS of the mosaic if not given under the 'spatial_ref' variable.
        kwargs : dict
            Key-word arguments for the constructor of `RasterGeometry`.

        Returns
        -------
        raster_geom : geospade.raster.RasterGeometry
            Raster geometry representing the spatial extent of the xarray dataset.

        """
        coord_names = list(data.coords.keys())
        sref_coord = None
        for coord_name in coord_names:
            if data[coord_name].attrs.get('spatial_ref'):
                sref_coord = data[coord_name]
                break

        if sref_coord is None and sref is None:
            err_msg = "Neither the data contains CRS information nor the keyword."
            raise ValueError(err_msg)
        if sref_coord is not None:
            sref = SpatialRef(sref_coord.attrs.get('spatial_ref'))

        x_pixel_size, y_pixel_size = RasterData._pixel_sizes_from_data(data)
        extent = RasterData._extent_from_data(data)
        raster_geom = RasterGeometry.from_extent(extent, sref,
                                                 x_pixel_size=x_pixel_size, y_pixel_size=y_pixel_size, **kwargs)
        return raster_geom

    def apply_nan(self, nodatavals=None):
        """
        Converts no data values given as an attribute '_FillValue' or keyword `nodatavals` to np.nan.

        Parameters
        ----------
        nodatavals : dict
            Data variable name to no data value map.

        Notes
        -----
        This replacement implicitly converts the data format to float.

        """
        nodatavals = nodatavals or dict()
        if self._data is not None:
            for dvar in self._data.data_vars:
                dar = self._data[dvar]
                nodataval = dar.attrs.get('_FillValue', nodatavals.get(dvar, 0))
                self._data[dvar] = dar.where(dar != nodataval)

    def select(self, cmds, inplace=False) -> "RasterData":
        """
        Executes several select operations from a dict/JSON compatible set of commands.

        Parameters
        ----------
        cmds : list of 3-tuple
            List of tuples containing the select operator to execute, its positional arguments, and its key-word
            arguments.
        inplace : bool, optional
            If True, the current raster data object is modified.
            If False, a new raster data instance will be returned (default).

        Returns
        -------
        RasterData :
            Raster data object with a mosaic and a file register in compliance with the provided select operations.

        """
        if not inplace:
            new_raster_data = copy.deepcopy(self)
            return new_raster_data.select(cmds, inplace=True)

        for cmd in cmds:
            fun_name = cmd[0]
            args = cmd[1]
            kwargs = cmd[2]

            sref = kwargs.get('sref')
            if sref is not None:
                sref = SpatialRef(sref)
                kwargs['sref'] = sref

            getattr(self, fun_name)(*args, inplace=True, **kwargs)

        return self

    def select_tiles(self, tile_names, inplace=False) -> "RasterData":
        """
        Selects certain tiles from a raster data object.

        Parameters
        ----------
        tile_names : list of str
            Tile names/IDs.
        inplace : bool, optional
            If True, the current raster data object is modified.
            If False, a new raster data instance will be returned (default).

        Returns
        -------
        RasterData :
            Raster data object with a mosaic and a file register only consisting of the given tiles.

        """
        if not inplace:
            new_raster_data = copy.deepcopy(self)
            return new_raster_data.select_tiles(tile_names, inplace=True)

        self._file_register = self._file_register.loc[self._file_register[self._tile_dim].isin(tile_names)]
        self._mosaic.select_by_tile_names(tile_names, inplace=True)

        return self

    def select_layers(self, layer_ids, inplace=False) -> "RasterData":
        """
        Selects layers according to the given layer IDs.

        Parameters
        ----------
        layer_ids : list
            Layer IDs to select.
        inplace : bool, optional
            If True, the current raster data object is modified.
            If False, a new raster data instance will be returned (default).

        Returns
        -------
        RasterData :
            Raster data object with a file register only consisting of the given layer IDs.

        """
        if not inplace:
            new_raster_data = copy.deepcopy(self)
            return new_raster_data.select_layers(layer_ids, inplace=True)

        layer_ids_close = set(self._file_register[self._file_dim]) - set(layer_ids)
        self.close(layer_ids=layer_ids_close)
        self._file_register = self._file_register[self._file_register[self._file_dim].isin(layer_ids)]

        return self

    def select_px_window(self, row, col, height=1, width=1, inplace=False) -> "RasterData":
        """
        Selects the pixel coordinates according to the given pixel window.

        Parameters
        ----------
        row : int
            Top-left row number of the pixel window anchor.
        col : int
            Top-left column number of the pixel window anchor.
        height : int, optional
            Number of rows/height of the pixel window. Defaults to 1.
        width : int, optional
            Number of columns/width of the pixel window. Defaults to 1.
        inplace : bool, optional
            If True, the current raster data object is modified.
            If False, a new raster data instance will be returned (default).

        Returns
        -------
        RasterData :
            Raster data object with a data and a mosaic geometry only consisting of the intersected tile with the
            pixel window.

        Notes
        -----
        The mosaic will be only sliced if it consists of one tile to prevent ambiguities in terms of the definition
        of the pixel window.

        """
        if not inplace:
            new_raster_data = copy.deepcopy(self)
            return new_raster_data.select_px_window(row, col, height=height, width=width, inplace=True)

        if self._data_geom is not None:
            self._data_geom.slice_by_rc(row, col, height=height, width=width, inplace=True, name='0')
            if self._data_geom is None:
                wrn_msg = "Pixels are outside the extent of the raster mosaic."
                warnings.warn(wrn_msg)

        if len(self._mosaic.tiles) == 1:
            tile_oi = self._mosaic.tiles[0]
            tile_oi.slice_by_rc(row, col, height=height, width=width, inplace=True, name='0')
            tile_oi.active = True
            self._mosaic = self._mosaic.from_tile_list([tile_oi])

        return self

    def select_xy(self, x, y, sref=None, inplace=False) -> "RasterData":
        """
        Selects a pixel according to the given coordinate tuple.

        Parameters
        ----------
        x : number
            Coordinate in X direction.
        y : number
            Coordinate in Y direction.
        sref : geospade.crs.SpatialRef, optional
            CRS of the given coordinate tuple. Defaults to the CRS of the mosaic.
        inplace : bool, optional
            If True, the current raster data object is modified.
            If False, a new raster data instance will be returned (default).

        Returns
        -------
        RasterData :
            Raster data object with a file register and a mosaic only consisting of the intersected tile containing
            information on the location of the time series.

        """
        if not inplace:
            new_raster_data = copy.deepcopy(self)
            return new_raster_data.select_xy(x, y, sref=sref, inplace=True)

        if self._data_geom is not None:
            row, col = self._data_geom.xy2rc(x, y, sref=sref)
            self._data_geom.slice_by_rc(row, col, inplace=True, name='0')
            if self._data_geom is None:
                wrn_msg = "Coordinates are outside the spatial extent of the raster mosaic."
                warnings.warn(wrn_msg)

        tile_oi = self._mosaic.xy2tile(x, y, sref=sref)
        if tile_oi is not None:
            row, col = tile_oi.xy2rc(x, y, sref=sref)
            tile_oi.slice_by_rc(row, col, inplace=True, name='0')
            tile_oi.active = True
            self._mosaic = self._mosaic.from_tile_list([tile_oi])
            self._file_register = self._file_register[self._file_register[self._tile_dim] == tile_oi.parent_root.name]
        else:
            wrn_msg = "Coordinates are outside the spatial extent of the raster mosaic files."
            warnings.warn(wrn_msg)
            return

        return self

    def select_bbox(self, bbox, sref=None, inplace=False) -> "RasterData":
        """
        Selects tile and pixel coordinates according to the given bounding box.

        Parameters
        ----------
        bbox : list of 2 2-tuple
            Bounding box to select, i.e. [(x_min, y_min), (x_max, y_max)]
        sref : geospade.crs.SpatialRef, optional
            CRS of the given bounding box coordinates. Defaults to the CRS of the mosaic.
        inplace : bool, optional
            If True, the current raster data object is modified.
            If False, a new raster data instance will be returned (default).

        Returns
        -------
        RasterData :
            Raster data object with a file register and a mosaic only consisting of the intersected tiles.

        """
        if not inplace:
            new_raster_data = copy.deepcopy(self)
            return new_raster_data.select_bbox(bbox, sref=sref, inplace=True)
        return self.select_polygon(bbox, apply_mask=False, inplace=inplace)

    def select_polygon(self, polygon, sref=None, apply_mask=True, inplace=False) -> "RasterData":
        """
        Selects tile and pixel coordinates according to the given polygon.

        Parameters
        ----------
        polygon : ogr.Geometry
            Polygon specifying the pixels to collect.
        sref : geospade.crs.SpatialRef, optional
            CRS of the given bounding box coordinates. Defaults to the CRS of the mosaic.
        apply_mask : bool, optional
            True if pixels outside the polygon should be set to a no data value (default).
            False if every pixel withing the bounding box of the polygon should be included.
        inplace : bool, optional
            If True, the current raster data object is modified.
            If False, a new raster data instance will be returned (default).

        Returns
        -------
        RasterData :
            Raster data object with a file register and a mosaic only consisting of the intersected tiles.

        """
        if not inplace:
            new_raster_data = copy.deepcopy(self)
            return new_raster_data.select_polygon(polygon, sref=sref, apply_mask=apply_mask, inplace=True)

        sref = sref or self.mosaic.sref
        polygon = any_geom2ogr_geom(polygon, sref=sref)

        if self._data_geom is not None:
            self._data_geom.slice_by_geom(polygon, inplace=True, name='0')
            if self._data_geom is None:
                wrn_msg = "Polygon is outside the spatial extent of the raster mosaic."
                warnings.warn(wrn_msg)

        sliced_mosaic = self._mosaic.slice_by_geom(polygon, sref=sref, active_only=False, apply_mask=apply_mask,
                                                   inplace=False, name='0')
        if sliced_mosaic is None:
            wrn_msg = "Polygon is outside the spatial extent of the raster mosaic files."
            warnings.warn(wrn_msg)
            return

        self._mosaic = sliced_mosaic
        tile_names_oi = [tile.parent_root.name for tile in self._mosaic.tiles]
        self._file_register = self._file_register.loc[self._file_register[self._tile_dim].isin(tile_names_oi)]

        return self

    def _view_data(self) -> xr.Dataset:
        """ Returns a subset of the data according to the intersected mosaic and current layer ID's. """
        data = self._data
        if data is not None:
            origin = (self._data_geom.parent_root.ul_x, self._data_geom.parent_root.ul_y)
            min_col, min_row, max_col, max_row = rel_extent(origin, self._data_geom.coord_extent,
                                                            x_pixel_size=self._data_geom.x_pixel_size,
                                                            y_pixel_size=self._data_geom.y_pixel_size)

            xrars = dict()
            for dvar in data.data_vars:
                xrars[dvar] = data[dvar][..., min_row: max_row + 1, min_col:max_col + 1]
            data = xr.Dataset(xrars)

            if self._file_dim in data.coords:
                data = data.sel({self._file_dim: list(np.unique(self._file_register[self._file_dim]))})

        return data

    def _add_grid_mapping(self):
        """ Adds grid mapping information to the xr.Dataset. """
        if self._data is not None and self._data_geom is not None:
            self._data.rio.write_crs(self._data_geom.sref.wkt, inplace=True)
            self._data.rio.write_transform(Affine(*self._data_geom.geotrans), inplace=True)

    def close(self, layer_ids=None):
        """
        Closes open file handles and optionally data stored in RAM.

        Parameters
        ----------
        layer_ids : list, optional
            Layer IDs indicating the file handles which should be closed. Defaults to None, i.e. all file handles are
            closed.

        """
        if layer_ids is not None:
            bool_idxs = self._file_register[self._file_dim].isin(layer_ids)
            file_ids = set(self._file_register.loc[bool_idxs, 'file_id'])
            self._file_register.loc[bool_idxs, 'file_id'] = None
            file_ids = list(set(self._files.keys()) - file_ids)
        else:
            self._file_register['file_id'] = None
            file_ids = list(self._files.keys())

        self.__close_file_handles(file_ids)

    def clear_ram(self):
        """ Releases memory allocated by the internal data object. """
        self._data = None

    def __close_file_handles(self, file_ids):
        """
        Closes stored file handles and removes them from the internal dictionary.

        Parameters
        ----------
        file_ids : list of int
            List of file IDs.

        """
        for file_id in file_ids:
            self._files[file_id].close()
            del self._files[file_id]

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    def __deepcopy__(self, memo):
        """
        Deepcopy method of the `RasterData` class.

        Parameters
        ----------
        memo : dict

        Returns
        -------
        RasterData
            Deepcopy of raster data.

        """

        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == '_files':  # skip existing file pointers, can't be copied
                setattr(result, k, dict())
            else:
                setattr(result, k, copy.deepcopy(v, memo))

        result._file_register['file_id'] = None  # remove existing file IDs
        return result

    def _repr_html_(self) -> str:
        """ HTML table representation of the file register of a raster data instance.  """
        return self.file_register.style.set_properties(subset=['filepath'], **{'text-align': 'right'})._repr_html_()

    def __repr__(self) -> str:
        """ General string representation of a raster data instance. """
        return f"{self.__class__.__name__}({self._file_dim}, {self.mosaic.__class__.__name__}):\n\n" \
               f"{repr(self.file_register)}"


class RasterDataReader(RasterData):
    """ Allows to read and manage a stack of raster data. """
    def __init__(self, file_register, mosaic, stack_dimension='layer_id', stack_coords=None, tile_dimension='tile_id'):
        """
        Constructor of `RasterDataReader`.

        Parameters
        ----------
        file_register : pd.Dataframe
            Data frame managing a stack/list of files containing the following columns:
                - 'filepath' : str
                    Full file path to a geospatial file.
                - `stack_dimension` : object
                    Specifies an ID to which layer a file belongs to, e.g. a layer counter or a timestamp. Must
                    correspond to `stack_dimension`.
                - `tile_dimension` : str
                    Tile name or ID to which tile a file belongs to.
        mosaic : geospade.raster.MosaicGeometry
            Mosaic representing the spatial allocation of the given files. The tiles of the mosaic have to match the
            ID's/names of the `tile_dimension` column.
        stack_dimension : str, optional
            Dimension/column name of the dimension, where to stack the files along (first axis), e.g. time, bands etc.
            Defaults to 'layer_id', i.e. the layer ID's are used as the main coordinates to stack the files.
        stack_coords : list, optional
            Additional columns of `file_register` to use as coordinates. Defaults to None, i.e. only coordinates along
            `stack_dimension` are used.
        tile_dimension : str, optional
            Dimension/column name of the dimension containing tile ID's in correspondence with the tiles in `mosaic`.
            Defaults to 'tile_id'.

        """
        super().__init__(file_register, mosaic, stack_dimension=stack_dimension, stack_coords=stack_coords,
                         tile_dimension=tile_dimension)

    @abc.abstractmethod
    def read(self, *args, auto_decode=False, decoder=None, decoder_kwargs=None, **kwargs):
        """
        Read data from disk.

        Parameters
        ----------
        auto_decode : bool, optional
            True if data should be decoded according to the information available in its metadata. Defaults to False.
        decoder : callable, optional
            Function allowing to decode data read from disk.
        decoder_kwargs : dict, optional
            Keyword arguments for the decoder.

        """
        pass

    @abc.abstractmethod
    def _to_xarray(self, *args, **kwargs):
        """
        Converts data read from disk to an xarray dataset.

        Returns
        -------
        xr.Dataset

        """
        pass

    def load(self, *args, **kwargs):
        """
        Loads mosaic either from disk or RAM.

        Parameters
        ----------
        args :
            Positional arguments for the `read` method.
        kwargs :
            Key-word arguments for the `read` method.

        """
        if self._data is not None and self._data_geom is not None:
            self._data = self.data_view
            self._data_geom.parent = None
        else:
            self.read(*args, **kwargs)

    @staticmethod
    def _create_tile_and_layer_info_from_files(filepaths, tile_class, file_class,
                                               file_class_kwargs) -> Tuple[List[Tile], list, list]:
        """
        Loops over a given list of files to assign a tile and layer to each file and creates the corresponding indexes.

        Parameters
        ----------
        filepaths : list of str
            List of file paths.
        tile_class : class
            Class constructor for a tile class.
        file_class : class
            Class constructor for a file class.
        file_class_kwargs : dict
            Keyword arguments for calling `file_class`.

        Returns
        -------
        tiles : list of Tile
            Unique set of tiles representing all input file paths.
        tile_ids : list of str
            List of tile ids containing one ID per file.
        layer_ids : list of str
            List of layer ids containing one ID per file.

        """
        file_class_kwargs = file_class_kwargs or dict()
        tile_ids = []
        layer_ids = []
        tiles = []
        tile_idx = 0
        for filepath in filepaths:
            with file_class(filepath, 'r', **file_class_kwargs) as f:
                sref_wkt = f.sref_wkt
                geotrans = f.geotrans
                n_rows, n_cols = f.raster_shape
            curr_tile = tile_class(n_rows, n_cols, sref=SpatialRef(sref_wkt), geotrans=geotrans, name=str(tile_idx))
            curr_tile_id = find_congruent_tile_id_from_tiles(curr_tile, tiles)
            if curr_tile_id is None:
                tiles.append(curr_tile)
                curr_tile_id = str(tile_idx)
                tile_idx += 1

            tile_ids.append(curr_tile_id)
            # define the layer ID as the next index of all filepaths, which have already been assigned to one tile
            layer_id = sum(np.array(tile_ids) == curr_tile_id) + 1
            layer_ids.append(layer_id)

        return tiles, tile_ids, layer_ids


class RasterDataWriter(RasterData):
    """ Allows to write and manage a stack of raster data. """
    def __init__(self, mosaic, file_register=None, data=None, stack_dimension='layer_id', stack_coords=None,
                 tile_dimension='tile_id', dirpath=None, fn_pattern='{layer_id}.xyz', fn_formatter=None):
        """
        Constructor of `RasterDataWriter`.

        Parameters
        ----------

        mosaic : geospade.raster.MosaicGeometry
            Mosaic representing the spatial allocation of the given files. The tiles of the mosaic have to match the
            ID's/names of the `tile_dimension` column.
        file_register : pd.Dataframe, optional
            Data frame managing a stack/list of files containing the following columns:
                - 'filepath' : str
                    Full file path to a geospatial file.
                - `stack_dimension` : object
                    Specifies an ID to which layer a file belongs to, e.g. a layer counter or a timestamp. Must
                    correspond to `stack_dimension`.
                - `tile_dimension` : str
                    Tile name or ID to which tile a file belongs to.
            If it is None, then the constructor tries to create a file from other keyword arguments, i.e. `data`,
            `dirpath`, `fn_pattern`, and `fn_formatter`.
        data : xr.Dataset, optional
            Raster data stored in memory. It must match the spatial sampling and CRS of the mosaic, but not its spatial
            extent or tiling. Moreover, the dimension of the data along the first dimension (stack dimension), must
            match the entries/filepaths in `file_register`.
        stack_dimension : str, optional
            Dimension/column name of the dimension, where to stack the files along (first axis), e.g. time, bands etc.
            Defaults to 'layer_id', i.e. the layer ID's are used as the main coordinates to stack the files.
        stack_coords : list, optional
            Additional columns of `file_register` to use as coordinates. Defaults to None, i.e. only coordinates along
            `stack_dimension` are used.
        tile_dimension : str, optional
            Dimension/column name of the dimension containing tile ID's in correspondence with the tiles in `mosaic`.
            Defaults to 'tile_id'.
        dirpath : str, optional
            Directory path to the folder where the GeoTIFF files should be written to. Defaults to None, i.e. the
            current working directory is used.
        fn_pattern : str, optional
            Pattern for the filename of the new GeoTIFF files. To fill in specific parts of the new file name with
            information from the file register, you can specify the respective file register column names in curly
            brackets and add them to the pattern string as desired. Defaults to '{layer_id}.tif'.
        fn_formatter : dict, optional
            Dictionary mapping file register column names with functions allowing to encode their values as strings.

        """
        fn_formatter = fn_formatter or dict()
        if file_register is None and data is None:
            err_msg = "Either a file register ('file_register') or an xarray dataset ('data') has to be provided."
            raise ValueError(err_msg)
        elif file_register is None and data is not None:
            file_register = RasterDataWriter._file_register_from_data(data, stack_dimension)

        if tile_dimension not in file_register.columns:
            file_register = RasterDataWriter._add_tile_names_to_file_register(file_register, mosaic, tile_dimension)

        if stack_dimension not in file_register.columns:
            file_register = RasterDataWriter._add_stack_dims_to_file_register(file_register, stack_dimension, data)

        if 'filepath' not in file_register.columns:
            file_register = RasterDataWriter._add_filepaths_to_file_register(file_register, dirpath,
                                                                             fn_pattern, fn_formatter)

        super().__init__(file_register, mosaic, data=data, stack_dimension=stack_dimension, stack_coords=stack_coords,
                         tile_dimension=tile_dimension)

    @abc.abstractmethod
    def write(self, data, encoder=None, encoder_kwargs=None, overwrite=False, **kwargs):
        """
        Writes a certain chunk of data to disk.

        Parameters
        ----------
        data : xr.Dataset
            Data chunk to be written to disk or being appended to existing data.
        encoder : callable, optional
            Function allowing to encode data before writing it to disk.
        encoder_kwargs : dict, optional
            Keyword arguments for the encoder.
        overwrite : bool, optional
            True if data should be overwritten, False if not (default).

        """
        pass

    @abc.abstractmethod
    def export(self, apply_tiling=False, encoder=None, encoder_kwargs=None, overwrite=False, **kwargs):
        """
        Writes all the internally stored data to disk.

        Parameters
        ----------
        apply_tiling : bool, optional
            True if the internal data should be tiled according to the mosaic.
            False if the internal data composes a new tile and should not be tiled (default).
        encoder : callable, optional
            Function allowing to encode data before writing it to disk.
        encoder_kwargs : dict, optional
            Keyword arguments for the encoder.
        overwrite : bool, optional
            True if data should be overwritten, False if not (default).

        """
        pass

    @classmethod
    def from_xarray(cls, data, file_register, mosaic=None, **kwargs) -> "RasterDataWriter":
        """
        Converts an xarray dataset and a file register to a `RasterDataWriter` instance.

        Parameters
        ----------
        data : xr.Dataset
            Raster data.
        file_register : pd.Dataframe, optional
            Data frame managing a stack/list of files containing the following columns:
                - 'filepath' : str
                    Full file path to a geospatial file.
                - `stack_dimension` : object
                    Specifies an ID to which layer a file belongs to, e.g. a layer counter or a timestamp. Must
                    correspond to `stack_dimension`.
                - `tile_dimension` : str
                    Tile name or ID to which tile a file belongs to.
        mosaic : geospade.raster.MosaicGeometry, optional
            Mosaic representing the spatial allocation of the given files. The tiles of the mosaic have to match the
            ID's/names of the `tile_dimension` column. If it is None, a one-tile mosaic will be created from the given
            data.
        kwargs :
            Key-word arguments for the `RasterDataWriter` constructor.

        Returns
        -------
        RasterDataWriter

        """
        if mosaic is None:
            mosaic = cls._mosaic_from_data(data)
        return cls(mosaic, file_register=file_register, data=data, **kwargs)

    def load(self, *args, **kwargs):
        """ Loads data from RAM. """
        if self._data is not None and self._data_geom is not None:
            self._data = self.data_view
            self._data_geom.parent = None

    @staticmethod
    def _mosaic_from_data(data, sref=None) -> MosaicGeometry:
        """
        Creates a default mosaic from a given xarray dataset.

        Parameters
        ----------
        data : xr.Dataset
            Raster data.
        sref : geospade.crs.SpatialRef, optional
            CRS of the mosaic if not given under the 'spatial_ref' variable of `data`.

        Returns
        -------
        MosaicGeometry :
            Mosaic with one tile '0'.

        """
        raster_geom = RasterDataWriter.raster_geom_from_data(data, sref=sref)
        n_rows, n_cols = raster_geom.shape
        tile = Tile(n_rows, n_cols, raster_geom.sref, geotrans=raster_geom.geotrans,
                    name='0')
        return MosaicGeometry.from_tile_list([tile], check_consistency=False)

    @staticmethod
    def _file_register_from_data(data, stack_dimension) -> pd.DataFrame:
        """
        Creates a file register with stack dimension coordinates in one column.

        Parameters
        ----------
        data : xr.Dataset
            Raster data.
        stack_dimension : str
            Dimension name representing the stack dimension of the file register.

        Returns
        -------
        pd.DataFrame :
            File register with stack dimension coordinates in one column.

        """
        layers = data[stack_dimension]
        file_register_dict = dict()
        file_register_dict[stack_dimension] = layers
        return pd.DataFrame(file_register_dict)

    @staticmethod
    def _add_tile_names_to_file_register(file_register, mosaic, tile_dimension='tile_id') -> pd.DataFrame:
        """
        Adds all tiles of a mosaic to the given file register under the column `tile_dimension`.

        Parameters
        ----------
        file_register : pd.Dataframe
            File register to add the tile names to.
        mosaic : MosaicGeometry
            Mosaic to extract tile information from.
        tile_dimension : str, optional
            Dimension/column name of the dimension containing tile ID's in correspondence with the tiles in `mosaic`.
            Defaults to 'tile_id'.

        Returns
        -------
        file_register : pd.Dataframe
            Modified file register containing the tile names of the given mosaic.

        """
        n_entries = len(file_register)
        tile_names = mosaic.all_tile_names
        n_tiles = len(tile_names)
        file_register = pd.DataFrame(np.repeat(file_register.values, n_tiles, axis=0),
                                     columns=file_register.columns)
        file_register[tile_dimension] = np.repeat([tile_names], n_entries, axis=0).flatten()

        return file_register

    @staticmethod
    def _add_stack_dims_to_file_register(file_register, stack_dimension, data=None):
        """
        Adds coordinate values along the stack dimension of the data to the given file register. If no data is given,
        then a simple increment is used to represent the stack dimension.

        Parameters
        ----------
        file_register : pd.Dataframe
            File register to add the stack dimension coordinates to.
        stack_dimension : str
            Dimension name representing the stack dimension of the file register.
        data : xr.Dataset, optional
            Raster data.

        Returns
        -------
        file_register : pd.DataFrame
            File register with stack dimension coordinates added to.

        """
        n_entries = len(file_register)
        if data is not None:
            layers = data[stack_dimension]
            n_layers = len(layers)
            file_register = pd.DataFrame(np.repeat(file_register.values, n_layers, axis=0),
                                         columns=file_register.columns)
            file_register[stack_dimension] = np.repeat([layers], n_entries, axis=0).flatten()
        else:
            layers = list(range(1, n_entries + 1))
            file_register[stack_dimension] = layers

        return file_register

    @staticmethod
    def _add_filepaths_to_file_register(file_register, dirpath=None,
                                        fn_pattern='{layer_id}.rd', fn_formatter=None) -> pd.DataFrame:
        """
        Adds a column containing the file paths to new datasets following the naming
        convention derived from `dirpath`, `fn_pattern`, and `fn_formatter` to the given file register.

        Parameters
        ----------
        file_register : pd.Dataframe
            File register to add the stack dimension coordinates to.
        dirpath : str, optional
            Directory path to the folder where the GeoTIFF files should be written to. Defaults to none, i.e. the
            current working directory is used.
        fn_pattern : str, optional
            Pattern for the filename of the new files. To fill in specific parts of the new file name with
            information from the file register, you can specify the respective file register column names in curly
            brackets and add them to the pattern string as desired. Defaults to '{layer_id}.rd'.
        fn_formatter : dict, optional
            Dictionary mapping file register column names with functions allowing to encode their values as strings.

        Returns
        -------
        file_register : pd.DataFrame
            File register with an additional column containing the file paths to new datasets following the naming
            convention derived from `dirpath`, `fn_pattern`, and `fn_formatter`.

        """
        dirpath = dirpath or os.getcwd()
        fn_formatter = fn_formatter or dict()
        filepaths = []
        for _, row in file_register.iterrows():
            fn_entries = dict()
            for k, v in row.items():
                if k in fn_formatter.keys():
                    v = fn_formatter[k](v)
                if isinstance(v, str):
                    fn_entries[k] = v

            filename = fn_pattern.format(**fn_entries)
            filepaths.append(os.path.join(dirpath, filename))
        file_register['filepath'] = filepaths

        return file_register


if __name__ == '__main__':
    pass
