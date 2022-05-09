import copy
import gc
import abc
import warnings
import rioxarray  # this import is needed as an extension for xarray
import xarray as xr
from affine import Affine

from geospade.tools import any_geom2ogr_geom
from geospade.tools import rel_extent
from geospade.crs import SpatialRef
from geospade.raster import RasterGeometry
from geospade.raster import MosaicGeometry


class RasterAccess:
    """
    Helper class to build the link between indexes of the source array/tile (access) and the target array/tile
    (assignment).

    """
    def __init__(self, src_raster_geom, dst_raster_geom, src_root_raster_geom=None):
        """
        Constructor of `RasterAccess`.

        Parameters
        ----------
        src_raster_geom : geospade.raster.RasterGeometry
            Raster geometry representing the extent and indices of the mosaic to access.
        dst_raster_geom : geospade.raster.RasterGeometry
            Raster geometry representing the extent and indices of the mosaic to assign.
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
    def src_row_slice(self):
        """ slice : Indices for the rows of the mosaic to access. """
        return slice(self.src_window[0], self.src_window[2] + 1)

    @property
    def src_col_slice(self):
        """ slice : Indices for the columns of the mosaic to access. """
        return slice(self.src_window[1], self.src_window[3] + 1)

    @property
    def dst_row_slice(self):
        """ slice : Indices for the rows of the mosaic to assign. """
        return slice(self.dst_window[0], self.dst_window[2] + 1)

    @property
    def dst_col_slice(self):
        """ slice : Indices for the cols of the mosaic to assign. """
        return slice(self.dst_window[1], self.dst_window[3] + 1)


class RasterData(metaclass=abc.ABCMeta):
    """
    Combines spatial information represented as a mosaic with raster mosaic given as 3D array mosaic in memory or as
    geospatial files on disk.

    """
    def __init__(self, file_register, mosaic, data=None, file_dimension='layer_id', file_coords=None, **kwargs):
        """
        Constructor of `RasterData`.

        Parameters
        ----------
        file_register : pd.Dataframe
            Data frame managing a stack/list of files containing the following columns:
                - 'filepath' : str
                    Full file path to a geospatial file.
                - 'layer_id' : int
                    Specifies an ID to which layer a file belongs to.
                - 'tile_id' : str or int
                    Tile name or ID to which tile a file belongs to.
        mosaic : geospade.raster.MosaicGeometry
            Mosaic representing the spatial allocation of the given files. The tiles of the mosaic have to match the
            ID's/names of the 'tile_id' column.
        data : xr.Dataset, optional
            Raster mosaic stored in memory. It must match the spatial sampling and CRS of the mosaic, but not its spatial
            extent or tiling. Moreover, the dimension of the mosaic along the first dimension (stack/file dimension), must
            match the entries/filepaths in `file_register`.
        file_dimension : str, optional
            Dimension/column name of the dimension, where to stack the files along (first axis), e.g. time, bands etc.
            Defaults to 'layer_id', i.e. the layer ID's are used as the main coordinates to stack the files.
        file_coords : list, optional
            Additional columns of `file_register` to use as coordinates. Defaults to None, i.e. only coordinates along
            `file_dimension` are used.

        """
        self._file_register = file_register
        self._drivers = dict()
        self._mosaic = mosaic
        self._data = data
        self._data_geom = None if data is None else self.raster_geom_from_data(data, sref=mosaic.sref)
        self._file_dim = file_dimension
        self._file_coords = [self._file_dim] if file_coords is None else file_coords

        if 'driver_id' not in self._file_register.columns:
            self._file_register['driver_id'] = [None] * len(self._file_register)

    @property
    def mosaic(self):
        """ geospade.raster.MosaicGeometry : Mosaic geometry of the raster mosaic files. """
        return self._mosaic

    @property
    def data_geom(self):
        """ geospade.raster.Tile : Mosaic geometry of the raster mosaic files. """
        return self._data_geom

    @property
    def n_layers(self):
        """ int : Number of layers. """
        return max(self._file_register['layer_id']) + 1

    @property
    def n_tiles(self):
        """ int : Number of tiles. """
        return len(self._mosaic.all_tiles)

    @property
    def shape(self):
        """
        3-tuple or 2-tuple : Shape of the file-based mosaic structure, i.e. a tuple of the number of layers and the
        number of tiles. If the mosaic is regular, then the shape is a 3-tuple, where the last two entries are the
        number of tiles in x- and y-direction.

        """
        shape = None
        if getattr(self._mosaic, 'shape') is not None:
            shape = tuple([self.n_layers] + list(self._mosaic.shape))
        else:
            shape = (self.n_layers, self.n_tiles)

        return shape

    @property
    def file_register(self):
        """ pd.Dataframe : File register of the raster mosaic object. """
        return self._file_register.drop(columns=['driver_id'])

    @property
    def data(self):
        """ xr.Dataset : View on internal raster mosaic. """
        return self._view_data()

    @abc.abstractmethod
    def load(self, *args, **kwargs):
        """ An abstract method for loading mosaic either from disk or RAM. """
        pass

    @staticmethod
    def raster_geom_from_data(data, sref=None, **kwargs):
        """
        Creates a raster geometry from an xarray dataset.

        Parameters
        ----------
        data : xr.Dataset
            Raster mosaic.
        sref : geospade.crs.SpatialRef, optional
            CRS of the mosaic if not given under the 'spatial_ref' variable.

        Returns
        -------
        raster_geom : geospade.raster.RasterGeometry
            Raster geometry representing the spatial extent of the xarray dataset.

        """
        ds_sref = data.get('spatial_ref')
        if ds_sref is None and sref is None:
            err_msg = "Neither the mosaic contains CRS information nor the keyword"
            raise ValueError(err_msg)
        if ds_sref is not None:
            sref = SpatialRef(ds_sref.attrs.get('spatial_ref'))
        x_pixel_size = data.x.data[1] - data.x.data[0]
        y_pixel_size = data.y.data[0] - data.y.data[1]
        extent = [data.x.data[0], data.y.data[-1] - y_pixel_size, data.x.data[-1] + x_pixel_size, data.y.data[0]]
        raster_geom = RasterGeometry.from_extent(extent, sref,
                                                 x_pixel_size=x_pixel_size, y_pixel_size=y_pixel_size, **kwargs)
        return raster_geom

    def apply_nan(self):
        """
        Converts no mosaic values given as an attribute 'fill_value' to np.nan. Note that this replacement implicitly
        converts the mosaic format to float.

        """
        if self._data is not None:
            for dvar in self._data.data_vars:
                dar = self._data[dvar]
                self._data[dvar] = dar.where(dar != dar.attrs['fill_value'])

    def select(self, cmds, inplace=False):
        """
        Executes several select operations from a dict/JSON compatible set of commands.

        Parameters
        ----------
        cmds : list of 2-tuple

        inplace : bool, optional
            If true, the current raster mosaic object is modified.
            If false, a new raster mosaic instance will be returned (default).

        Returns
        -------
        RasterData :
            Raster mosaic object with a mosaic and a file register in compliance with the provided select operations.

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

    def select_tiles(self, tile_names, inplace=False):
        """
        Selects a certain tile from a raster mosaic object.

        Parameters
        ----------
        tile_names : list of str or int
            Tile names/IDs.
        inplace : bool, optional
            If true, the current raster mosaic object is modified.
            If false, a new raster mosaic instance will be returned (default).

        Returns
        -------
        RasterData :
            Raster mosaic object with a mosaic and a file register only consisting of the given tiles.

        """
        if not inplace:
            new_raster_data = copy.deepcopy(self)
            return new_raster_data.select_tiles(tile_names, inplace=True)

        self._file_register = self._file_register.loc[self._file_register['geom_id'].isin(tile_names)]
        self._mosaic.select_by_tile_names(tile_names)

        return self

    def select_layers(self, layer_ids, inplace=False):
        """
        Selects layers according to the given layer IDs.

        Parameters
        ----------
        layer_ids : list
            Layer IDs to select.
        inplace : bool, optional
            If true, the current raster mosaic object is modified.
            If false, a new raster mosaic instance will be returned (default).

        Returns
        -------
        RasterData :
            Raster mosaic object with a file register only consisting of the given layer IDs.

        """
        if not inplace:
            new_raster_data = copy.deepcopy(self)
            return new_raster_data.select_layers(layer_ids, inplace=True)

        layer_ids_close = set(self._file_register['layer_id']) - set(layer_ids)
        self.close(layer_ids=layer_ids_close, clear_ram=False)
        self._file_register = self._file_register[self._file_register['layer_id'].isin(layer_ids)]

        return self

    def select_px_window(self, row, col, height=1, width=1, inplace=False):
        """
        Selects the pixel coordinates according to the given pixel window.

        Parameters
        ----------
        row : int
            Top-left row number of the pixel window anchor.
        col : int
            Top-left column number of the pixel window anchor.
        height : int, optional
            Number of rows/height of the pixel window.
        width : int, optional
            Number of columns/width of the pixel window.
        inplace : bool, optional
            If true, the current raster mosaic object is modified.
            If false, a new raster mosaic instance will be returned (default).

        Returns
        -------
        RasterData :
            Raster mosaic object a mosaic and a mosaic geometry only consisting of the intersected tile with the
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
            self._data_geom.slice_by_rc(row, col, height=height, width=width, inplace=True, name=0)
            if self._data_geom is None:
                wrn_msg = "Pixels are outside the extent of the raster mosaic."
                warnings.warn(wrn_msg)

        if len(self._mosaic.all_tiles) == 1:
            tile_oi = self._mosaic.all_tiles[0]
            tile_oi.slice_by_rc(row, col, height=height, width=width, inplace=True, name=0)
            tile_oi.active = True
            self._mosaic.from_tile_list([tile_oi], inplace=True)

        return self

    def select_xy(self, x, y, sref=None, inplace=False):
        """
        Selects tile and pixel coordinates according to the given coordinate tuple.

        Parameters
        ----------
        x : number
            Coordinates in X direction.
        y : number
            Coordinates in Y direction.
        sref : geospade.crs.SpatialRef
            CRS of the given coordinate tuple.
        inplace : bool, optional
            If true, the current raster mosaic object is modified.
            If false, a new raster mosaic instance will be returned (default).

        Returns
        -------
        RasterData :
            Raster mosaic object with a file register and a mosaic only consisting of the intersected tile containing
            information on the location of the time series.

        """
        if not inplace:
            new_raster_data = copy.deepcopy(self)
            return new_raster_data.select_xy(x, y, sref=sref, inplace=True)

        if self._data_geom is not None:
            row, col = self._data_geom.xy2rc(x, y, sref=sref)
            self._data_geom.slice_by_rc(row, col, inplace=True, name=0)
            if self._data_geom is None:
                wrn_msg = "Coordinates are outside the spatial extent of the raster mosaic."
                warnings.warn(wrn_msg)

        tile_oi = self._mosaic.xy2tile(x, y, sref=sref)
        if tile_oi is not None:
            row, col = tile_oi.xy2rc(x, y, sref=sref)
            tile_oi.slice_by_rc(row, col, inplace=True, name=0)
            tile_oi.active = True
            self._mosaic.from_tile_list([tile_oi], inplace=True)
            self._file_register = self._file_register[self._file_register['geom_id'] == tile_oi.parent_root.name]
        else:
            wrn_msg = "Coordinates are outside the spatial extent of the raster mosaic files."
            warnings.warn(wrn_msg)
            return

        return self

    def select_bbox(self, bbox, sref=None, inplace=False):
        """
        Selects tile and pixel coordinates according to the given bounding box.

        Parameters
        ----------
        bbox : list of 2 2-tuple
            Bounding box to select, i.e. [(x_min, y_min), (x_max, y_max)]
        sref : geospade.crs.SpatialRef
            CRS of the given bounding box coordinates.
        inplace : bool, optional
            If true, the current raster mosaic object is modified.
            If false, a new raster mosaic instance will be returned (default).

        Returns
        -------
        RasterData :
            Raster mosaic object with a file register and a mosaic only consisting of the intersected tiles.

        """
        if not inplace:
            new_raster_data = copy.deepcopy(self)
            return new_raster_data.select_bbox(bbox, sref=sref, inplace=True)
        ogr_geom = any_geom2ogr_geom(bbox, sref=sref)
        return self.select_polygon(ogr_geom, apply_mask=False, inplace=inplace)

    def select_polygon(self, polygon, sref=None, apply_mask=True, inplace=False):
        """
        Selects tile and pixel coordinates according to the given polygon.

        Parameters
        ----------
        polygon : ogr.Geometry
            Polygon specifying the pixels to collect.
        sref : geospade.crs.SpatialRef
            CRS of the given bounding box coordinates.
        apply_mask : bool, optional
            True if pixels outside the polygon should be set to a no mosaic value (default).
            False if every pixel withing the bounding box of the polygon should be included.
        inplace : bool, optional
            If true, the current raster mosaic object is modified.
            If false, a new raster mosaic instance will be returned (default).

        Returns
        -------
        RasterData :
            Raster mosaic object with a file register and a mosaic only consisting of the intersected tiles.

        """
        if not inplace:
            new_raster_data = copy.deepcopy(self)
            return new_raster_data.select_polygon(polygon, sref=sref, apply_mask=apply_mask, inplace=True)

        if self._data_geom is not None:
            self._data_geom.slice_by_geom(polygon, inplace=True)
            if self._data_geom is None:
                wrn_msg = "Polygon is outside the spatial extent of the raster mosaic."
                warnings.warn(wrn_msg)

        polygon = any_geom2ogr_geom(polygon, sref=sref)
        sliced_mosaic = self._mosaic.slice_by_geom(polygon, active_only=False, apply_mask=apply_mask, inplace=False)
        if sliced_mosaic is None:
            wrn_msg = "Polygon is outside the spatial extent of the raster mosaic files."
            warnings.warn(wrn_msg)
            return

        self._mosaic = sliced_mosaic
        tile_names_oi = [tile.parent_root.name for tile in self._mosaic.tiles]
        self._file_register = self._file_register.loc[self._file_register['geom_id'].isin(tile_names_oi)]

        return self

    def _view_data(self):
        """ xr.Dataset : Returns a subset of the mosaic according to the intersected mosaic. """
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

            data = data.sel({self._file_dim: list(set(self._file_register[self._file_dim]))})

        return data

    def _add_grid_mapping(self):
        """ Adds grid mapping information the xr.Dataset. """
        if self._data is not None and self._data_geom is not None:
            self._data.rio.write_crs(self._data_geom.sref.wkt, inplace=True)
            self._data.rio.write_transform(Affine(*self._data_geom.geotrans), inplace=True)

    def close(self, layer_ids=None, clear_ram=True):
        """
        Closes open file handles and optionally mosaic in stored in RAM.

        Parameters
        ----------
        layer_ids : list, optional
            Layer IDs indicating the file handles which should be closed. Defaults to None, i.e. all file handles are
            closed.
        clear_ram : bool, optional
            If true (default), memory allocated by the internal mosaic object is released.

        """
        if layer_ids is not None:
            bool_idxs = self._file_register['layer_id'].isin(layer_ids)
            driver_ids = set(self._file_register.loc[bool_idxs, 'driver_id'])
            self._file_register.loc[bool_idxs, 'driver_id'] = None
            driver_ids = list(set(self._drivers.keys()) - driver_ids)
        else:
            self._file_register['driver_id'] = None
            driver_ids = list(self._drivers.keys())

        for driver_id in driver_ids:
            self._drivers[driver_id].close()
            del self._drivers[driver_id]

        if clear_ram:
            self._data = None
            gc.collect()

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
            Deepcopy of raster mosaic.

        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result


class RasterDataReader(RasterData):
    """ Allows to read and modify a stack of raster mosaic. """
    def __init__(self, file_register, mosaic, file_dimension='idx', file_coords=None):
        """
        Constructor of `RasterDataReader`.

        Parameters
        ----------
        file_register : pd.Dataframe
            Data frame managing a stack/list of files containing the following columns:
                - 'filepath' : str
                    Full file path to a geospatial file.
                - 'layer_id' : int
                    Specifies an ID to which layer a file belongs to.
                - 'tile_id' : str or int
                    Tile name or ID to which tile a file belongs to.
        mosaic : geospade.raster.MosaicGeometry
            Mosaic representing the spatial allocation of the given files. The tiles of the mosaic have to match the
            ID's/names of the 'tile_id' column.
        file_dimension : str, optional
            Dimension/column name of the dimension, where to stack the files along (first axis), e.g. time, bands etc.
            Defaults to 'layer_id', i.e. the layer ID's are used as the main coordinates to stack the files.
        file_coords : list, optional
            Additional columns of `file_register` to use as coordinates. Defaults to None, i.e. only coordinates along
            `file_dimension` are used.

        """
        super().__init__(file_register, mosaic, file_dimension=file_dimension, file_coords=file_coords)

    @abc.abstractmethod
    def read(self, *args, auto_decode=False, decoder=None, decoder_kwargs=None, **kwargs):
        """
        Read mosaic from disk.

        Parameters
        ----------
        auto_decode : bool, optional
            True if mosaic should be decoded according to the information available in its metadata (default).
        decoder : callable, optional
            Function allowing to decode mosaic read from disk.
        decoder_kwargs : dict, optional
            Keyword arguments for the decoder.

        """
        pass

    @abc.abstractmethod
    def _to_xarray(self, *args, **kwargs):
        """
        Converts mosaic read from disk to an xarray dataset.

        Returns
        -------
        xr.Dataset

        """
        pass

    def load(self, *args, **kwargs):
        """ Loads mosaic either from disk or RAM. """
        if self._data is not None and self._data_geom is not None:
            self._data = self.data
            self._data_geom.parent = None
        else:
            return self.read(*args, **kwargs)


class RasterDataWriter(RasterData):
    """ Allows to write and modify a stack of raster mosaic. """
    def __init__(self, file_register, mosaic, data=None, file_dimension='idx', file_coords=None):
        """
        Constructor of `RasterDataWriter`.

        Parameters
        ----------
        file_register : pd.Dataframe
            Data frame managing a stack/list of files containing the following columns:
                - 'filepath' : str
                    Full file path to a geospatial file.
                - 'layer_id' : int
                    Specifies an ID to which layer a file belongs to.
                - 'tile_id' : str or int
                    Tile name or ID to which tile a file belongs to.
        mosaic : geospade.raster.MosaicGeometry
            Mosaic representing the spatial allocation of the given files. The tiles of the mosaic have to match the
            ID's/names of the 'tile_id' column.
        data : xr.Dataset, optional
            Raster mosaic stored in memory. It must match the spatial sampling and CRS of the mosaic, but not its spatial
            extent or tiling. Moreover, the dimension of the mosaic along the first dimension (stack/file dimension), must
            match the entries/filepaths in `file_register`.
        file_dimension : str, optional
            Dimension/column name of the dimension, where to stack the files along (first axis), e.g. time, bands etc.
            Defaults to 'layer_id', i.e. the layer ID's are used as the main coordinates to stack the files.
        file_coords : list, optional
            Additional columns of `file_register` to use as coordinates. Defaults to None, i.e. only coordinates along
            `file_dimension` are used.

        """
        super().__init__(file_register, mosaic, data=data, file_dimension=file_dimension, file_coords=file_coords)

    @abc.abstractmethod
    def write(self, data, encoder=None, encoder_kwargs=None, overwrite=False, **kwargs):
        """
        Writes a certain chunk of mosaic to disk.

        Parameters
        ----------
        data : xr.Dataset
            Data chunk to be written to disk or being appended to existing mosaic.
        encoder : callable, optional
            Function allowing to encode mosaic before writing it to disk.
        encoder_kwargs : dict, optional
            Keyword arguments for the encoder.
        overwrite : bool, optional
            True if mosaic should be overwritten, false if not (default).

        """
        pass

    @abc.abstractmethod
    def export(self, apply_tiling=False, encoder=None, encoder_kwargs=None, overwrite=False, **kwargs):
        """
        Writes all the internally stored mosaic to disk.

        Parameters
        ----------
        apply_tiling : bool, optional
            True if the internal mosaic should be tiled according to the mosaic.
            False if the internal mosaic composes a new tile and should not be tiled (default).
        encoder : callable, optional
            Function allowing to encode mosaic before writing it to disk.
        encoder_kwargs : dict, optional
            Keyword arguments for the encoder.
        overwrite : bool, optional
            True if mosaic should be overwritten, false if not (default).

        """
        pass

    @classmethod
    def from_xarray(cls, data, file_register, mosaic=None):
        """
        Converts an xarray dataset and a file register to a `RasterDataWrite` instance.

        Parameters
        ----------
        data : xr.Dataset
            Raster mosaic.
        file_register : pd.Dataframe
            Data frame managing a stack/list of files containing the following columns:
                - 'filepath' : str
                    Full file path to a geospatial file.
                - 'layer_id' : int
                    Specifies an ID to which layer a file belongs to.
                - 'tile_id' : str or int
                    Tile name or ID to which tile a file belongs to.
        mosaic : geospade.raster.MosaicGeometry, optional
            Mosaic representing the spatial allocation of the given files. The tiles of the mosaic have to match the
            ID's/names of the 'tile_id' column. If it is None, a one-tile mosaic will be created from the given
            mosaic.

        Returns
        -------
        RasterDataWriter

        """
        tile = cls.raster_geom_from_data(data, name=0)
        mosaic = mosaic or MosaicGeometry([tile], check_consistency=False)
        return cls(file_register, mosaic, data=data)

    def load(self, *args, **kwargs):
        """ Loads mosaic from RAM. """
        if self._data is not None and self._data_geom is not None:
            self._data = self.data
            self._data_geom.parent = None


if __name__ == '__main__':
    pass