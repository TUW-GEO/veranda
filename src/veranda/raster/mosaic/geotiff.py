""" Raster data class managing I/O for multiple GeoTIFF files. """

import os
import uuid
import tempfile
import xarray as xr
import numpy as np
import pandas as pd
from osgeo import gdal
from datetime import datetime
from typing import Tuple, List
from multiprocessing import Pool, RawArray

from geospade.crs import SpatialRef
from geospade.raster import Tile
from geospade.raster import MosaicGeometry

from veranda.utils import to_list
from veranda.raster.native.geotiff import GeoTiffFile
from veranda.raster.native.geotiff import create_vrt_file
from veranda.raster.gdalport import GDAL_TO_NUMPY_DTYPE
from veranda.raster.mosaic.base import RasterDataReader, RasterDataWriter, RasterAccess

PROC_OBJS = {}


def read_init(fr, am, sm, sd, td, ad, dc, dk):
    """ Helper method for setting the entries of global variable `PROC_OBJS` to be available during multiprocessing. """
    PROC_OBJS['global_file_register'] = fr
    PROC_OBJS['access_map'] = am
    PROC_OBJS['shm_map'] = sm
    PROC_OBJS['stack_dimension'] = sd
    PROC_OBJS['tile_dimension'] = td
    PROC_OBJS['auto_decode'] = ad
    PROC_OBJS['decoder'] = dc
    PROC_OBJS['decoder_kwargs'] = dk


class GeoTiffAccess(RasterAccess):
    """
    Helper class to build the link between indexes of the source array (access) and the target array (assignment).
    The base class `RasterAccess` is extended with some properties needed for accessing GeoTIFF files.

    """
    def __init__(self, src_raster_geom, dst_raster_geom, src_root_raster_geom=None):
        """
        Constructor of `GeoTiffAccess`.

        Parameters
        ----------
        src_raster_geom : geospade.raster.RasterGeometry
            Raster geometry representing the extent and indices of the array to access.
        dst_raster_geom : geospade.raster.RasterGeometry
            Raster geometry representing the extent and indices of the array to assign.
        src_root_raster_geom : geospade.raster.RasterGeometry, optional
            Raster geometry representing the origin to which `src_raster_geom` should be referred to. Defaults to None,
            i.e. the root parent of `src_raster_geom` is used.

        """
        super().__init__(src_raster_geom, dst_raster_geom, src_root_raster_geom=src_root_raster_geom)
        self.src_wkt = src_raster_geom.parent_root.sref.wkt
        self.src_geotrans = src_raster_geom.parent_root.geotrans
        self.src_shape = src_raster_geom.parent_root.shape

    @property
    def gdal_args(self) -> Tuple[int, int, int, int]:
        """ Prepares the needed positional arguments for GDAL's `ReadAsArray()` function. """
        min_row, min_col, max_row, max_col = self.src_window
        n_cols, n_rows = max_col - min_col + 1, max_row - min_row + 1
        return min_col, min_row, n_cols, n_rows

    @property
    def read_args(self) -> Tuple[int, int, int, int]:
        """ Prepares the needed positional arguments for the `read()` function of the internal GeoTIFF native. """
        min_col, min_row, n_cols, n_rows = self.gdal_args
        return min_row, min_col, n_rows, n_cols


class GeoTiffReader(RasterDataReader):
    """ Allows to read and manage a stack of GeoTIFF data. """
    def __init__(self, file_register, mosaic, stack_dimension='layer_id', stack_coords=None, tile_dimension='tile_id',
                 space_dims=None, file_class=GeoTiffFile, file_class_kwargs=None):
        """
        Constructor of `GeoTiffReader`.

        Parameters
        ----------
        file_register : pd.Dataframe, optional
            Data frame managing a stack/list of files containing the following columns:
                - 'filepath' : str
                    Full file path to a geospatial file.
                - 'layer_id' : object
                    Specifies an ID to which layer a file belongs to, e.g. a layer counter or a timestamp. Must
                    correspond to `stack_dimension`.
                - 'tile_id' : str
                    Tile name or ID to which tile a file belongs to.
        mosaic : geospade.raster.MosaicGeometry
            Mosaic representing the spatial allocation of the given files. The tiles of the mosaic have to match the
            ID's/names of the 'tile_id' column.
        stack_dimension : str, optional
            Dimension/column name of the dimension, where to stack the files along (first axis), e.g. time, bands etc.
            Defaults to 'layer_id', i.e. the layer ID's are used as the main coordinates to stack the files.
        stack_coords : list, optional
            Additional columns of `file_register` to use as coordinates. Defaults to None, i.e. only coordinates along
            `stack_dimension` are used.
        tile_dimension : str, optional
            Dimension/column name of the dimension containing tile ID's in correspondence with the tiles in `mosaic`.
            Defaults to 'tile_id'.
        space_dims : list, optional
            Dictionary containing the spatial dimension names. By default it is set to ['y', 'x'].
        file_class : class, optional
            Class for opening GeoTIFF files. Defaults to `GeoTiffFile`.
        file_class_kwargs : dict, optional
            Keyword arguments for `file_class`.

        """
        super().__init__(file_register, mosaic, stack_dimension=stack_dimension, stack_coords=stack_coords,
                         tile_dimension=tile_dimension)

        file_class_kwargs = file_class_kwargs or dict()
        ref_filepath = self._file_register['filepath'].iloc[0]
        with file_class(ref_filepath, 'r', **file_class_kwargs) as gt_file:
            self._ref_dtypes = gt_file.dtypes
            self._ref_nodatavals = gt_file.nodatavals

        self._ref_space_dims = space_dims or ['y', 'x']

    @classmethod
    def from_filepaths(cls, filepaths, mosaic_class=MosaicGeometry, mosaic_kwargs=None, tile_kwargs=None,
                       stack_dimension='layer_id', tile_dimension='tile_id', **kwargs) -> "GeoTiffReader":
        """
        Creates a `GeoTiffReader` instance as one stack of GeoTIFF files.

        Parameters
        ----------
        filepaths : list of str
            List of full system paths to a GeoTIFF file.
        mosaic_class : geospade.raster.MosaicGeometry, optional
            Mosaic class used to manage the spatial properties of the file stack. If None, the most generic mosaic will
            be used by default. The initialised mosaic will only contain one tile.
        mosaic_kwargs : dict, optional
            Additional arguments for initialising `mosaic_class`.
        tile_kwargs : dict, optional
            Additional arguments for initialising a tile class associated with `mosaic_class`.
        stack_dimension : str, optional
            Dimension/column name of the dimension, where to stack the files along (first axis), e.g. time, bands etc.
            Defaults to 'layer_id', i.e. the layer ID's are used as the main coordinates to stack the files.
        tile_dimension : str, optional
            Dimension/column name of the dimension containing tile ID's in correspondence with the tiles in `mosaic`.
            Defaults to 'tile_id'.
        kwargs : dict, optional
            Key-word arguments for the `GeoTiffReader` constructor.

        Returns
        -------
        GeoTiffReader

        """
        mosaic_kwargs = mosaic_kwargs or dict()
        tile_kwargs = tile_kwargs or dict()

        n_filepaths = len(filepaths)
        file_register_dict = dict()
        file_register_dict['filepath'] = filepaths
        file_register_dict[tile_dimension] = ['0'] * n_filepaths
        file_register_dict[stack_dimension] = list(range(1, n_filepaths + 1))
        file_register = pd.DataFrame(file_register_dict)

        ref_filepath = filepaths[0]
        with GeoTiffFile(ref_filepath, 'r') as gt_file:
            sref_wkt = gt_file.sref_wkt
            geotrans = gt_file.geotrans
            n_rows, n_cols = gt_file.raster_shape

        tile_class = mosaic_class.get_tile_class()
        tile = tile_class(n_rows, n_cols, sref=SpatialRef(sref_wkt), geotrans=geotrans, name='0', **tile_kwargs)
        mosaic_geom = mosaic_class.from_tile_list([tile], check_consistency=False, **mosaic_kwargs)

        return cls(file_register, mosaic_geom, stack_dimension=stack_dimension, tile_dimension=tile_dimension,
                   **kwargs)

    @classmethod
    def from_mosaic_filepaths(cls, filepaths, mosaic_class=MosaicGeometry, mosaic_kwargs=None,
                              stack_dimension='layer_id', tile_dimension='tile_id', **kwargs) -> "GeoTiffReader":
        """
        Creates a `GeoTiffDataReader` instance as multiple stacks of GeoTIFF files.

        Parameters
        ----------
        filepaths : list of str
            List of full system paths to a GeoTIFF file.
        mosaic_class : geospade.raster.MosaicGeometry, optional
            Mosaic class used to manage the spatial properties of the file stacks. If None, the most generic mosaic will
            be used by default.
        mosaic_kwargs : dict, optional
            Additional arguments for initialising `mosaic_class`.
        stack_dimension : str, optional
            Dimension/column name of the dimension, where to stack the files along (first axis), e.g. time, bands etc.
            Defaults to 'layer_id', i.e. the layer ID's are used as the main coordinates to stack the files.
        tile_dimension : str, optional
            Dimension/column name of the dimension containing tile ID's in correspondence with the tiles in `mosaic`.
            Defaults to 'tile_id'.
        kwargs : dict, optional
            Key-word arguments for the `GeoTiffReader` constructor.

        Returns
        -------
        GeoTiffReader

        """
        mosaic_kwargs = mosaic_kwargs or dict()
        file_register_dict = dict()
        file_register_dict['filepath'] = filepaths
        tile_class = mosaic_class.get_tile_class()
        tiles, tile_ids, layer_ids = RasterDataReader._create_tile_and_layer_info_from_files(filepaths, tile_class,
                                                                                             GeoTiffFile)

        file_register_dict[tile_dimension] = tile_ids
        file_register_dict[stack_dimension] = layer_ids
        file_register = pd.DataFrame(file_register_dict)

        mosaic_geom = mosaic_class.from_tile_list(tiles, check_consistency=False, **mosaic_kwargs)

        return cls(file_register, mosaic_geom, stack_dimension=stack_dimension, tile_dimension=tile_dimension,
                   **kwargs)

    def apply_nan(self):
        """ Converts no data values given as an attribute '_FillValue' or keyword `nodatavals` to np.nan. """
        nodatavals = {dvar: self._ref_nodatavals[i] for i, dvar in enumerate(self.data_view.data_vars)}
        super().apply_nan(nodatavals=nodatavals)

    def read(self, bands=1, band_names=None, engine='vrt', n_cores=1,
             auto_decode=False, decoder=None, decoder_kwargs=None) -> "GeoTiffReader":
        """
        Reads data from disk.

        Parameters
        ----------
        bands : tuple of int or int, optional
            The GeoTIFF bands of interest. Defaults to the first band, i.e. 1.
        band_names : tuple of str or str, optional
            Names associated with the respective bands of the GeoTIFF files. Defaults to None, i.e. the band numbers
            will be used as a name.
        engine : str, optional
            Engine used in the background to read data. The following options are available:
                - 'vrt' : Uses GDAL's VRT format to stack the GeoTIFF files per tile and load them as once. This option
                          yields good performance if the mosaic is stored locally on one drive. Parallelisation is applied
                          across tiles.
                - 'parallel' : Reads file by file, but in a parallelised manner. This option yields good performance if
                               the mosaic is stored on a distributed file system.
        n_cores : int, optional
            Number of cores used to read data in a parallelised manner (defaults to 1).
        auto_decode : bool, optional
            True if data should be decoded according to the information available in its metadata. Defaults to False.
        decoder : callable, optional
            Function allowing to decode data read from disk.
        decoder_kwargs : dict, optional
            Keyword arguments for the decoder.

        """
        bands = to_list(bands)
        band_names = to_list(band_names)
        dst_tile = Tile.from_extent(self._mosaic.outer_extent, sref=self._mosaic.sref,
                                    x_pixel_size=self._mosaic.x_pixel_size,
                                    y_pixel_size=self._mosaic.y_pixel_size,
                                    name='0')

        shm_map = {band: self.__init_band_data(band, dst_tile) for band in bands}
        access_map = {src_tile.parent_root.name: GeoTiffAccess(src_tile, dst_tile) for src_tile in self._mosaic.tiles}
        data_mask = self.__create_data_mask_from(access_map, dst_tile.shape)

        if engine == 'vrt':
            self.__read_vrt_stack(access_map, shm_map, n_cores=n_cores, auto_decode=auto_decode, decoder=decoder,
                                  decoder_kwargs=decoder_kwargs)
        elif engine == 'parallel':
            self.__read_parallel(access_map, shm_map, n_cores=n_cores, auto_decode=auto_decode, decoder=decoder,
                                 decoder_kwargs=decoder_kwargs)
        else:
            err_msg = f"Engine '{engine}' is not supported!"
            raise ValueError(err_msg)

        data = {band: self.__load_band_data(band, shm_map, data_mask) for band in bands}

        self._data_geom = dst_tile
        self._data = self._to_xarray(data, band_names)
        self._add_grid_mapping()
        return self

    def __init_band_data(self, band, tile) -> Tuple[RawArray, tuple]:
        """
        Initialises shared memory array for a specific band.

        Parameters
        ----------
        band : int
            Band number.
        tile : Tile
            Tile containing information about the raster shape of the band data.

        Returns
        -------
        shm_ar : RawArray
            Shared memory array.
        shm_ar_shape : tuple
            Shape of the array.

        """
        np_dtype = np.dtype(self._ref_dtypes[band - 1])
        self._ref_nodatavals[band - 1] = np.array((self._ref_nodatavals[band - 1])).astype(np_dtype)
        data_nshm = np.ones((self.n_layers, tile.n_rows, tile.n_cols), dtype=np_dtype) * \
                    self._ref_nodatavals[band - 1]
        shm_ar_shape = data_nshm.shape
        c_dtype = np.ctypeslib.as_ctypes_type(data_nshm.dtype)
        shm_rar = RawArray(c_dtype, data_nshm.size)
        shm_data = np.frombuffer(shm_rar, dtype=np_dtype).reshape(shm_ar_shape)
        shm_data[:] = data_nshm[:]

        return shm_rar, shm_ar_shape

    def __load_band_data(self, band, shm_map, mask) -> np.ndarray:
        """
        Loads band data from the corresponding shared memory array.

        Parameters
        ----------
        band : int
            Band number.
        shm_map : dict
            Dictionary mapping the band number with a tuple containing a shared memory array and its shape.
        mask : np.array
            Data mask.

        Returns
        -------
        shm_ar : RawArray
            Shared memory array.

        """
        shm_rar, shm_ar_shape = shm_map[band]
        shm_data = np.frombuffer(shm_rar, dtype=self._ref_dtypes[band - 1]).reshape(shm_ar_shape)
        shm_data[:, ~mask.astype(bool)] = self._ref_nodatavals[band - 1]

        return shm_data

    def __create_data_mask_from(self, access_map, shape) -> np.ndarray:
        """
        Creates data mask from all individual tile masks contributing to the spatial extent of the data.

        Parameters
        ----------
        access_map : dict
            Dictionary mapping tile names with `GeoTiffAccess` instances for storing the access relation between
            the data mask and the tiles.
        shape : tuple
            Shape of the data mask.

        Returns
        -------
        data_mask : np.ndarray
            Data mask (1 valid data, 0 no data).

        """
        data_mask = np.ones(shape)
        for tile in self._mosaic.tiles:
            gt_access = access_map[tile.parent_root.name]
            data_mask[gt_access.dst_row_slice, gt_access.dst_col_slice] = tile.mask
        return data_mask

    def __read_vrt_stack(self, access_map, shm_map, n_cores=1,
                         auto_decode=False, decoder=None, decoder_kwargs=None):
        """
        Reads GeoTIFF data from a stack of GeoTIFF files by using GDAL's VRT format.

        Parameters
        ----------
        access_map : dict
            Dictionary mapping tile/geometry ID's with `GeoTiffAccess` instances to define the access patterns between
            the data to load and to assign.
        shm_map : dict
            Dictionary mapping band numbers with the respective name of the memory buffer of the shared numpy raw array.
        n_cores : int, optional
            Number of cores used to read data in a parallelised manner (defaults to 1).
        auto_decode : bool, optional
            True if data should be decoded according to the information available in its metadata. Defaults to False.
        decoder : callable, optional
            Function allowing to decode data read from disk.
        decoder_kwargs : dict, optional
            Keyword arguments for the decoder.

        """
        decoder_kwargs = decoder_kwargs or dict()
        global_file_register = self._file_register

        with Pool(n_cores, initializer=read_init, initargs=(global_file_register, access_map, shm_map,
                                                            self._file_dim, self._tile_dim,
                                                            auto_decode, decoder, decoder_kwargs)) as p:
            p.map(read_vrt_stack, access_map.keys())

    def __read_parallel(self, access_map, shm_map, n_cores=1,
                        auto_decode=False, decoder=None, decoder_kwargs=None):
        """
        Reads GeoTIFF mosaic on a file-by-file basis, in a parallel manner.

        Parameters
        ----------
        access_map : dict
            Dictionary mapping tile/geometry ID's with `GeoTiffAccess` instances to define the access patterns between
            the mosaic to load and to assign.
        shm_map : dict
            Dictionary mapping band numbers with the respective name of the memory buffer of the shared numpy raw array.
        n_cores : int, optional
            Number of cores used to read data in a parallelised manner (defaults to 1).
        auto_decode : bool, optional
            True if data should be decoded according to the information available in its metadata. Defaults to False.
        decoder : callable, optional
            Function allowing to decode data read from disk.
        decoder_kwargs : dict, optional
            Keyword arguments for the decoder.

        """
        decoder_kwargs = decoder_kwargs or None
        global_file_register = self._file_register

        with Pool(n_cores, initializer=read_init, initargs=(global_file_register, access_map, shm_map,
                                                            self._file_dim, self._tile_dim,
                                                            auto_decode, decoder, decoder_kwargs)) as p:
            p.map(read_single_files, global_file_register.index)

    def _to_xarray(self, data, band_names=None) -> xr.Dataset:
        """
        Converts data being available as a NumPy array to an xarray dataset.

        Parameters
        ----------
        data : dict
            Dictionary mapping band numbers to GeoTIFF raster data being available as a NumPy array.
        band_names : list of str, optional
            Band names associated with the respective band number.

        Returns
        -------
        xrds : xr.Dataset

        """
        dims = [self._file_dim] + self._ref_space_dims

        # retrieve file register with unique stack dimension
        ref_tile_id = list(self.file_register[self._tile_dim])[0]
        file_register_uni = self._file_register.loc[self._file_register[self._tile_dim] == ref_tile_id]

        coord_dict = dict()
        for coord in self._file_coords:
            coord_dict[coord] = file_register_uni[coord]

        coord_dict[self._ref_space_dims[0]] = self._data_geom.y_coords
        coord_dict[self._ref_space_dims[1]] = self._data_geom.x_coords

        xar_dict = dict()
        bands = list(data.keys())
        band_names = band_names or bands
        for i, band in enumerate(bands):
            xar_dict[band_names[i]] = xr.DataArray(data[band], coords=coord_dict, dims=dims,
                                                   attrs={'_FillValue': self._ref_nodatavals[band - 1]})

        xrds = xr.Dataset(data_vars=xar_dict)

        return xrds


class GeoTiffWriter(RasterDataWriter):
    """ Allows to write and manage a stack of GeoTIFF files. """
    def __init__(self, mosaic, file_register=None, data=None, stack_dimension='layer_id', stack_coords=None,
                 tile_dimension='tile_id', dirpath=None, fn_pattern='{layer_id}.tif', fn_formatter=None):
        """
        Constructor of `GeoTiffWriter`.

        Parameters
        ----------
        mosaic : geospade.raster.MosaicGeometry
            Mosaic representing the spatial allocation of the given files. The tiles of the mosaic have to match the
            ID's/names of the 'tile_id' column.
        file_register : pd.Dataframe, optional
            Data frame managing a stack/list of files containing the following columns:
                - 'filepath' : str
                    Full file path to a geospatial file.
                - 'layer_id' : object
                    Specifies an ID to which layer a file belongs to, e.g. a layer counter or a timestamp. Must
                    correspond to `stack_dimension`.
                - 'tile_id' : str
                    Tile name or ID to which tile a file belongs to.
            If it is None, then the constructor tries to create a file from other keyword arguments, i.e. `data`,
            `dirpath`, `fn_pattern`, and `fn_formatter`.
        data : xr.Dataset, optional
            Raster data stored in memory. It must match the spatial sampling and CRS of the mosaic, but not its spatial
            extent or tiling. Moreover, the dimension of the mosaic along the first dimension (stack dimension), must
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

        super().__init__(mosaic, file_register=file_register, data=data, stack_dimension=stack_dimension,
                         stack_coords=stack_coords, tile_dimension=tile_dimension, dirpath=dirpath,
                         fn_pattern=fn_pattern, fn_formatter=fn_formatter)

    def __get_encoding_info_from_data(self, data, band_names) -> Tuple[dict, dict, dict, dict]:
        """
        Extracts encoding information from an xarray dataset.

        Parameters
        ----------
        data : xr.Dataset
            Data to extract encoding info from.
        band_names : list of str
            List of band names.

        Returns
        -------
        nodatavals : dict
            Band number mapped to no data value (defaults to 0).
        scale_factors : dict
            Band number mapped to scale factor (defaults to 1).
        offsets : dict
            Band number mapped to offset (defaults to 0).
        dtypes : dict
            Band number mapped to data type.

        """
        nodatavals = dict()
        scale_factors = dict()
        offsets = dict()
        dtypes = dict()
        for i, band_name in enumerate(band_names):
            dtypes[i + 1] = data[band_name].data.dtype.name
            nodatavals[i + 1] = data[band_name].attrs.get('_FillValue', 0)
            scale_factors[i + 1] = data[band_name].attrs.get('scale_factor', 1)
            offsets[i + 1] = data[band_name].attrs.get('add_offset', 0)

        return nodatavals, scale_factors, offsets, dtypes

    def write(self, data, use_mosaic=False, data_variables=None, encoder=None, encoder_kwargs=None, overwrite=False,
              **kwargs):
        """
        Writes a certain chunk of data to disk.

        Parameters
        ----------
        data : xr.Dataset
            Data chunk to be written to disk or being appended to existing data.
        use_mosaic : bool, optional
            True if data should be written according to the mosaic.
            False if data composes a new tile and should not be tiled (default).
        data_variables : list of str, optional
            Data variables to write. Defaults to None, i.e. all data variables are written.
        encoder : callable, optional
            Function allowing to encode data before writing it to disk.
        encoder_kwargs : dict, optional
            Keyword arguments for the encoder.
        overwrite : bool, optional
            True if data should be overwritten, False if not (default).

        """
        data_geom = self.raster_geom_from_data(data, sref=self.mosaic.sref)
        data_filt = data if data_variables is None else data[data_variables]
        band_names = list(data_filt.data_vars)
        n_bands = len(band_names)
        nodatavals, scale_factors, offsets, dtypes = self.__get_encoding_info_from_data(data_filt, band_names)

        for filepath, file_group in self._file_register.groupby('filepath'):
            tile_id = file_group.iloc[0].get(self._tile_dim, '0')

            file_coords = list(file_group[self._file_dim])
            xrds = data_filt.sel(**{self._file_dim: file_coords})
            data_write = xrds[band_names].to_array().data

            if use_mosaic:
                src_tile = self._mosaic[tile_id]
                if not src_tile.intersects(data_geom):
                    continue
                dst_tile = data_geom.slice_by_geom(src_tile, inplace=False, name='0')
                gt_access = GeoTiffAccess(dst_tile, src_tile, src_root_raster_geom=data_geom)
                data_write = data_write[..., gt_access.src_row_slice, gt_access.src_col_slice]
            else:
                src_tile = data_geom
                dst_tile = data_geom
                gt_access = GeoTiffAccess(dst_tile, src_tile, src_root_raster_geom=data_geom)

            file_id = file_group.iloc[0].get('file_id', None)
            if file_id is None:
                gt_file = GeoTiffFile(filepath, mode='w', geotrans=src_tile.geotrans, sref_wkt=src_tile.sref.wkt,
                                      raster_shape=src_tile.shape, n_bands=n_bands, dtypes=dtypes,
                                      scale_factors=scale_factors, offsets=offsets, nodatavals=nodatavals)
                file_id = len(list(self._files.keys())) + 1
                self._files[file_id] = gt_file
                self._file_register.loc[file_group.index, 'file_id'] = file_id

            gt_file = self._files[file_id]
            gt_file.write(data_write[..., gt_access.src_row_slice,
                                     gt_access.src_col_slice].reshape((-1, dst_tile.n_rows, dst_tile.n_cols)),
                          row=gt_access.dst_window[0], col=gt_access.dst_window[1],
                          encoder=encoder, encoder_kwargs=encoder_kwargs)

    def export(self, use_mosaic=False, data_variables=None, encoder=None, encoder_kwargs=None, overwrite=False,
               **kwargs):
        """
        Writes all internally stored data to disk.

        Parameters
        ----------
        use_mosaic : bool, optional
            True if data should be written according to the mosaic.
            False if data composes a new tile and should not be tiled (default).
        data_variables : list of str, optional
            Data variables to write. Defaults to None, i.e. all data variables are written.
        encoder : callable, optional
            Function allowing to encode data before writing it to disk.
        encoder_kwargs : dict, optional
            Keyword arguments for the encoder.
        overwrite : bool, optional
            True if data should be overwritten, False if not (default).

        """

        self.write(self.data_view, use_mosaic, data_variables, encoder, encoder_kwargs, overwrite, **kwargs)


def read_vrt_stack(tile_id):
    """
    Function being responsible to create a new VRT file from a stack of GeoTIFF files, read data from this files, and
    assign it to a shared memory array. This function is meant to be executed in parallel on different cores.

    Parameters
    ----------
    tile_id : str
        Tile/geometry ID coming from the Pool's mapping function.

    """
    global_file_register = PROC_OBJS['global_file_register']
    access_map = PROC_OBJS['access_map']
    shm_map = PROC_OBJS['shm_map']
    tile_dimension = PROC_OBJS['tile_dimension']

    gt_access = access_map[tile_id]
    bands = list(shm_map.keys())
    file_register = global_file_register.loc[global_file_register[tile_dimension] == tile_id]

    path = tempfile.gettempdir()
    vrt_filepath = os.path.join(path, f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex}.vrt")
    while os.path.exists(vrt_filepath):
        vrt_filepath = os.path.join(path, f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex}.vrt")
    filepaths = list(file_register['filepath'])
    create_vrt_file(filepaths, vrt_filepath, gt_access.src_shape, gt_access.src_wkt, gt_access.src_geotrans,
                    bands=bands)

    src = gdal.Open(vrt_filepath, gdal.GA_ReadOnly)
    vrt_data = src.ReadAsArray(*gt_access.gdal_args)
    for band in bands:
        _assign_vrt_stack_per_band(tile_id, band, src, vrt_data)


def _assign_vrt_stack_per_band(tile_id, band, src, vrt_data):
    """
    Assigns loaded raster data to shared memory array for a specific band.

    Parameters
    ----------
    tile_id : str
        Tile/geometry ID coming from the Pool's mapping function.
    band : int
        Band number.
    src : gdal.Dataset
        GDAL dataset handle.
    vrt_data : np.ndarray
        In-memory data read from disk.

    """
    auto_decode = PROC_OBJS['auto_decode']
    decoder = PROC_OBJS['decoder']
    decoder_kwargs = PROC_OBJS['decoder_kwargs']
    access_map = PROC_OBJS['access_map']
    shm_map = PROC_OBJS['shm_map']

    gt_access = access_map[tile_id]
    bands = list(shm_map.keys())
    n_bands = len(bands)
    b_idx = bands.index(band)

    band_data = vrt_data[b_idx::n_bands, ...]
    scale_factor = src.GetRasterBand(b_idx + 1).GetScale()
    nodataval = src.GetRasterBand(b_idx + 1).GetNoDataValue()
    offset = src.GetRasterBand(b_idx + 1).GetOffset()
    dtype = GDAL_TO_NUMPY_DTYPE[src.GetRasterBand(b_idx + 1).DataType]
    if auto_decode:
        band_data = band_data.astype(float)
        band_data[band_data == nodataval] = np.nan
        band_data = band_data * scale_factor + offset
    else:
        if decoder is not None:
            band_data = decoder(band_data, nodataval=nodataval, band=band, scale_factor=scale_factor,
                                offset=offset,
                                dtype=dtype, **decoder_kwargs)

    shm_rar, shm_ar_shape = shm_map[band]
    shm_data = np.frombuffer(shm_rar, dtype=dtype).reshape(shm_ar_shape)
    shm_data[:, gt_access.dst_row_slice, gt_access.dst_col_slice] = band_data


def read_single_files(file_idx):
    """
    Function being responsible to read data from a single GeoTIFF file and assign it to a shared memory array.
    This function is meant to be executed in parallel on different cores.

    Parameters
    ----------
    file_idx : any
        Index value to access a specific row of the file register. The actual value should come from the Pool's
        mapping function.

    """
    global_file_register = PROC_OBJS['global_file_register']
    access_map = PROC_OBJS['access_map']
    shm_map = PROC_OBJS['shm_map']
    auto_decode = PROC_OBJS['auto_decode']
    decoder = PROC_OBJS['decoder']
    decoder_kwargs = PROC_OBJS['decoder_kwargs']
    stack_dimension = PROC_OBJS['stack_dimension']
    tile_dimension = PROC_OBJS['tile_dimension']

    file_entry = global_file_register.loc[file_idx]
    layer_id = file_entry[stack_dimension]
    tile_id = file_entry[tile_dimension]
    filepath = file_entry['filepath']
    gt_access = access_map[tile_id]
    bands = list(shm_map.keys())

    with GeoTiffFile(filepath, mode='r', auto_decode=auto_decode) as gt_file:
        gt_data = gt_file.read(*gt_access.read_args, bands=bands, decoder=decoder, decoder_kwargs=decoder_kwargs)
        for band in bands:
            dtype = gt_file.dtypes[band]
            shm_rar, shm_ar_shape = shm_map[band]
            shm_data = np.frombuffer(shm_rar, dtype=dtype).reshape(shm_ar_shape)
            shm_data[layer_id, gt_access.dst_row_slice, gt_access.dst_col_slice] = gt_data[(band - 1), ...]


if __name__ == '__main__':
    pass
