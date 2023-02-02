""" Raster data class managing I/O for multiple NetCDF files. """

import xarray as xr
import numpy as np
import pandas as pd
import netCDF4
from netCDF4 import MFDataset
from typing import Tuple

from geospade.crs import SpatialRef
from geospade.raster import Tile
from geospade.raster import MosaicGeometry
from geospade import DECIMALS

from veranda.utils import to_list
from veranda.raster.native.netcdf import NetCdf4File
from veranda.raster.mosaic.base import RasterDataReader, RasterDataWriter, RasterAccess


class NetCdfReader(RasterDataReader):
    """ Allows to read and manage a stack of NetCDF files. """
    def __init__(self, file_register, mosaic, stack_dimension='layer_id', stack_coords=None, tile_dimension='tile_id',
                 file_class=NetCdf4File, file_class_kwargs=None):
        """
        Constructor of `NetCdfReader`.

        Parameters
        ----------
        file_register : pd.Dataframe
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
        file_class : class, optional
            Class to use for opening files. Defaults to `NetCdf4File`.
        file_class_kwargs : dict, optional
            Keyword arguments for calling `file_class`. Defaults to none.

        """
        super().__init__(file_register, mosaic, stack_dimension=stack_dimension, stack_coords=stack_coords,
                         tile_dimension=tile_dimension)

        file_class_kwargs = file_class_kwargs or dict()
        ref_filepath = self._file_register['filepath'].iloc[0]
        with file_class(ref_filepath, 'r', **file_class_kwargs) as nc_file:
            self._ref_data_variables = nc_file.data_variables
            self._ref_nodatavals = nc_file.nodatavals
            self._ref_scale_factors = nc_file.scale_factors
            self._ref_offsets = nc_file.offsets
            self._ref_dtypes = nc_file.dtypes
            self._ref_metadata = nc_file.metadata
            space_dims = nc_file.space_dims
            stack_dims = nc_file.stack_dims
            self._ref_space_dims = list(space_dims.keys()) if isinstance(space_dims, dict) else space_dims
            self._ref_stack_dims = list(stack_dims.keys()) if isinstance(stack_dims, dict) else stack_dims

    @classmethod
    def from_filepaths(cls, filepaths, mosaic_class=MosaicGeometry, mosaic_kwargs=None, tile_kwargs=None,
                       stack_dimension='layer_id', tile_dimension='tile_id',
                       file_class=NetCdf4File, file_class_kwargs=None,
                       **kwargs) -> "NetCdfReader":
        """
        Creates a `NetCdfReader` instance as one stack of NetCDF files.

        Parameters
        ----------
        filepaths : list of str
            List of full system paths to a NetCDF file.
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
        file_class : class, optional
            Class to use for opening files. Defaults to `NetCdf4File`.
        file_class_kwargs : dict, optional
            Keyword arguments for calling `file_class`. Defaults to none.
        kwargs : dict, optional
            Key-word arguments for the `NetCdfReader` constructor.

        Returns
        -------
        NetCdfReader

        """
        mosaic_kwargs = mosaic_kwargs or dict()
        tile_kwargs = tile_kwargs or dict()
        file_class_kwargs = file_class_kwargs or dict()

        n_filepaths = len(filepaths)
        file_register_dict = dict()
        file_register_dict['filepath'] = filepaths
        file_register_dict[tile_dimension] = ['0'] * n_filepaths
        file_register_dict[stack_dimension] = list(range(n_filepaths))
        file_register = pd.DataFrame(file_register_dict)

        ref_filepath = filepaths[0]
        with file_class(ref_filepath, 'r', **file_class_kwargs) as nc_file:
            sref_wkt = nc_file.sref_wkt
            geotrans = nc_file.geotrans
            n_rows, n_cols = nc_file.raster_shape

        tile_class = mosaic_class.get_tile_class()
        tile = tile_class(n_rows, n_cols, sref=SpatialRef(sref_wkt), geotrans=geotrans, name='0', **tile_kwargs)
        mosaic_geom = mosaic_class.from_tile_list([tile], check_consistency=False, **mosaic_kwargs)

        return cls(file_register, mosaic_geom, stack_dimension=stack_dimension, tile_dimension=tile_dimension,
                   **kwargs)

    @classmethod
    def from_mosaic_filepaths(cls, filepaths, mosaic_class=MosaicGeometry, mosaic_kwargs=None,
                              stack_dimension='layer_id', tile_dimension='tile_id',
                              file_class=NetCdf4File, file_class_kwargs=None, **kwargs) -> "NetCdfReader":
        """
        Creates a `NetCdfReader` instance as multiple stacks of NetCDF files.

        Parameters
        ----------
        filepaths : list of str
            List of full system paths to a NetCDF file.
        mosaic_class : geospade.raster.MosaicGeometry, optional
            Mosaic class used to manage the spatial properties of the file stacks. If None, the most generic mosaic
            will be used by default.
        mosaic_kwargs : dict, optional
            Additional arguments for initialising `mosaic_class`.
        stack_dimension : str, optional
            Dimension/column name of the dimension, where to stack the files along (first axis), e.g. time, bands etc.
            Defaults to 'layer_id', i.e. the layer ID's are used as the main coordinates to stack the files.
        tile_dimension : str, optional
            Dimension/column name of the dimension containing tile ID's in correspondence with the tiles in `mosaic`.
            Defaults to 'tile_id'.
        file_class : class, optional
            Class to use for opening files. Defaults to `NetCdf4File`.
        file_class_kwargs : dict, optional
            Keyword arguments for calling `file_class`. Defaults to none.
        kwargs : dict, optional
            Key-word arguments for the `NetCdfReader` constructor.

        Returns
        -------
        NetCdfReader

        """
        mosaic_kwargs = mosaic_kwargs or dict()
        file_class_kwargs = file_class_kwargs or dict()
        file_register_dict = dict()
        file_register_dict['filepath'] = filepaths
        tile_class = mosaic_class.get_tile_class()
        tiles, tile_ids, layer_ids = RasterDataReader._create_tile_and_layer_info_from_files(filepaths, tile_class,
                                                                                             file_class,
                                                                                             file_class_kwargs)

        file_register_dict[tile_dimension] = tile_ids
        file_register_dict[stack_dimension] = layer_ids
        file_register = pd.DataFrame(file_register_dict)

        mosaic_geom = mosaic_class.from_tile_list(tiles, check_consistency=False, **mosaic_kwargs)

        return cls(file_register, mosaic_geom, stack_dimension=stack_dimension, tile_dimension=tile_dimension,
                   **kwargs)

    def apply_nan(self):
        """ Converts no data values given as an attribute '_FillValue' or keyword `nodatavals` to np.nan. """
        super().apply_nan(nodatavals=self._ref_nodatavals)

    def read(self, data_variables=None, engine='netcdf4', agg_dim='time', parallel=True, compute=True, auto_decode=False,
             decoder=None, decoder_kwargs=None, **kwargs) -> "NetCdfReader":
        """
        Reads NetCdf data from disk and assigns it to the class.

        Parameters
        ----------
        data_variables : list, optional
            Data variables to read. Default is to read all available data variables.
        engine : str, optional
            Engine used in the background to read NetCDF data. The following options are available:
                - 'netcdf4' : Uses the netCDF4 library to create an `MFDataset` object.
                - 'xarray' : Uses xarray's `open_mfdataset` function.
        agg_dim : str, optional
            Dimension to aggregate on (defaults to 'layer_id').
        parallel : bool, optional
            Flag to activate parallelisation or not when using 'xarray' as an engine. Defaults to True.
        compute : bool, optional
            True if values from a dask array should be loaded into RAM (default).
        auto_decode : bool, optional
            True if NetCDF data should be decoded according to the information available in its metadata. Defaults to
            False.
        decoder : callable, optional
            Function allowing to decode NetCDF data read from disk.
        decoder_kwargs : dict, optional
            Keyword arguments for the decoder.

        """
        data_variables = to_list(data_variables)
        dst_tile = Tile.from_extent(self._mosaic.outer_extent, sref=self._mosaic.sref,
                                    x_pixel_size=self._mosaic.x_pixel_size,
                                    y_pixel_size=self._mosaic.y_pixel_size,
                                    name='0')
        if engine == 'netcdf4':
            data = self.__read_netcdf4(dst_tile, data_variables=data_variables, agg_dim=agg_dim,
                                       auto_decode=auto_decode, decoder=decoder, decoder_kwargs=decoder_kwargs,
                                       **kwargs)
        elif engine == 'xarray':
            data = self.__read_xarray(dst_tile, data_variables=data_variables, parallel=parallel,
                                      agg_dim=agg_dim, compute=compute, auto_decode=auto_decode, decoder=decoder,
                                      decoder_kwargs=decoder_kwargs, **kwargs)
        else:
            err_msg = f"Engine '{engine}' is not supported!"
            raise ValueError(err_msg)

        self._data_geom = dst_tile
        self._data = data
        self._add_grid_mapping()
        return self

    def __load_data_per_data_variable(self, ds, data_variable, raster_access,
                                      decoder=None, decoder_kwargs=None) -> xr.DataArray:
        """
        Selects and slices an xarray data array from an xarray dataset.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset to subset.
        data_variable : str
            Data variable to select.
        raster_access : RasterAccess
            Helper instance to slice the data array.
        decoder : callable, optional
            Function allowing to decode NetCDF data read from disk.
        decoder_kwargs : dict, optional
            Keyword arguments for the decoder.

        Returns
        -------
        dar : xr.DataArray
            Decoded data subset.

        """
        dar = ds[data_variable][..., raster_access.src_row_slice, raster_access.src_col_slice]
        if decoder:
            dar = decoder(dar, nodataval=self._ref_nodatavals[data_variable],
                          data_variable=data_variable,
                          scale_factor=self._ref_scale_factors[data_variable],
                          offset=self._ref_offsets[data_variable],
                          dtype=self._ref_dtypes[data_variable],
                          **decoder_kwargs)
        return dar

    def __read_netcdf4(self, dst_tile, data_variables=None, agg_dim='layer_id', auto_decode=False,
                       decoder=None, decoder_kwargs=None, **kwargs) -> xr.Dataset:
        """
        Reads NetCDF data using the `MFDataset` class of the netCDF4 library.

        Parameters
        ----------
        dst_tile : geospade.raster.Tile
            Target tile representing the spatial extent of the data window to read from.
        data_variables : list, optional
            Data variables to read. Default is to read all available data variables.
        agg_dim : str, optional
            Dimension to aggregate on (defaults to 'layer_id').
        auto_decode : bool, optional
            True if NetCDF data should be decoded according to the information available in its metadata. Defaults to
            False.
        decoder : callable, optional
            Function allowing to decode NetCDF data read from disk.
        decoder_kwargs : dict, optional
            Keyword arguments for the decoder.

        Returns
        -------
        xr.Dataset :
            Read NetCDF variables represented as an xarray.Dataset instance.

        """
        data_variables = data_variables or self._ref_data_variables
        decoder_kwargs = decoder_kwargs or dict()
        data = [self.__load_data_per_tile_netcdf4(src_tile, dst_tile, data_variables, agg_dim, auto_decode, decoder,
                                                  decoder_kwargs)
                for src_tile in self._mosaic.tiles]

        return xr.combine_by_coords(data)

    def __post_proc_data_netcdf4(self, ar, dw_tile, dm_tile, nodataval=0) -> xr.DataArray:
        """
        Masks the given array.

        Parameters
        ----------
        ar : np.ndarray or xr.DataArray
            Array to mask.
        dw_tile : Tile
            Tile representing the data window to read from.
        dm_tile : Tile
            Tile to extract a data mask from.
        nodataval : float, optional
            No data value being assigned where the mask values evaluate to false (defaults to 0).

        Returns
        -------
        ar : np.ndarray or xr.DataArray
            Masked array.

        """
        if dm_tile.mask is not None:
            raster_access = RasterAccess(dm_tile, dw_tile, src_root_raster_geom=dw_tile)
            tile_mask = np.zeros(dw_tile.shape)
            tile_mask[raster_access.src_row_slice, raster_access.src_col_slice] = dm_tile.mask
            ar[:, ~tile_mask.astype(bool)] = nodataval

        return ar

    def __load_data_per_data_variable_netcdf4(self, ds, data_variable, dw_tile, dm_tile, raster_access,
                                              decoder=None, decoder_kwargs=None) -> xr.DataArray:
        """
        Selects, mask and slices a netCDF4 data variable.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset to subset.
        data_variable : str
            Data variable to select.
        dw_tile : Tile
            Tile representing the data window to read from.
        dm_tile : Tile
            Tile to extract a data mask from.
        raster_access : RasterAccess
            Helper instance to slice the data array.
        decoder : callable, optional
            Function allowing to decode NetCDF data read from disk.
        decoder_kwargs : dict, optional
            Keyword arguments for the decoder.

        Returns
        -------
        dar : xr.DataArray
            Decoded data subset.

        """
        dar = self.__load_data_per_data_variable(ds, data_variable, raster_access,
                                                 decoder, decoder_kwargs)
        dar = self.__post_proc_data_netcdf4(dar, dw_tile, dm_tile, self._ref_nodatavals[data_variable])

        return dar

    def __load_data_per_tile_netcdf4(self, src_tile, dst_tile, data_variables, agg_dim='layer_id', auto_decode=False,
                                     decoder=None, decoder_kwargs=None) -> xr.Dataset:
        """
        Creates an xarray dataset per tile for a given set of data variables from a multi-file netCDF4 dataset.

        Parameters
        ----------
        src_tile : Tile
            Source tile representing the spatial extent of the data window to read from.
        dst_tile : Tile
            Target tile representing the spatial extent of the data window to write to.
        data_variables : list
            Data variables to read.
        agg_dim : str, optional
            Dimension to aggregate on (defaults to 'layer_id').
        auto_decode : bool, optional
            True if NetCDF data should be decoded according to the information available in its metadata. Defaults to
            False.
        decoder : callable, optional
            Function allowing to decode NetCDF data read from disk.
        decoder_kwargs : dict, optional
            Keyword arguments for the decoder.

        Returns
        -------
        xr.Dataset:
            Dataset corresponding to the spatial and variable selection.

        """
        tile_id = src_tile.parent_root.name
        # create new tile representing the part to actually read
        dw_tile = src_tile.parent_root.slice_by_geom(dst_tile)
        raster_access = RasterAccess(dw_tile, dst_tile)
        file_register = self.file_register.loc[self.file_register[self._tile_dim] == tile_id]
        filepaths = list(file_register['filepath'])
        nc_ds = MFDataset(filepaths, aggdim=agg_dim)
        nc_ds.set_auto_maskandscale(auto_decode)
        tile_data = {data_variable: self.__load_data_per_data_variable_netcdf4(nc_ds, data_variable, dw_tile, src_tile,
                                                                               raster_access,
                                                                               decoder, decoder_kwargs)
                     for data_variable in data_variables}
        metadata = {data_variable: NetCdf4File.get_metadata(nc_ds[data_variable])
                    for data_variable in data_variables}

        times = netCDF4.num2date(nc_ds['time'][:],  # TODO: generalise stack dimension
                                 units=getattr(nc_ds['time'], 'units', None),
                                 calendar=getattr(nc_ds['time'], 'calendar', 'standard'),
                                 only_use_cftime_datetimes=False,
                                 only_use_python_datetimes=True)

        return self._to_xarray(tile_data, dw_tile, times, metadata)

    def __read_xarray(self, dst_tile, data_variables=None, parallel=True, agg_dim='layer_id', compute=True,
                      auto_decode=False, decoder=None, decoder_kwargs=None, **kwargs) -> xr.Dataset:
        """
        Reads NetCDF data using the `open_mfdataset` function of the xarray library.

        Parameters
        ----------
        dst_tile : geospade.raster.Tile
            Target tile representing the spatial extent of the data window to read from.
        data_variables : list, optional
            Data variables to read. Default is to read all available data variables.
        parallel : bool, optional
            Flag to activate parallelisation or not when using 'xarray' as an engine. Defaults to True.
        agg_dim : str, optional
            Dimension to aggregate on (defaults to 'layer_id').
        compute : bool, optional
            True if values from a dask array should be loaded into RAM (default).
        auto_decode : bool, optional
            True if NetCDF data should be decoded according to the information available in its metadata. Defaults to
            False.
        decoder : callable, optional
            Function allowing to decode NetCDF data read from disk.
        decoder_kwargs : dict, optional
            Keyword arguments for the decoder.

        Returns
        -------
        xr.Dataset :
            Read NetCDF variables represented as an xarray.Dataset instance.

        """
        data_variables = data_variables or self._ref_data_variables
        decoder_kwargs = decoder_kwargs or dict()
        data = [self.__load_data_per_tile_xarray(src_tile, dst_tile, data_variables, parallel, agg_dim, compute,
                                                 auto_decode, decoder, decoder_kwargs, **kwargs)
                for src_tile in self._mosaic.tiles]

        return xr.combine_by_coords(data)

    def __post_proc_data_xarray(self, dar, dw_tile, dm_tile, nodataval=0, compute=True) -> xr.DataArray:
        """
        Masks the given array.

        Parameters
        ----------
        dar : xr.DataArray
            Array to mask.
        dw_tile : Tile
            Tile representing the data window to read from.
        dm_tile : Tile
            Tile to extract a data mask from.
        nodataval : float, optional
            No data value being assigned where the mask values evaluate to false (defaults to 0).
        compute : bool, optional
            True if values from a dask array should be loaded into RAM (default).

        Returns
        -------
        ar : xr.DataArray
            Masked array.

        """
        if compute:
            dar = dar.compute()
        if dm_tile.mask is not None:
            raster_access = RasterAccess(dm_tile, dw_tile, src_root_raster_geom=dw_tile)
            tile_mask = np.zeros(dw_tile.shape)
            tile_mask[raster_access.src_row_slice, raster_access.src_col_slice] = dm_tile.mask
            dar = dar.where(tile_mask.astype(bool), nodataval)

        return dar

    def __load_data_per_data_variable_xarray(self, ds, data_variable, dw_tile, dm_tile, raster_access,
                                      decoder=None, decoder_kwargs=None, compute=True) -> xr.DataArray:
        """
        Selects, mask and slices an xarray data array.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset to subset.
        data_variable : str
            Data variable to select.
        dw_tile : Tile
            Tile representing the data window to read from.
        dm_tile : Tile
            Tile to extract a data mask from.
        raster_access : RasterAccess
            Helper instance to slice the data array.
        decoder : callable, optional
            Function allowing to decode NetCDF data read from disk.
        decoder_kwargs : dict, optional
            Keyword arguments for the decoder.
        compute : bool, optional
            True if values from a dask array should be loaded into RAM (default).

        Returns
        -------
        dar : xr.DataArray
            Decoded data subset.

        """
        dar = self.__load_data_per_data_variable(ds, data_variable, raster_access, decoder, decoder_kwargs)
        dar = self.__post_proc_data_xarray(dar, dw_tile, dm_tile, self._ref_nodatavals[data_variable], compute)

        return dar

    def __load_data_per_tile_xarray(self, src_tile, dst_tile, data_variables, parallel=True,
                                    agg_dim='layer_id', compute=True, auto_decode=False, decoder=None,
                                    decoder_kwargs=None, **kwargs) -> xr.Dataset:
        """
        Creates an xarray dataset per tile for a given set of data variables.

        Parameters
        ----------
        src_tile : Tile
            Source tile representing the spatial extent of the data window to read from.
        dst_tile : Tile
            Target tile representing the spatial extent of the data window to write to.
        data_variables : list
            Data variables to read.
        parallel : bool, optional
            Flag to activate parallelisation or not when using 'xarray' as an engine. Defaults to True.
        agg_dim : str, optional
            Dimension to aggregate on (defaults to 'layer_id').
        compute : bool, optional
            True if values from a dask array should be loaded into RAM (default).
        auto_decode : bool, optional
            True if NetCDF data should be decoded according to the information available in its metadata. Defaults to
            False.
        decoder : callable, optional
            Function allowing to decode NetCDF data read from disk.
        decoder_kwargs : dict, optional
            Keyword arguments for the decoder.

        Returns
        -------
        xr.Dataset:
            Dataset corresponding to the spatial and variable selection.

        """
        tile_id = src_tile.parent_root.name
        # create new tile representing the part to actually read
        dw_tile = src_tile.parent_root.slice_by_geom(dst_tile)
        raster_access = RasterAccess(dw_tile, dst_tile)
        file_register = self.file_register.loc[self.file_register[self._tile_dim] == tile_id]
        filepaths = file_register['filepath']
        xr_ds = xr.open_mfdataset(filepaths, concat_dim=agg_dim, combine="nested", data_vars='minimal',
                                  coords='minimal', compat='override', parallel=parallel,
                                  mask_and_scale=auto_decode, **kwargs)
        data_tile = {data_variable: self.__load_data_per_data_variable_xarray(xr_ds, data_variable, dw_tile, src_tile,
                                                                              raster_access, decoder,
                                                                              decoder_kwargs, compute)
                     for data_variable in data_variables}
        ref_coords = data_tile[data_variables[0]].coords
        return xr.Dataset(data_tile, coords=ref_coords, attrs=xr_ds.attrs)

    def _to_xarray(self, data, tile, times, metadata) -> xr.Dataset:
        """
        Converts NetCDF data being available as a NumPy array to an xarray dataset.

        Parameters
        ----------
        data : dict
            Dictionary mapping data variables with NetCDF variable data being available as a NumPy array.
        tile : geospade.raster.Tile
            Tile representing the spatial extent of `data`.
        times : list
            List of datetime instances representing the temporal coordinates of the image stack.
        metadata : dict
            Metadata attributes for each data variable.

        Returns
        -------
        xrds : xr.Dataset

        """
        space_dim_names = self._ref_space_dims
        stack_dim_names = self._ref_stack_dims
        all_dim_names = stack_dim_names + space_dim_names
        coord_dict = dict()
        coord_dict[stack_dim_names[0]] = times
        coord_dict[space_dim_names[0]] = tile.y_coords
        coord_dict[space_dim_names[1]] = tile.x_coords

        data_variables = list(data.keys())
        xar_dict = {data_variable: xr.DataArray(data[data_variable], coords=coord_dict, dims=all_dim_names,
                                                attrs=metadata[data_variable])
                    for data_variable in data_variables}
        xrds = xr.Dataset(data_vars=xar_dict)

        return xrds


class NetCdfWriter(RasterDataWriter):
    """ Allows to write and manage a stack of NetCDF files. """
    def __init__(self, mosaic, file_register=None, data=None, stack_dimension='layer_id', stack_coords=None,
                 tile_dimension='tile_id', dirpath=None, fn_pattern='{layer_id}.tif', fn_formatter=None):
        """
        Constructor of `NetCdfWriter`.

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
            extent or tiling. Moreover, the dimension of the mosaic along the first dimension (stack/file dimension),
            must match the entries/filepaths in `file_register`.
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
            Directory path to the folder where the NetCDF files should be written to. Defaults to None, i.e. the
            current working directory is used.
        fn_pattern : str, optional
            Pattern for the filename of the new NetCDF files. To fill in specific parts of the new file name with
            information from the file register, you can specify the respective file register column names in curly
            brackets and add them to the pattern string as desired. Defaults to '{layer_id}.tif'.
        fn_formatter : dict, optional
            Dictionary mapping file register column names with functions allowing to encode their values as strings.

        """

        super().__init__(mosaic, file_register=file_register, data=data, stack_dimension=stack_dimension,
                         stack_coords=stack_coords, tile_dimension=tile_dimension, dirpath=dirpath,
                         fn_pattern=fn_pattern, fn_formatter=fn_formatter)

    @classmethod
    def from_data(self, data, filepath, mosaic=None, tile_dimension='tile_id', **kwargs) -> "NetCdfWriter":
        """
        Creates `NetCdfWriter` instance from an xarray.Dataset instance and a target file path, i.e. this function
        should help to write/export the whole image stack to one file.

        Parameters
        ----------
        data : xr.Dataset
            Dataset to write to disk.
        filepath : str
            Full system path to NetCDF file to write to.
        mosaic : geospade.raster.MosaicGeometry, optional
            Mosaic representing the spatial allocation of the given file. The tiles of the mosaic have to match the
            ID's/names of the 'tile_id' column. If it is None, a one-tile mosaic will be created from the given
            mosaic.
        tile_dimension : str, optional
            Dimension/column name of the dimension containing tile ID's in correspondence with the tiles in `mosaic`.
            Defaults to 'tile_id'.
        kwargs : dict, optional
            Key-word arguments for initialising the `NetCdfWriter` class.

        Returns
        -------
        NetCdfWriter

        """
        file_register_dict = dict()
        file_register_dict[tile_dimension] = ['0']
        file_register_dict['filepath'] = [filepath]
        file_register = pd.DataFrame(file_register_dict)
        return super().from_xarray(data, file_register, mosaic=mosaic, tile_dimension=tile_dimension, **kwargs)

    def __get_encoding_info_from_data(self, data, data_variables) -> Tuple[dict, dict, dict, dict]:
        """
        Extracts encoding information from an xarray dataset.

        Parameters
        ----------
        data : xr.Dataset
            Data to extract encoding info from.
        data_variables : list of str
            List of data variable names.

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
        for data_variable in data_variables:
            dtypes[data_variable] = data[data_variable].data.dtype.name
            nodatavals[data_variable] = data[data_variable].attrs.get('_FillValue', 0)
            scale_factors[data_variable] = data[data_variable].attrs.get('scale_factor', 1)
            offsets[data_variable] = data[data_variable].attrs.get('add_offset', 0)

        return nodatavals, scale_factors, offsets, dtypes

    def write(self, data, use_mosaic=False, data_variables=None, encoder=None, encoder_kwargs=None, overwrite=False,
              unlimited_dims=None, **kwargs):
        """
        Writes a certain chunk of NetCDF data to disk.

        Parameters
        ----------
        data : xr.Dataset
            Data chunk to be written to disk or being appended to existing mosaic.
        use_mosaic : bool, optional
            True if data should be written according to the mosaic.
            False if data composes a new tile and should not be tiled (default).
        data_variables : list of str, optional
            Data variables to write. Defaults to None, i.e. all data variables are written.
        encoder : callable, optional
            Function allowing to encode a xarray dataset before writing it to disk.
        encoder_kwargs : dict, optional
            Keyword arguments for the encoder.
        overwrite : bool, optional
            True if the NetCDF file(s) should be overwritten, False if not (default).
        unlimited_dims : list of str or str, optional
            List of dimension names specifying the dimensions to be stored as unlimited.
        kwargs : dict, optional
            Key-word arguments for creating a `NetCdf4File` instance.

        """
        data_geom = self.raster_geom_from_data(data, sref=self.mosaic.sref)
        unlimited_dims = to_list(unlimited_dims)
        data_filt = data if data_variables is None else data[data_variables]
        data_variables = list(data_filt.data_vars)
        all_dims = list(data_filt.dims)
        space_dims = all_dims[-2:]
        stack_dim_names = all_dims[:-2]
        nodatavals, scale_factors, offsets, dtypes = self.__get_encoding_info_from_data(data, data_variables)

        for filepath, file_group in self._file_register.groupby('filepath'):
            tile_id = file_group.iloc[0].get(self._tile_dim, '0')

            if use_mosaic:
                src_tile = self._mosaic[tile_id]
                if not src_tile.intersects(data_geom):
                    continue
                dst_tile = data_geom.slice_by_geom(src_tile, inplace=False)
                data_write = data_filt.sel(**{space_dims[0]: np.around(dst_tile.y_coords, decimals=DECIMALS),
                                              space_dims[1]: np.around(dst_tile.x_coords, decimals=DECIMALS)})
            else:
                dst_tile = data_geom
                src_tile = data_geom
                data_write = data_filt

            file_coords = list(file_group[self._file_dim])
            data_write = data_write.sel(**{self._file_dim: file_coords})
            data_write = data_write[data_variables]
            stack_dims = {stack_dim_name: None if stack_dim_name in unlimited_dims else len(data_write[stack_dim_name])
                          for stack_dim_name in stack_dim_names}

            file_id = file_group.iloc[0].get('file_id', None)
            if file_id is None:
                gt_driver = NetCdf4File(filepath, mode='w', geotrans=src_tile.geotrans, sref_wkt=src_tile.sref.wkt,
                                        stack_dims=stack_dims,
                                        space_dims={space_dims[0]: src_tile.n_rows, space_dims[1]: src_tile.n_cols},
                                        data_variables=data_variables, dtypes=dtypes,
                                        scale_factors=scale_factors, offsets=offsets, nodatavals=nodatavals,
                                        attrs={'time': {'units': 'days since 1950-01-01 00:00:00'}},  # TODO: make this more flexible (this needs to be defined from outside)
                                        metadata=data_write.attrs, **kwargs)
                file_id = len(list(self._files.keys())) + 1
                self._files[file_id] = gt_driver
                self._file_register.loc[file_group.index, 'file_id'] = file_id

            nc_file = self._files[file_id]
            raster_access = RasterAccess(dst_tile, src_tile, src_root_raster_geom=data_geom)
            nc_file.write(data_write, row=raster_access.dst_window[0], col=raster_access.dst_window[1],
                          encoder=encoder, encoder_kwargs=encoder_kwargs)

    def export(self, use_mosaic=False, data_variables=None, encoder=None, encoder_kwargs=None, overwrite=False,
               unlimited_dims=None, **kwargs):
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
            Function allowing to encode an xarray dataset before writing it to disk.
        encoder_kwargs : dict, optional
            Keyword arguments for the encoder.
        overwrite : bool, optional
            True if the NetCDF file(s) should be overwritten, False if not (default).
        unlimited_dims : list of str or str, optional
            List of dimension names specifying the dimensions to be stored as unlimited.
        kwargs : dict, optional
            Key-word arguments for creating a `NetCdf4File` instance.

        """
        self.write(self.data_view, use_mosaic, data_variables, encoder, encoder_kwargs, overwrite, unlimited_dims,
                   **kwargs)


if __name__ == '__main__':
    pass
