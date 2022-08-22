""" Manages I/O for a NetCDF file. """

import os.path
from collections import OrderedDict
import netCDF4
import numpy as np
from affine import Affine
import rioxarray as rio
import xarray as xr
from osgeo import osr
from veranda.utils import to_list
from typing import Tuple


class NetCdf4File:
    """
    Wrapper for reading and writing NetCDF files with netCDF4 as a backend. By default, it will create three
    predefined dimensions 'time', 'y', 'x', with time as an unlimited dimension
    and x, y are either defined by a pre-defined shape or by an in-memory dataset.

    The spatial dimensions are always limited, but can have an arbitrary name, whereas the dimension(s) which are used
    to stack spatial data are unlimited by default and can also have an arbitrary name.

    """

    def __init__(self, filepath, mode="r", data_variables=None, stack_dims=None, space_dims=None, scale_factors=1,
                 offsets=0, nodatavals=127, dtypes='int8', zlibs=True, complevels=2, chunksizes=None,
                 var_chunk_caches=None, attrs=None, geotrans=(0, 1, 0, 0, 0, 1), sref_wkt=None,
                 metadata=None, nc_format="NETCDF4_CLASSIC", overwrite=True, auto_decode=False):
        """
        Constructor of `NetCdf4File`.

        Parameters
        ----------
        filepath : str
            Full system path to a NetCDF file.
        mode : str, optional
            File opening mode. The following modes are available:
                'r' : read data from an existing netCDF4.Dataset (default)
                'w' : write data to a new netCDF4.Dataset
                'a' : append data to an existing netCDF4.Dataset
        data_variables : list of str, optional
            Data variables stored in the NetCDF file. If `mode='w'`, then the data variable names are used to match
            the given encoding attributes, e.g. `scale_factors`, `offsets`, ...
        stack_dims : dict, optional
            Dictionary containing the dimension names used to stack the data over space as keys and their length as
            values. By default it is set to {'time': None}, i.e. an unlimited temporal dimension.
        space_dims : dict, optional
            Dictionary containing the spatial dimension names as keys and their length as values. By default it is set
            to {'y': None, 'x': None}. Note that the spatial dimensions will never be stored as unlimited - they are
            set as soon as data is read from or written to disk.
        scale_factors : dict or number, optional
            Scale factor used for de- or encoding. Defaults to 1. It can either be one value (will be used for all
            data variables), or a dictionary mapping the data variable with the respective scale factor.
        offsets : dict or number, optional
            Offset used for de- or encoding. Defaults to 0. It can either be one value (will be used for all data
            variables), or a dictionary mapping the data variable with the respective offset.
        nodatavals : dict or number, optional
            No data value used for de- or encoding. Defaults to 127. It can either be one value (will be used for all
            data variables), or a dictionary mapping the data variable with the respective no data value.
        dtypes : dict or str, optional
            Data types used for de- or encoding (NumPy-style). Defaults to 'int8'. It can either be one value (will
            be used for all data variables), or a dictionary mapping the data variable with the respective data type.
        zlibs : dict or bool, optional
            Flag if ZLIB compression should be applied or not. Defaults to True. It can either be one value (will
            be used for all data variables), or a dictionary mapping the data variable with the respective flag value.
        complevels : dict or int, optional
            Compression levels used during de- or encoding. Defaults to 2. It can either be one value (will
            be used for all data variables), or a dictionary mapping the data variable with the respective compression
            level.
        chunksizes : dict or tuple, optional
            Chunk sizes given as a 3-tuple specifying the chunk sizes for each dimension. Defaults to None, i.e. no
            chunking is used. It can either be one value/3-tuple (will be used for all data variables), or a dictionary
            mapping the data variable with the respective chunk sizes.
        var_chunk_caches : dict or tuple, optional
            Chunk cache settings given as a 3-tuple specifying the chunk cache settings size, nelems, preemption.
            Defaults to None, i.e. the default NetCDF4 settings are used. It can either be one value/3-tuple (will be
            used for all data variables), or a dictionary mapping the data variable with the respective chunk cache
            settings.
        attrs : dict, optional
            Data variable specific attributes. This can be an important parameter, when for instance setting the units
            of a data variable. Defaults to None. It can either be one dictionary (will be used for all data variables,
            so maybe the `metadata` parameter is more suitable), or a dictionary mapping the data variable with the
            respective attributes.
        geotrans : 6-tuple or list, optional
            Geo-transformation parameters with the following entries:
                0: Top-left X coordinate
                1: W-E pixel sampling
                2: Rotation, 0 if image is axis-parallel
                3: Top-left Y coordinate
                4: Rotation, 0 if image is axis-parallel
                5: N-S pixel sampling (negative value if North-up)
            Defaults to (0, 1, 0, 0, 0, 1).
        sref_wkt : str, optional
            Coordinate Reference System (CRS) information in WKT format. Defaults to None.
        metadata : dict, optional
            Global metadata attributes. Defaults to None.
        nc_format : str, optional
            NetCDF formats:
                - 'NETCDF4_CLASSIC' (default)
                - 'NETCDF3_CLASSIC',
                - 'NETCDF3_64BIT_OFFSET',
                - 'NETCDF3_64BIT_DATA'.
        overwrite : bool, optional
            Flag if the file can be overwritten if it already exists (defaults to False).
        auto_decode : bool, optional
            True if data should be decoded according to the information available in its header.
            False if not (default).

        """

        self.src = None
        self.src_vars = {}
        self.filepath = filepath
        self.mode = mode
        self.data_variables = to_list(data_variables)
        self.sref_wkt = sref_wkt
        self.geotrans = geotrans
        self.overwrite = overwrite
        self.metadata = metadata or dict()
        self.auto_decode = auto_decode
        self.gm_name = None
        self.nc_format = nc_format
        self.stack_dims = stack_dims or {'time': None}
        self.space_dims = space_dims or {'y': None, 'x': None}
        if len(self.space_dims) != 2:
            err_msg = "The number of spatial dimensions must equal 2."
            raise ValueError(err_msg)
        self.attrs = attrs or dict()

        scale_factors = self.__to_dict(scale_factors)
        offsets = self.__to_dict(offsets)
        nodatavals = self.__to_dict(nodatavals)
        dtypes = self.__to_dict(dtypes)
        chunksizes = self.__to_dict(chunksizes)
        zlibs = self.__to_dict(zlibs)
        complevels = self.__to_dict(complevels)
        var_chunk_caches = self.__to_dict(var_chunk_caches)

        self.__set_coding_info_from_input(nodatavals, scale_factors, offsets, dtypes, zlibs, complevels, chunksizes,
                                          var_chunk_caches)

        if self.mode == 'r':
            self._open()
        elif self.mode == 'w' and None not in self.space_dims.values():
            self._open()

    @property
    def all_variables(self) -> list:
        """ Returns all relevant (data, spatial and stack) variables of the NetCDF file. """
        return self.data_variables + list(self.stack_dims.keys()) + list(self.space_dims.keys())

    @property
    def raster_shape(self) -> Tuple[int, int]:
        """ Tuple specifying the shape of the raster (defined by the spatial dimensions). """
        space_dims = list(self.space_dims.keys())
        return len(self.src_vars[space_dims[0]]), len(self.src_vars[space_dims[1]])

    def _reset(self):
        """ Resets internal class variables with properties from an existing NetCDF dataset. """
        all_dims = list(self.src.dimensions.keys())
        space_dim_names = all_dims[-2:]
        stack_dim_names = all_dims[:-2]
        self.stack_dims = {stack_dim_name: self._get_stack_dim_size(stack_dim_name)
                           for stack_dim_name in stack_dim_names}
        self.space_dims = {space_dim_name: self.src.dimensions[space_dim_name].size
                           for space_dim_name in space_dim_names}

        dims = tuple(list(self.stack_dims.keys()) + list(self.space_dims.keys()))
        self.data_variables = [src_var.name for src_var in self.src_vars.values()
                               if src_var.dimensions == dims]
        for variable in self.all_variables:
            self.__set_coding_per_var_from_src(variable)

    def _reset_from_ds(self, ds):
        """ Resets internal class variables with properties from an in-memory xarray dataset. """
        if len(self.data_variables) == 0:
            self.data_variables = list(ds.data_vars.keys())

        space_dims = list(self.space_dims.keys())
        self.space_dims[space_dims[0]] = self.space_dims[space_dims[0]] or len(ds[space_dims[0]])
        self.space_dims[space_dims[1]] = self.space_dims[space_dims[1]] or len(ds[space_dims[1]])
        stack_dims = {dim: len(ds[dim]) for dim in ds.dims if dim not in space_dims}
        stack_dims.update(self.stack_dims)
        self.stack_dims = stack_dims
        self.__set_coding_from_xarray(ds)

    def _open(self):
        """ Internal method for opening a NetCDF file. """
        if self.mode == "r":
            self.__open_read()
        if self.mode == "a":
            self.__open_append()
        if self.mode in ["r", "a"]:
            self.__set_spatial_attrs()
        if self.mode == "w":
            self.__open_write()

    def __set_spatial_attrs(self):
        """
        Sets/updates all spatial attributes, i.e. geo-transformation parameters, CRS information as a WKT string,
        and the grid mapping name.

        """
        self.gm_name = self.__get_gm_name()
        if self.gm_name is not None:
            metadata = self.get_metadata(self.src_vars[self.gm_name])
            geotrans = metadata.get('GeoTransform', None)
            if geotrans is not None:
                self.geotrans = tuple(map(float, geotrans.split(' ')))
            self.sref_wkt = metadata.get('spatial_ref', None)

    def __create_x_variable(self):
        """ Creates variable for storing coordinates in X direction. """
        space_dims = list(self.space_dims.keys())
        space_dim2 = space_dims[1]
        n_cols = self.space_dims[space_dim2]
        self.src.createDimension(space_dim2, n_cols)
        attr = OrderedDict([
            ('standard_name', 'projection_x_coordinate'),
            ('long_name', 'x coordinate of projection')])
        x = self.src.createVariable(space_dim2, 'float64', (space_dim2,))
        attr.update(self.attrs.get(space_dim2, dict([('units', 'm')])))
        x.setncatts(attr)
        x[:] = self.geotrans[0] + \
               (0.5 + np.arange(n_cols)) * self.geotrans[1] + \
               (0.5 + np.arange(n_cols)) * self.geotrans[2]
        self.src_vars[space_dim2] = x

    def __create_y_variable(self):
        """ Creates variable for storing coordinates in Y direction. """
        space_dims = list(self.space_dims.keys())
        space_dim1 = space_dims[0]
        n_rows = self.space_dims[space_dim1]
        self.src.createDimension(space_dim1, n_rows)
        attr = dict([('standard_name', 'projection_y_coordinate'),
                     ('long_name', 'y coordinate of projection')])
        y = self.src.createVariable(space_dim1, 'float64', (space_dim1,), )
        attr.update(self.attrs.get(space_dim1, dict([('units', 'm')])))
        y.setncatts(attr)
        y[:] = self.geotrans[3] + \
               (0.5 + np.arange(n_rows)) * self.geotrans[4] + \
               (0.5 + np.arange(n_rows)) * self.geotrans[5]
        self.src_vars[space_dim1] = y

    def __create_stack_variables(self):
        """ Creates variables for storing coordinates along the stack dimensions. """
        stck_counter = 0
        for stack_dim, n_vals in self.stack_dims.items():
            self.src.createDimension(stack_dim, n_vals)
            chunksizes = self._chunksizes[stack_dim]
            chunksizes = (chunksizes[stck_counter],) if chunksizes is not None else chunksizes
            zlib = self._zlibs[stack_dim]
            complevel = self._complevels[stack_dim]
            self.src_vars[stack_dim] = self.src.createVariable(stack_dim, np.float64, (stack_dim,),
                                                               chunksizes=chunksizes, zlib=zlib,
                                                               complevel=complevel)
            self.src_vars[stack_dim].setncatts(self.attrs.get(stack_dim, dict()))
            stck_counter += 1

    def __create_gm_variable(self):
        """ Creates variable for storing grid mapping information. """
        gm_name = None
        sref = None
        if self.sref_wkt is not None:
            sref = osr.SpatialReference()
            sref.ImportFromWkt(self.sref_wkt)
            gm_name = sref.GetName()

        if gm_name is not None:
            self.gm_name = gm_name.lower()
            proj4_dict = {}
            for subset in sref.ExportToProj4().split(' '):
                x = subset.split('=')
                if len(x) == 2:
                    proj4_dict[x[0]] = x[1]

            false_e = float(proj4_dict.get('+x_0', np.nan))
            false_n = float(proj4_dict.get('+y_0', np.nan))
            lat_po = float(proj4_dict.get('+lat_0', np.nan))
            lon_po = float(proj4_dict.get('+lon_0', np.nan))
            long_name = 'CRS definition'
            semi_major_axis = sref.GetSemiMajor()
            inverse_flattening = sref.GetInvFlattening()
            geotrans = "{:} {:} {:} {:} {:} {:}".format(
                self.geotrans[0], self.geotrans[1],
                self.geotrans[2], self.geotrans[3],
                self.geotrans[4], self.geotrans[5])

            attr = OrderedDict([('grid_mapping_name', self.gm_name),
                                ('false_easting', false_e),
                                ('false_northing', false_n),
                                ('latitude_of_projection_origin', lat_po),
                                ('longitude_of_projection_origin', lon_po),
                                ('long_name', long_name),
                                ('semi_major_axis', semi_major_axis),
                                ('inverse_flattening', inverse_flattening),
                                ('spatial_ref', self.sref_wkt),
                                ('GeoTransform', geotrans)])
        else:
            self.gm_name = "proj_unknown"
            geotrans = "{:} {:} {:} {:} {:} {:}".format(
                self.geotrans[0], self.geotrans[1],
                self.geotrans[2], self.geotrans[3],
                self.geotrans[4], self.geotrans[5])
            attr = OrderedDict([('grid_mapping_name', self.gm_name),
                                ('GeoTransform', geotrans)])

        crs = self.src.createVariable(self.gm_name, 'S1', ())
        crs.setncatts(attr)

    def __create_data_variable(self, data_var_name):
        """
        Creates data variable.

        Parameters
        ----------
        data_var_name : str
            Data variable name.
        """
        dims = tuple(list(self.stack_dims.keys()) + list(self.space_dims.keys()))

        zlib = self._zlibs.get(data_var_name)
        complevel = self._complevels[data_var_name]
        chunksizes = self._chunksizes[data_var_name]
        var_chunk_cache = self._var_chunk_caches[data_var_name]
        nodataval = self.nodatavals[data_var_name]
        dtype = self.dtypes[data_var_name]

        self.src_vars[data_var_name] = self.src.createVariable(
            data_var_name, dtype, dims,
            chunksizes=chunksizes, zlib=zlib,
            complevel=complevel, fill_value=nodataval)
        self.src_vars[data_var_name].set_auto_scale(self.auto_decode)

        if var_chunk_cache is not None:
            self.src_vars[data_var_name].set_var_chunk_cache(*var_chunk_cache[:3])

        if self.gm_name is not None:
            self.src_vars[data_var_name].setncattr('grid_mapping', self.gm_name)

    def __open_read(self):
        """ Creates a new netCDF4 dataset source in read mode. """
        self.src = netCDF4.Dataset(self.filepath, mode="r")
        self.src.set_auto_maskandscale(self.auto_decode)
        self.src_vars = self.src.variables
        self._reset()

        for var in self.data_variables:
            var_chunk_cache = self._var_chunk_caches[var]
            if var_chunk_cache is not None:
                self.src_vars[var].set_var_chunk_cache(*var_chunk_cache[:3])

    def __open_append(self):
        """ Creates a new netCDF4 dataset source in append mode. """
        self.src = netCDF4.Dataset(self.filepath, mode="a")
        self.src_vars = self.src.variables
        self._reset()

    def __open_write(self):
        """ Creates a new netCDF4 dataset source in write mode. """
        self.src = netCDF4.Dataset(self.filepath, mode="w",
                                   clobber=self.overwrite,
                                   format=self.nc_format)
        self.__create_gm_variable()
        self.__create_stack_variables()
        self.__create_y_variable()
        self.__create_x_variable()
        for data_variable in self.data_variables:
            self.__create_data_variable(data_variable)

    def read(self, row=0, col=0, n_rows=None, n_cols=None, data_variables=None, decoder=None,
             decoder_kwargs=None) -> xr.Dataset:
        """
        Read data from a NetCDF file.

        Parameters
        ----------
        row : int, optional
            Row number/index. Defaults to 0.
        col : int, optional
            Column number/index. Defaults to 0.
        n_rows : int, optional
            Number of rows of the reading window (counted from `row`). If None (default), then the full extent of
            the raster is used.
        n_cols : int, optional
            Number of columns of the reading window (counted from `col`). If None (default), then the full extent of
            the raster is used.
        data_variables : list, optional
            Data variables to read. Default is to read all available data variables.
        decoder : callable, optional
            Decoding function expecting a NumPy array as input.
        decoder_kwargs : dict, optional
            Keyword arguments for the decoder.

        Returns
        -------
        data : xarray.Dataset
            Data stored in the NetCDF file.

        """
        decoder_kwargs = decoder_kwargs or dict()
        n_rows = self.raster_shape[0] if n_rows is None else n_rows
        n_cols = self.raster_shape[1] if n_cols is None else n_cols
        data_variables = data_variables or self.data_variables
        chunks = self._get_chunks(data_variables[0])
        data_xr = xr.open_dataset(xr.backends.NetCDF4DataStore(self.src), mask_and_scale=self.auto_decode,
                                  chunks=chunks)

        data = None
        for data_variable in data_variables:
            data = self._read_data_variable(data_xr, data_variable, row, col, n_rows, n_cols, decoder, decoder_kwargs,
                                            data)

        return data

    def write(self, ds, row=0, col=0, encoder=None, encoder_kwargs=None):
        """
        Write an xarray dataset into a NetCDF file.

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset to write to disk.
        row : int, optional
            Offset row number/index (defaults to 0).
        col : int, optional
            Offset column number/index (defaults to 0).
        encoder : callable, optional
            Encoding function expecting an xarray.DataArray as input.
        encoder_kwargs : dict, optional
            Keyword arguments for the encoder.

        """
        encoder_kwargs = encoder_kwargs or dict()
        data_variables = list(ds.data_vars.keys())
        self._reset_from_ds(ds)

        # open file and create dimensions and coordinates
        if self.src is None:
            self._open()

        stack_idxs = self._write_and_slice_stack_dim(ds)
        y_slice = self._write_and_slice_y_dim(ds, row)
        x_slice = self._write_and_slice_x_dim(ds, col)
        ds_idxs = stack_idxs + [y_slice] + [x_slice]

        for data_variable in data_variables:
            self._write_data_variable(ds, ds_idxs, data_variable, encoder, encoder_kwargs)

        self.src.setncatts(ds.attrs)
        self.src.setncatts(self.metadata)

    def _read_data_variable(self, src, data_variable, row, col, n_rows, n_cols, decoder, decoder_kwargs, data=None):
        """
        Reads and slices variable specific data and optionally merges it with existing data.

        Parameters
        ----------
        src : xarray.Dataset
            Xarray dataset to extract the data from.
        data_variable : str
            Name of a data variable of interest.
        row : int
            Row number/index.
        col : int
            Column number/index.
        n_rows : int
            Number of rows of the reading window (counted from `row`).
        n_cols : int
            Number of columns of the reading window (counted from `col`).
        decoder : callable
            Decoding function expecting a NumPy array as input.
        decoder_kwargs : dict
            Keyword arguments for the decoder.
        data : xr.Dataset, optional
            Existing dataset to merge with.

        Returns
        -------
        data : xr.Dataset
            Existing dataset (optional) plus data from the given data variable.

        """
        data_sliced = src[data_variable][..., row: row + n_rows, col: col + n_cols]
        if decoder:
            data_sliced = decoder(data_sliced,
                                  nodataval=self.nodatavals[data_variable],
                                  data_variable=data_variable, scale_factor=self.scale_factors[data_variable],
                                  offset=self.offsets[data_variable], dtype=self.dtypes[data_variable],
                                  **decoder_kwargs)
        if data is None:
            data = data_sliced.to_dataset()
        else:
            data = data.merge(data_sliced.to_dataset())

        return data

    def _get_chunks(self, ref_data_var_name) -> dict:
        """
        Retrieves chunks from specific data variable.

        Parameters
        ----------
        ref_data_var_name : str
            Name of the data variable serving as reference.

        Returns
        -------
        chunks : dict
            Maps dimension names with chunk size.

        """
        ref_chunksize = self._chunksizes[ref_data_var_name]
        chunks = {stack_dim: ref_chunksize[i] for i, stack_dim in enumerate(self.stack_dims.keys())}
        space_dims = list(self.space_dims.keys())
        chunks[space_dims[0]] = ref_chunksize[-2]
        chunks[space_dims[1]] = ref_chunksize[-1]

        return chunks

    def _get_stack_dim_size(self, stack_dim_name) -> int:
        """
        Retrieves size of a stack dimension. If the dimension is unlimited, then the size is set to None.

        Parameters
        ----------
        stack_dim_name : str
            Name of the stack dimension.

        Returns
        -------
        int
            Size of the stack dimension.

        """
        is_unlimited = self.src.dimensions[stack_dim_name].isunlimited()
        return self.src.dimensions[stack_dim_name].size if not is_unlimited else None

    def _encode_temporal_dim(self, ds, stack_dim) -> np.ndarray:
        """
        Retrieves a temporal stack dimension from a given dataset and converts its timestamps to numbers referring to
        a certain reference time.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset to extract the timestamps from.
        stack_dim : str
            Name of the temporal dimension.

        Returns
        -------
        np.ndarray
            Array containing the original timestamps converted to their numerical representation.

        """
        attr = self.attrs.get(stack_dim, dict())
        units = ds[stack_dim].attrs.get('units', attr.get('units', None))
        calendar = ds[stack_dim].attrs.get('calendar', attr.get('calendar', None))
        calendar = calendar or 'standard'
        units = units or 'days since 1900-01-01 00:00:00'
        return netCDF4.date2num(ds[stack_dim].to_index().to_pydatetime(), units, calendar=calendar)

    def _write_and_slice_stack_dim(self, ds) -> list:
        """
        Writes coordinates of a stack dimension to its corresponding variable and returns a list of slices representing
        an indexing operation of the source data.

        Parameters
        ----------
        ds : xr.Dataset
            Xarray dataset.

        Returns
        -------
        ds_idxs : list
            List of stack dimension slices.

        """
        ds_idxs = []
        for stack_dim in self.stack_dims.keys():
            # determine index where to append
            if self.mode == 'a':
                append_start = self.src_vars[stack_dim].shape[0]
            else:
                append_start = 0

            # convert timestamps to index if the stack dimension is temporal
            if ds[stack_dim].dtype.name == 'datetime64[ns]':
                stack_vals = self._encode_temporal_dim(ds, stack_dim)
            else:
                stack_vals = ds[stack_dim]
            n_stack_vals = len(stack_vals)
            self.src_vars[stack_dim][append_start:append_start + n_stack_vals] = stack_vals
            ds_idxs.append(slice(append_start, None))

        return ds_idxs

    def _write_and_slice_x_dim(self, ds, col) -> slice:
        """
        Writes coordinates of a X dimension to its corresponding variable and returns a slice representing an indexing
        operation of the source data.

        Parameters
        ----------
        ds : xr.Dataset
            Xarray dataset.
        col : int
            Column number/index.

        Returns
        -------
        slice :
            Slice representing the indexing of the source data in X/column direction.

        """
        space_dims = list(self.space_dims.keys())
        dim_name = space_dims[1]
        n_cols = len(ds[dim_name])
        if dim_name in ds.coords:
            self.src_vars[dim_name][col:col + n_cols] = ds[dim_name].data
        return slice(col, col + n_cols)

    def _write_and_slice_y_dim(self, ds, row) -> slice:
        """
        Writes coordinates of a Y dimension to its corresponding variable and returns a slice representing an indexing
        operation of the source data.

        Parameters
        ----------
        ds : xr.Dataset
            Xarray dataset.
        row : int
            Row number/index.

        Returns
        -------
        slice :
            Slice representing the indexing of the source data in Y/row direction.

        """
        space_dims = list(self.space_dims.keys())
        dim_name = space_dims[0]
        n_rows = len(ds[dim_name])
        if dim_name in ds.coords:
            self.src_vars[dim_name][row:row + n_rows] = ds[dim_name].data
        return slice(row, row + n_rows)

    def _write_data_variable(self, ds, ds_idxs, data_variable, encoder, encoder_kwargs):
        """
        Encodes and writes variable specific data to disk.

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset to write to disk.
        ds_idxs : list
            List of slices defining a subset of the data source to write to.
        data_variable : str
            Name of the data variable to write.
        encoder : callable, optional
            Encoding function expecting an xarray.DataArray as input.
        encoder_kwargs : dict, optional
            Keyword arguments for the encoder.

        """
        scale_factor = self.scale_factors[data_variable]
        offset = self.offsets[data_variable]
        dtype = self.dtypes[data_variable]
        nodataval = self.nodatavals[data_variable]
        if encoder is not None:
            data_write = encoder(ds[data_variable].data,
                                 nodataval=nodataval,
                                 scale_factor=scale_factor,
                                 offset=offset,
                                 data_variable=data_variable,
                                 dtype=dtype,
                                 **encoder_kwargs)
        else:
            data_write = ds[data_variable].data
        self.src_vars[data_variable][ds_idxs] = data_write
        dar_md = ds[data_variable].attrs
        dar_md.pop('_FillValue', None)  # remove this attribute because it already exists
        dar_md.update(self.attrs.get(data_variable, dict()))
        self.src_vars[data_variable].setncatts(dar_md)

    def __set_coding_info_from_input(self, nodatavals, scale_factors, offsets, dtypes, zlibs,
                                     complevels, chunksizes, var_chunk_caches):
        """
        Sets/overwrites internal dictionaries used to store all the coding information applied during en- or decoding
        with externally provided arguments.

        Parameters
        ----------
        nodatavals : dict
            Maps the data variable with the respective no data value used for de- or encoding.
        scale_factors : dict
            Maps the data variable with the respective scale factor used for de- or encoding.
        offsets : dict
            Maps the data variable with the respective offset used for de- or encoding.
        dtypes : dict
            Maps the data variable with the respective data type (NumPy-style) used for de- or encoding.
        zlibs : dict
            Maps the data variable with the respective flag value indicating if ZLIB compression should be applied or
            not during de- or encoding..
        complevels : dict
            Maps the data variable with the respective compression level used during de- or encoding.
        chunksizes : dict
            Maps the data variable with the respective chunk sizes given as a 3-tuple specifying the chunk size for
            each dimension.
        var_chunk_caches : dict
            Maps the data variable with the respective chunk cache settings given as a 3-tuple specifying the chunk
            cache settings size, nelems, preemption.

        """
        self._var_chunk_caches = dict()
        self._zlibs = dict()
        self._complevels = dict()
        self._chunksizes = dict()
        self.scale_factors = dict()
        self.offsets = dict()
        self.nodatavals = dict()
        self.dtypes = dict()

        for variable in self.all_variables:
            self._zlibs[variable] = zlibs.get(variable, True)
            self._complevels[variable] = complevels.get(variable, 2)
            self._chunksizes[variable] = chunksizes.get(variable, None)
            self._var_chunk_caches[variable] = var_chunk_caches.get(variable, None)
            if variable in self.data_variables:
                self.scale_factors[variable] = scale_factors.get(variable, 1)
                self.offsets[variable] = offsets.get(variable, 0)
                self.nodatavals[variable] = nodatavals.get(variable, 127)
                self.dtypes[variable] = dtypes.get(variable, 'int8')

    def __set_coding_per_var_from_src(self, variable):
        """
        Sets/overwrites internal dictionaries used to store all the coding information applied during en- or decoding
        with coding information retrieved from the netCDF4 source dataset.

        Parameters
        ----------
        variable : str
            NetCDF variable name.

        """
        src_var = self.src_vars[variable]
        self._chunksizes[variable] = src_var.chunking()
        self._var_chunk_caches[variable] = self._var_chunk_caches.get(variable, src_var.get_var_chunk_cache())
        if variable in self.data_variables:
            src_var_md = self.get_metadata(src_var)
            def_scale_factor = self.scale_factors.get(variable, 1)
            self.scale_factors[variable] = src_var_md.get('scale_factor', def_scale_factor)
            def_offset = self.offsets.get(variable, 0)
            self.offsets[variable] = src_var_md.get('add_offset', def_offset)
            def_nodataval = self.nodatavals.get(variable, 127)
            self.nodatavals[variable] = src_var_md.get('_FillValue', def_nodataval)
            self.dtypes[variable] = self.dtypes.get(variable, src_var.dtype)

    def __set_coding_from_xarray(self, ds):
        """
        Sets/overwrites internal dictionaries used to store all the coding information applied during en- or decoding
        with coding information retrieved from an xarray dataset.

        Parameters
        ----------
        ds : xarray.Dataset
            Xarray dataset to retrieve coding information from.

        """
        for variable in self.all_variables:
            dar = ds[variable]
            src_var_md = dar.attrs
            self._zlibs[variable] = self._zlibs.get(variable, True)
            self._complevels[variable] = self._complevels.get(variable, 2)
            self._chunksizes[variable] = self._chunksizes.get(variable, dar.chunks)
            self._var_chunk_caches[variable] = self._var_chunk_caches.get(variable, None)
            if variable in ds.data_vars:
                def_scale_factor = self.scale_factors.get(variable, 1)
                self.scale_factors[variable] = src_var_md.get('scale_factor', def_scale_factor)
                def_offset = self.offsets.get(variable, 0)
                self.offsets[variable] = src_var_md.get('add_offset', def_offset)
                def_nodataval = self.nodatavals.get(variable, 127)
                self.nodatavals[variable] = src_var_md.get('_FillValue', def_nodataval)
                self.dtypes[variable] = self.dtypes.get(variable, dar.dtype.name)

    def __to_dict(self, arg) -> dict:
        """
        Assigns non-iterable object to a dictionary with NetCDF4 variables as keys. If `arg` is already a dictionary the
        same object is returned.

        Parameters
        ----------
        arg : non-iterable or dict
            Non-iterable, which should be converted to a dict.

        Returns
        -------
        arg_dict : dict
            Dictionary mapping NetCDF4 variables with the value of `arg`.

        """

        return {var_name: arg for var_name in self.all_variables} if not isinstance(arg, dict) else arg

    @staticmethod
    def get_metadata(src_var) -> dict:
        """
        Collects all metadata attributes from a NetCDF4 variable.

        Parameters
        ----------
        src_var : netCDF4.Variable
            NetCDF4 variable to extract metadata from.

        Returns
        -------
        attrs : dict
            Metadata attributes stored in the NetCDF4 variable.

        """
        attrs = dict()
        attrs_keys = src_var.ncattrs()
        for key in attrs_keys:
            attrs[key] = getattr(src_var, key)
        return attrs

    def __get_gm_name(self) -> str:
        """
        str : The name of the NetCDF4 variable storing information about the CRS, i.e. the variable
        containing an attribute named 'grid_mapping_name'.
        """
        gm_name = None
        for var_name in self.src_vars.keys():
            metadata = self.get_metadata(self.src_vars[var_name])
            if "grid_mapping_name" in metadata.keys():
                gm_name = metadata["grid_mapping_name"]
                break
        return gm_name

    def close(self):
        """
        Close file.
        """
        if self.src is not None:
            self.src.close()

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()


class NetCdfXrFile:
    """
    Wrapper for reading and writing NetCDF files with xarray as a backend. By default, it will create three
    predefined dimensions 'time', 'y', 'x', but spatial dimensions or dimensions which are used
    to stack data can also have an arbitrary name.

    """

    def __init__(self, filepath, mode='r', data_variables=None, stack_dims=None, space_dims=None,
                 scale_factors=1, offsets=0, nodatavals=127, dtypes='int8', compressions=None,
                 chunksizes=None, attrs=None, geotrans=(0, 1, 0, 0, 0, 1), sref_wkt=None,
                 nc_format="NETCDF4_CLASSIC", metadata=None, overwrite=True, auto_decode=False, engine='netcdf4'):
        """
        Constructor of `NetCdfXrFile`.

        Parameters
        ----------
        filepath : str
            Full system path to a NetCDF file.
        mode : str, optional
            File opening mode. The following modes are available:
                'r' : read data from an existing NetCDF file (default)
                'w' : write data to a new NetCDF file
        data_variables : list of str, optional
            Data variables stored in the NetCDF file. If `mode='w'`, then the data variable names are used to match
            the given encoding attributes, e.g. `scale_factors`, `offsets`, ...
        stack_dims : dict, optional
            Dictionary containing the dimension names to stack the data over space as keys and their length as
            values. By default it is set to {'time': None}, i.e. an unlimited temporal dimension.
        space_dims : 2-list, optional
            The two names of the spatial dimension in Y and X direction. By default it is set to ['y', 'x'].
        scale_factors : dict or number, optional
            Scale factor used for de- or encoding. Defaults to 1. It can either be one value (will be used for all
            data variables), or a dictionary mapping the data variable with the respective scale factor.
        offsets : dict or number, optional
            Offset used for de- or encoding. Defaults to 0. It can either be one value (will be used for all data
            variables), or a dictionary mapping the data variable with the respective offset.
        nodatavals : dict or number, optional
            No data value used for de- or encoding. Defaults to 127. It can either be one value (will be used for all
            data variables), or a dictionary mapping the data variable with the respective no data value.
        dtypes : dict or str, optional
            Data types used for de- or encoding (NumPy-style). Defaults to 'int8'. It can either be one value (will
            be used for all data variables), or a dictionary mapping the data variable with the respective data type.
        compressions : dict, optional
            Compression settings used for de- or encoding. Defaults to None, i.e. xarray's default values are used. It
            can either be one dictionary (will be used for all data variables), or a dictionary mapping the data
            variable with the respective compression settings. See
            https://docs.xarray.dev/en/stable/user-guide/io.html#writing-encoded-data.
        chunksizes : dict or tuple, optional
            Chunk sizes given as a 3-tuple specifying the chunk sizes for each dimension. Defaults to None, i.e. no
            chunking is used. It can either be one value/3-tuple (will be used for all data variables), or a dictionary
            mapping the data variable with the respective chunk sizes.
        attrs : dict, optional
            Data variable specific attributes. This can be an important parameter, when for instance setting the units
            of a data variable. Defaults to None. It can either be one dictionary (will be used for all data variables,
            so maybe the `metadata` parameter is more suitable), or a dictionary mapping the data variable with the
            respective attributes.
        geotrans : 6-tuple or list, optional
            Geo-transformation parameters with the following entries:
                0: Top-left X coordinate
                1: W-E pixel sampling
                2: Rotation, 0 if image is axis-parallel
                3: Top-left Y coordinate
                4: Rotation, 0 if image is axis-parallel
                5: N-S pixel sampling (negative value if North-up)
            Defaults to (0, 1, 0, 0, 0, 1).
        sref_wkt : str, optional
            Coordinate Reference System (CRS) information in WKT format. Defaults to None.
        metadata : dict, optional
            Global metadata attributes. Defaults to None.
        nc_format : str, optional
            NetCDF formats:
                - 'NETCDF4_CLASSIC' (default)
                - 'NETCDF3_CLASSIC',
                - 'NETCDF3_64BIT_OFFSET',
                - 'NETCDF3_64BIT_DATA'.
        overwrite : bool, optional
            Flag if the file can be overwritten if it already exists (defaults to False).
        auto_decode : bool, optional
            True if data should be decoded according to the information available in its header.
            False if not (default).
        engine : str, optional
            Specifies what engine should be used in the background to read or write NetCDF data. Defaults to 'netcdf4'.
            See https://docs.xarray.dev/en/stable/generated/xarray.open_dataset.html.

        """

        self.src = None
        self.filepath = filepath
        self.mode = mode
        self.data_variables = to_list(data_variables)
        self.sref_wkt = sref_wkt
        self.geotrans = geotrans
        self.metadata = metadata or dict()
        self.overwrite = overwrite
        self._engine = engine
        self.nc_format = nc_format
        self.auto_decode = auto_decode
        self.stack_dims = stack_dims or {'time': None}
        self.space_dims = space_dims or ['y', 'x']
        if len(self.space_dims) != 2:
            err_msg = "The number of spatial dimensions must equal 2."
            raise ValueError(err_msg)
        self.attrs = attrs or dict()

        compressions = self.__to_dict(compressions)
        chunksizes = self.__to_dict(chunksizes)
        scale_factors = self.__to_dict(scale_factors)
        offsets = self.__to_dict(offsets)
        nodatavals = self.__to_dict(nodatavals)
        dtypes = self.__to_dict(dtypes)

        self.__set_coding_from_external_input(nodatavals, scale_factors, offsets, dtypes, chunksizes, compressions)

        if mode == 'r':
            self._open()

    @property
    def all_variables(self):
        """ Returns all relevant (data, spatial and stack) variables of the xarray dataset. """
        return self.data_variables + list(self.stack_dims.keys()) + self.space_dims

    @property
    def raster_shape(self) -> Tuple[int, int]:
        """ 2-tuple: Tuple specifying the raster_shape of the raster (defined by the spatial dimensions). """
        return len(self.src[self.space_dims[0]]), len(self.src[self.space_dims[1]])

    def _open(self, ds=None):
        """
        Opens a NetCDF file.

        Parameters
        ----------
        ds : xarray.Dataset, optional
            Dataset used to create a new NetCDF file.

        """

        if self.mode == 'r':
            self.__open_read()
        elif self.mode == 'w':
            self.__open_write(ds)
        else:
            err_msg = f"Mode '{self.mode}' not known."
            raise ValueError(err_msg)

    def _reset(self):
        """ Resets internal class variables with properties from an existing NetCDF dataset. """
        stack_dims = list(set(self.src.dims.keys()) - set(self.space_dims))
        if self.mode == 'r':
            self.stack_dims = {stack_dim: len(self.src[stack_dim]) for stack_dim in stack_dims}

        dims = stack_dims + self.space_dims
        if len(self.data_variables) == 0:
            self.data_variables = [self.src[dvar].name for dvar in self.src.data_vars
                                   if list(self.src[dvar].dims) == dims]

        for data_variable in self.all_variables:
            self.__set_coding_per_var_from_src(data_variable)

    def __open_read(self):
        """ Creates a new xarray dataset source in read mode. """
        if not os.path.exists(self.filepath):
            err_msg = f"File '{self.filepath}' does not exist."
            raise FileNotFoundError(err_msg)
        self.src = xr.open_mfdataset(self.filepath,
                                     mask_and_scale=self.auto_decode,
                                     engine=self._engine,
                                     use_cftime=False,
                                     decode_cf=True,
                                     decode_coords="all")

        self.geotrans = tuple(self.src.rio.transform())
        rio_crs = self.src.rio.crs
        self.sref_wkt = rio_crs.to_wkt() if rio_crs is not None else None
        self.metadata = self.src.attrs
        self._reset()

    def __open_write(self, ds):
        """
        Prepares a xarray dataset for writing.

        Parameters
        ----------
        ds : xr.Dataset
            Xarray dataset to write.

        """
        self.src = ds
        if self.sref_wkt is not None:
            self.src.rio.write_crs(self.sref_wkt, inplace=True)
        self.src.rio.write_transform(Affine(*self.geotrans), inplace=True)
        self.src.attrs.update(self.metadata)
        self._reset()
        n_rows, n_cols = self.raster_shape
        ds[self.space_dims[0]] = self.geotrans[3] + \
                                  (0.5 + np.arange(n_rows)) * self.geotrans[4] + \
                                  (0.5 + np.arange(n_rows)) * self.geotrans[5]
        ds[self.space_dims[1]] = self.geotrans[0] + \
                                  (0.5 + np.arange(n_cols)) * self.geotrans[1] + \
                                  (0.5 + np.arange(n_cols)) * self.geotrans[2]

    def read(self, row=0, col=0, n_rows=None, n_cols=None, data_variables=None, decoder=None,
             decoder_kwargs=None) -> xr.Dataset:
        """
        Read mosaic from netCDF4 file.

        Parameters
        ----------
        row : int, optional
            Row number/index. Defaults to 0.
        col : int, optional
            Column number/index. Defaults to 0.
        n_rows : int, optional
            Number of rows of the reading window (counted from `row`). If None (default), then the full extent of
            the raster is used.
        n_cols : int, optional
            Number of columns of the reading window (counted from `col`). If None (default), then the full extent of
            the raster is used.
        data_variables : list, optional
            Data variables to read. Default is to read all available data variables.
        decoder : callable, optional
            Decoding function expecting an xarray.DataArray as input.
        decoder_kwargs : dict, optional
            Keyword arguments for the decoder.

        Returns
        -------
        data : xarray.Dataset
            Data stored in the NetCDF file.

        """
        decoder_kwargs = decoder_kwargs or dict()
        n_rows = self.raster_shape[0] if n_rows is None else n_rows
        n_cols = self.raster_shape[1] if n_cols is None else n_cols
        data_variables = data_variables or self.data_variables

        data = None
        for data_variable in data_variables:
            data = self._read_data_variable(data_variable, row, col, n_rows, n_cols, decoder, decoder_kwargs, data)

        return data

    def _read_data_variable(self, data_variable, row, col, n_rows, n_cols, decoder, decoder_kwargs, data=None):
        """
        Reads and slices variable specific data and optionally merges it with existing data.

        Parameters
        ----------
        data_variable : str
            Name of a data variable of interest.
        row : int
            Row number/index.
        col : int
            Column number/index.
        n_rows : int
            Number of rows of the reading window (counted from `row`).
        n_cols : int
            Number of columns of the reading window (counted from `col`).
        decoder : callable
            Decoding function expecting an xarray.DataArray as input.
        decoder_kwargs : dict
            Keyword arguments for the decoder.
        data : xr.Dataset, optional
            Existing dataset to merge with.

        Returns
        -------
        data : xr.Dataset
            Existing dataset (optional) plus data from the given data variable.

        """
        data_sliced = self.src[data_variable][..., row: row + n_rows, col: col + n_cols]
        ref_chunksize = self._chunksizes[data_variable]
        if ref_chunksize is not None:
            chunks = {stack_dim: ref_chunksize[i] for i, stack_dim in enumerate(self.stack_dims.keys())}
            chunks[self.space_dims[0]] = ref_chunksize[-2]
            chunks[self.space_dims[1]] = ref_chunksize[-1]
            data_sliced = data_sliced.chunk(chunks)
        if decoder:
            data_sliced = decoder(data_sliced,
                                  nodataval=self.nodatavals[data_variable],
                                  data_variable=data_variable, scale_factor=self.scale_factors[data_variable],
                                  offset=self.offsets[data_variable], dtype=self.dtypes[data_variable],
                                  **decoder_kwargs)
        if data is None:
            data = data_sliced.to_dataset()
        else:
            data = data.merge(data_sliced.to_dataset())

        return data

    def write(self, ds, data_variables=None, encoder=None, encoder_kwargs=None, compute=True):
        """
        Write data to a NetCDF file.

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset to write to disk.
        data_variables : list, optional
            Data variables to write. Default is to write all available data variables.
        encoder : callable, optional
            Encoding function expecting an xarray.DataArray as input.
        encoder_kwargs : dict, optional
            Keyword arguments for the encoder.
        compute : bool, optional
            If True (default), compute immediately, otherwise only a delayed dask array is stored.

        """
        self._open(ds)
        data_variables = data_variables or self.data_variables
        encoder_kwargs = encoder_kwargs or dict()
        ds_write = self.src
        unlimited_dims = [k for k, v in self.stack_dims.items() if v is None]
        encoding = {data_variable: self._encode_data_variable(ds_write, data_variable, encoder, encoder_kwargs)
                    for data_variable in data_variables}

        for dim in ds_write.dims:
            units = ds_write[dim].attrs.get('units', None)
            if units:
                encoding.update({dim.name: {'units': units}})

        ds_write.to_netcdf(self.filepath, mode=self.mode, format=self.nc_format, engine=self._engine,
                           encoding=encoding, compute=compute, unlimited_dims=unlimited_dims)

    def _encode_data_variable(self, ds, data_variable, encoder, encoder_kwargs) -> dict:
        """

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset to write to disk.
        data_variable : str
            Data variable to encode.
        encoder : callable
            Encoding function expecting an xarray.DataArray as input.
        encoder_kwargs : dict
            Keyword arguments for the encoder.

        Returns
        -------
        encoding_info : dict
            Encoding information for the given data variable.

        """
        if encoder is not None:
            ds[data_variable] = encoder(ds[data_variable], data_variable=data_variable,
                                        nodataval=self.nodatavals[data_variable],
                                        scale_factor=self.scale_factors[data_variable],
                                        offset=self.offsets[data_variable],
                                        dtype=self.dtypes[data_variable],
                                        **encoder_kwargs)
        encoding_info = dict()
        compression = self._compressions[data_variable]
        if compression is not None:
            encoding_info.update(compression)
        chunksizes = self._chunksizes[data_variable]
        if chunksizes is not None:
            encoding_info['chunksizes'] = chunksizes

        return encoding_info

    def __set_coding_from_external_input(self, nodatavals, scale_factors, offsets, dtypes, chunksizes, compressions):
        """
        Sets/overwrites internal dictionaries used to store all the coding information applied during en- or decoding
        with externally provided arguments.

        Parameters
        ----------
        nodatavals : dict
            Maps the data variable with the respective no data value used for de- or encoding.
        scale_factors : dict
            Maps the data variable with the respective scale factor used for de- or encoding.
        offsets : dict or number, optional
            Maps the data variable with the respective offset used for de- or encoding.
        dtypes : dict
            Maps the data variable with the respective data type (NumPy style) used for de- or encoding.
        compressions : dict
            Maps the data variable with the respective compression settings. See
            https://docs.xarray.dev/en/stable/user-guide/io.html#writing-encoded-data.
        chunksizes : dict
            Maps the data variable with the respective chunk sizes given as a 3-tuple specifying the chunk sizes for
            each dimension.

        """
        self._compressions = dict()
        self._chunksizes = dict()
        self.scale_factors = dict()
        self.offsets = dict()
        self.nodatavals = dict()
        self.dtypes = dict()

        for variable in self.all_variables:
            self._compressions[variable] = compressions.get(variable, None)
            self._chunksizes[variable] = chunksizes.get(variable, None)
            if variable in self.data_variables:
                self.scale_factors[variable] = scale_factors.get(variable, 1)
                self.offsets[variable] = offsets.get(variable, 0)
                self.nodatavals[variable] = nodatavals.get(variable, 127)
                self.dtypes[variable] = dtypes.get(variable, 'int8')

    def __set_coding_per_var_from_src(self, variable):
        """
        Sets/overwrites internal dictionaries used to store all the coding information applied during en- or decoding
        with coding information retrieved from the xarray source dataset.

        Parameters
        ----------
        variable : str
            Xarray dataset variable name.

        """
        self._compressions[variable] = self._compressions.get(variable, None)
        ref_chunksizes = self.src[variable].encoding.get('chunksizes', None)
        self._chunksizes[variable] = self._chunksizes.get(variable, ref_chunksizes)
        if variable in self.data_variables:
            ref_scale_factor = self.scale_factors.get(variable, 1)
            self.scale_factors[variable] = self.src[variable].attrs.get('scale_factor', ref_scale_factor)
            ref_offset = self.offsets.get(variable, 0)
            self.offsets[variable] = self.src[variable].attrs.get('add_offset', ref_offset)
            ref_nodataval = self.nodatavals.get(variable, 127)
            self.nodatavals[variable] = self.src[variable].attrs.get('_FillValue', ref_nodataval)
            ref_dtype = self.dtypes.get(variable, self.src[variable].dtype.name)
            self.dtypes[variable] = ref_dtype

    def __to_dict(self, arg) -> dict:
        """
        Assigns non-iterable object to a dictionary with NetCDF variables as keys. If `arg` is already a dictionary the
        same object is returned.

        Parameters
        ----------
        arg : non-iterable or dict
            Non-iterable, which should be converted to a dict.

        Returns
        -------
        arg_dict : dict
            Dictionary mapping NetCDF variables with the value of `arg`.

        """
        all_variables = self.data_variables + list(self.stack_dims.keys()) + self.space_dims
        if not isinstance(arg, dict) or (isinstance(arg, dict) and not isinstance(arg[list(arg.keys())[0]], dict)):
            arg_dict = dict()
            for data_variable in all_variables:
                arg_dict[data_variable] = arg
        else:
            arg_dict = arg

        return arg_dict

    def close(self):
        """
        Close file.
        """
        if self.src is not None:
            self.src.close()

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()


if __name__ == '__main__':
    pass
