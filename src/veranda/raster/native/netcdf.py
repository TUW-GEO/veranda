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

        self._var_chunk_caches = dict()
        self._zlibs = dict()
        self._complevels = dict()
        self._chunksizes = dict()
        self.scale_factors = dict()
        self.offsets = dict()
        self.nodatavals = dict()
        self.dtypes = dict()

        all_variables = self.data_variables + list(self.stack_dims.keys()) + list(self.space_dims.keys())
        for variable in all_variables:
            self._zlibs[variable] = zlibs.get(variable, True)
            self._complevels[variable] = complevels.get(variable, 2)
            self._chunksizes[variable] = chunksizes.get(variable, None)
            self._var_chunk_caches[variable] = var_chunk_caches.get(variable, None)
            if variable in self.data_variables:
                self.scale_factors[variable] = scale_factors.get(variable, 1)
                self.offsets[variable] = offsets.get(variable, 0)
                self.nodatavals[variable] = nodatavals.get(variable, 127)
                self.dtypes[variable] = dtypes.get(variable, 'int8')

        if self.mode == 'r':
            self._open()
        elif self.mode == 'w' and None not in self.space_dims.values():
            self._open()

    @property
    def raster_shape(self) -> Tuple[int, int]:
        """ 2-tuple: Tuple specifying the shape of the raster (defined by the spatial dimensions). """
        space_dims = list(self.space_dims.keys())
        return len(self.src_vars[space_dims[0]]), len(self.src_vars[space_dims[1]])

    def _reset(self):
        """ Resets internal class variables with properties from an existing NetCDF dataset. """
        all_dims = list(self.src.dimensions.keys())
        space_dim_names = all_dims[-2:]
        stack_dim_names = all_dims[:-2]
        self.stack_dims = dict()
        for stack_dim_name in stack_dim_names:
            is_unlimited = self.src.dimensions[stack_dim_name].isunlimited()
            self.stack_dims[stack_dim_name] = self.src.dimensions[stack_dim_name].size if not is_unlimited else None
        self.space_dims = dict()
        for space_dim_name in space_dim_names:
            self.space_dims[space_dim_name] = self.src.dimensions[space_dim_name].size

        dims = tuple(list(self.stack_dims.keys()) + list(self.space_dims.keys()))
        all_variables = [src_var.name for src_var in self.src_vars.values()]
        self.data_variables = [src_var.name for src_var in self.src_vars.values()
                               if src_var.dimensions == dims]
        for variable in all_variables:
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

    def _reset_from_ds(self, ds):
        """ Resets internal class variables with properties from an in-memory xarray dataset. """
        if len(self.data_variables) == 0:
            self.data_variables = ds.data_vars

        space_dims = list(self.space_dims.keys())
        self.space_dims[space_dims[0]] = self.space_dims[space_dims[0]] or len(ds[space_dims[0]])
        self.space_dims[space_dims[1]] = self.space_dims[space_dims[1]] or len(ds[space_dims[1]])

        all_variables = list(ds.data_vars) + list(ds.dims)
        for variable in all_variables:
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

    def _open(self):
        """ Internal method for opening a NetCDF file. """
        if self.mode == "r":
            self.src = netCDF4.Dataset(self.filepath, mode="r")
            self.src.set_auto_maskandscale(self.auto_decode)
            self.src_vars = self.src.variables
            self._reset()

            for var in self.data_variables:
                var_chunk_cache = self._var_chunk_caches[var]
                if var_chunk_cache is not None:
                    self.src_vars[var].set_var_chunk_cache(*var_chunk_cache[:3])

        if self.mode == "a":
            self.src = netCDF4.Dataset(self.filepath, mode="a")
            self.src_vars = self.src.variables
            self._reset()

        if self.mode in ["r", "a"]:
            self.gm_name = self.__get_gm_name()
            if self.gm_name is not None:
                metadata = self.get_metadata(self.src_vars[self.gm_name])
                geotrans = metadata.get('GeoTransform', None)
                if geotrans is not None:
                    self.geotrans = tuple(map(float, geotrans.split(' ')))
                self.sref_wkt = metadata.get('spatial_ref', None)

        if self.mode == "w":
            self.src = netCDF4.Dataset(self.filepath, mode="w",
                                       clobber=self.overwrite,
                                       format=self.nc_format)

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
                attr = OrderedDict([('grid_mapping_name',  self.gm_name),
                                    ('GeoTransform', geotrans)])

            crs = self.src.createVariable(self.gm_name, 'S1', ())
            crs.setncatts(attr)

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

            dims = tuple(list(self.stack_dims.keys()) + list(self.space_dims.keys()))
            for data_variable in self.data_variables:
                zlib = self._zlibs.get(data_variable)
                complevel = self._complevels[data_variable]
                chunksizes = self._chunksizes[data_variable]
                var_chunk_cache = self._var_chunk_caches[data_variable]
                nodataval = self.nodatavals[data_variable]
                dtype = self.dtypes[data_variable]

                self.src_vars[data_variable] = self.src.createVariable(
                    data_variable, dtype, dims,
                    chunksizes=chunksizes, zlib=zlib,
                    complevel=complevel, fill_value=nodataval)
                self.src_vars[data_variable].set_auto_scale(self.auto_decode)

                if var_chunk_cache is not None:
                    self.src_vars[data_variable].set_var_chunk_cache(*var_chunk_cache[:3])

                if self.gm_name is not None:
                    self.src_vars[data_variable].setncattr('grid_mapping', self.gm_name)

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

        ref_chunksize = self._chunksizes[data_variables[0]]
        chunks = dict()
        for i, stack_dim in enumerate(self.stack_dims.keys()):
            chunks[stack_dim] = ref_chunksize[i]
        space_dims = list(self.space_dims.keys())
        chunks[space_dims[0]] = ref_chunksize[-2]
        chunks[space_dims[1]] = ref_chunksize[-1]
        data_xr = xr.open_dataset(xr.backends.NetCDF4DataStore(self.src), mask_and_scale=self.auto_decode,
                                  chunks=chunks)
        data = None
        for i, data_variable in enumerate(data_variables):
            data_sliced = data_xr[data_variable][..., row: row + n_rows, col: col + n_cols]
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
        data_variables = ds.data_vars
        self._reset_from_ds(ds)

        # open file and create dimensions and coordinates
        if self.src is None:
            self._open()

        ds_idxs = []
        for stack_dim in self.stack_dims.keys():
            # determine index where to append
            if self.mode == 'a':
                append_start = self.src_vars[stack_dim].shape[0]
            else:
                append_start = 0

            # convert timestamps to index if the stack dimension is temporal
            if ds[stack_dim].dtype.name == 'datetime64[ns]':
                attr = self.attrs.get(stack_dim, dict())
                units = ds[stack_dim].attrs.get('units', attr.get('units', None))
                calendar = ds[stack_dim].attrs.get('calendar', attr.get('calendar', None))
                calendar = calendar or 'standard'
                units = units or 'days since 1900-01-01 00:00:00'
                stack_vals = netCDF4.date2num(ds[stack_dim].to_index().to_pydatetime(), units, calendar=calendar)
            else:
                stack_vals = ds[stack_dim]
            n_stack_vals = len(stack_vals)
            self.src_vars[stack_dim][append_start:append_start + n_stack_vals] = stack_vals
            ds_idxs.append(slice(append_start, None))

        space_dims = list(self.space_dims)
        n_rows, n_cols = len(ds[space_dims[0]]), len(ds[space_dims[1]])
        if space_dims[0] in ds.coords:
            self.src_vars[space_dims[0]][row:row + n_rows] = ds[space_dims[0]].data
        ds_idxs.append(slice(row, row + n_rows))
        if space_dims[1] in ds.coords:
            self.src_vars[space_dims[1]][col:col + n_cols] = ds[space_dims[1]].data
        ds_idxs.append(slice(col, col + n_cols))

        for data_variable in data_variables:
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

        self.src.setncatts(ds.attrs)
        self.src.setncatts(self.metadata)

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
        all_variables = self.data_variables + list(self.stack_dims.keys()) + list(self.space_dims.keys())
        if not isinstance(arg, dict):
            arg_dict = dict()
            for all_variable in all_variables:
                arg_dict[all_variable] = arg
        else:
            arg_dict = arg

        return arg_dict

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
        self._stack_dims = stack_dims or {'time': None}
        self._space_dims = space_dims or ['y', 'x']
        if len(self._space_dims) != 2:
            err_msg = "The number of spatial dimensions must equal 2."
            raise ValueError(err_msg)
        self.attrs = attrs or dict()

        compressions = self.__to_dict(compressions)
        chunksizes = self.__to_dict(chunksizes)
        scale_factors = self.__to_dict(scale_factors)
        offsets = self.__to_dict(offsets)
        nodatavals = self.__to_dict(nodatavals)
        dtypes = self.__to_dict(dtypes)

        self._compressions = dict()
        self._chunksizes = dict()
        self.scale_factors = dict()
        self.offsets = dict()
        self.nodatavals = dict()
        self.dtypes = dict()

        all_variables = self.data_variables + list(self._stack_dims.keys()) + self._space_dims
        for variable in all_variables:
            self._compressions[variable] = compressions.get(variable, None)
            self._chunksizes[variable] = chunksizes.get(variable, None)
            if variable in self.data_variables:
                self.scale_factors[variable] = scale_factors.get(variable, 1)
                self.offsets[variable] = offsets.get(variable, 0)
                self.nodatavals[variable] = nodatavals.get(variable, 127)
                self.dtypes[variable] = dtypes.get(variable, 'int8')

        if mode == 'r':
            self._open()

    @property
    def raster_shape(self) -> Tuple[int, int]:
        """ 2-tuple: Tuple specifying the shape of the raster (defined by the spatial dimensions). """
        return len(self.src[self._space_dims[0]]), len(self.src[self._space_dims[1]])

    def _open(self, ds=None):
        """
        Opens a NetCDF file.

        Parameters
        ----------
        ds : xarray.Dataset, optional
            Dataset used to create a new NetCDF file.

        """

        if self.mode == 'r':
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
            self.sref_wkt = self.src.rio.crs
            self.metadata = self.src.attrs
            self._reset()
        elif self.mode == 'w':
            self.src = ds
            if self.sref_wkt is not None:
                self.src.rio.write_crs(self.sref_wkt, inplace=True)
            self.src.rio.write_transform(Affine(*self.geotrans), inplace=True)
            self.src.attrs.update(self.metadata)
            self._reset()
            n_rows, n_cols = self.raster_shape
            ds[self._space_dims[0]] = self.geotrans[3] + \
                      (0.5 + np.arange(n_rows)) * self.geotrans[4] + \
                      (0.5 + np.arange(n_rows)) * self.geotrans[5]
            ds[self._space_dims[1]] = self.geotrans[0] + \
                   (0.5 + np.arange(n_cols)) * self.geotrans[1] + \
                   (0.5 + np.arange(n_cols)) * self.geotrans[2]
        else:
            err_msg = f"Mode '{self.mode}' not known."
            raise ValueError(err_msg)

    def _reset(self):
        """ Resets internal class variables with properties from an existing NetCDF dataset. """
        stack_dims = list(set(self.src.dims.keys()) - set(self._space_dims))
        if self.mode == 'r':
            self._stack_dims = dict()
            for stack_dim in stack_dims:
                self._stack_dims[stack_dim] = len(self.src[stack_dim])

        dims = stack_dims + self._space_dims
        if len(self.data_variables) == 0:
            self.data_variables = [self.src[dvar].name for dvar in self.src.data_vars
                                   if list(self.src[dvar].dims) == dims]
        all_variables = self.data_variables + dims
        for variable in all_variables:
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
        for i, data_variable in enumerate(data_variables):
            data_sliced = self.src[data_variable][..., row: row + n_rows, col: col + n_cols]
            ref_chunksize = self._chunksizes[data_variable]
            if ref_chunksize is not None:
                chunks = dict()
                for i, stack_dim in enumerate(self._stack_dims.keys()):
                    chunks[stack_dim] = ref_chunksize[i]
                chunks[self._space_dims[0]] = ref_chunksize[-2]
                chunks[self._space_dims[1]] = ref_chunksize[-1]
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
        encoding = dict()
        ds_write = self.src
        unlimited_dims = [k for k, v in self._stack_dims.items() if v is None]
        for data_variable in data_variables:
            if encoder is not None:
                ds_write[data_variable] = encoder(ds_write[data_variable], data_variable=data_variable,
                                                  nodataval=self.nodatavals[data_variable],
                                                  scale_factor=self.scale_factors[data_variable],
                                                  offset=self.offsets[data_variable],
                                                  dtype=self.dtypes[data_variable],
                                                  **encoder_kwargs)
            encoding_dv = dict()
            compression = self._compressions[data_variable]
            if compression is not None:
                encoding_dv.update(compression)
            chunksizes = self._chunksizes[data_variable]
            if chunksizes is not None:
                encoding_dv['chunksizes'] = chunksizes
            encoding[data_variable] = encoding_dv

        for dim in ds_write.dims:
            units = ds_write[dim].attrs.get('units', None)
            if units:
                encoding.update({dim.name: {'units': units}})

        # TODO: create issue about unlimited dimension encoding issue
        ds_write.to_netcdf(self.filepath, mode=self.mode, format=self.nc_format, engine=self._engine,
                           encoding=encoding, compute=compute, unlimited_dims=unlimited_dims)

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
        all_variables = self.data_variables + list(self._stack_dims.keys()) + self._space_dims
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