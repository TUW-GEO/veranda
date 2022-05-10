import os.path
from collections import OrderedDict

import warnings
import netCDF4
import numpy as np
from affine import Affine
import rioxarray as rio
import xarray as xr
import dask.array as da
from osgeo import osr

# todo: enhance reading netcdf as NetCDF4 datasets
DECODING_ATTR = ["scale_factor", "add_offset"]


class NetCdf4File:
    """
    Wrapper for reading and writing netCDF4 files. It will create three
    predefined dimensions (time, x, y), with time as an unlimited dimension
    and x, y are defined by the shape of the mosaic.

    The arrays to be written should have the following dimensions: time, x, y

    Parameters
    ----------
    filepath : str
        File name.
    mode : str, optional
        File opening mode. Default: 'r' = xarray.open_dataset
        Other modes:
            'r'        ... reading with xarray.open_dataset
            'r_xarray' ... reading with xarray.open_dataset
            'r_netcdf' ... reading with netCDF4.Dataset
            'w'        ... writing with netCDF4.Dataset
            'a'        ... writing with netCDF4.Dataset
    complevel : int, optional
        Compression level (default 2)
    zlib : bool, optional
        If the optional keyword zlib is True, the mosaic will be compressed
        in the netCDF file using gzip compression (default True).
    geotrans : tuple or list, optional
        Geotransform parameters (default (0, 1, 0, 0, 0, 1)).
        0: Top left x
        1: W-E pixel resolution
        2: Rotation, 0 if image is "north up"
        3: Top left y
        4: Rotation, 0 if image is "north up"
        5: N-S pixel resolution (negative value if North up)
    sref : str, optional
        Coordinate Reference System (CRS) in Wkt form (default None).
    overwrite : boolean, optional
        Flag if file can be overwritten if it already exists (default True).
    nc_format : str, optional
        NetCDF format (default 'NETCDF4_CLASSIC' (because it is
        needed for netCDF4.mfdatasets))
    chunksizes : tuple, optional
        Chunksizes of dimensions. The right definition can increase read
        operations, depending on the access pattern (e.g. time series or
        images) (default None).
    time_units : str, optional
        Time unit of time stamps (default "days since 1900-01-01 00:00:00").
    var_chunk_cache : tuple, optional
        Change variable chunk cache settings. A tuple containing
        size, nelems, preemption (default None, using default cache size)
    auto_decode : bool, optional
        If true, when reading ds, "scale_factor" and "add_offset" is applied (default is True).
        ATTENTION: Also the xarray dataset may applies encoding/scaling what
                can mess up things
    """

    def __init__(self, filepath, mode="r", data_variables=None, time_variables='time',
                 scale_factors=1, offsets=0, nodatavals=127, dtypes='int8', complevels=2, zlibs=True,
                 chunksizes=None, var_chunk_caches=None, geotrans=(0, 1, 0, 0, 0, 1), sref_wkt=None,
                 metadata=None, nc_format="NETCDF4_CLASSIC", time_units="days since 1900-01-01 00:00:00",
                 overwrite=True, auto_decode=False):

        self.src = None
        self.src_vars = {}
        self.filepath = filepath
        self.mode = mode
        self.data_variables = self.__to_iterable(data_variables) if data_variables is not None else []
        self.sref_wkt = sref_wkt
        self.geotrans = geotrans
        self.overwrite = overwrite
        self.metadata = metadata or dict()
        self.auto_decode = auto_decode
        self.gm_name = None
        self.nc_format = nc_format
        self.time_units = time_units

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
        self._scale_factors = dict()
        self._offsets = dict()
        self._nodatavals = dict()
        self._dtypes = dict()

        for data_variable in self.data_variables:
            self._zlibs[data_variable] = zlibs.get(data_variable, False)
            self._complevels[data_variable] = complevels.get(data_variable, 0)
            self._chunksizes[data_variable] = chunksizes.get(data_variable, None)
            self._var_chunk_caches[data_variable] = var_chunk_caches.get(data_variable, None)
            self._scale_factors[data_variable] = scale_factors.get(data_variable, 1)
            self._offsets[data_variable] = offsets.get(data_variable, 0)
            self._nodatavals[data_variable] = nodatavals.get(data_variable, 127)
            self._dtypes[data_variable] = dtypes.get(data_variable, 'int8')

        self._time_units = dict()
        time_variables = self.__to_iterable(time_variables)
        for time_variable in time_variables:
            self._time_units[time_variable] = time_units

        if self.mode == 'r':
            self._open()

    def _reset(self):
        self.data_variables = [src_var.name for src_var in self.src_vars.values()
                               if src_var.dimensions == ('time', 'y', 'x')]
        for data_variable in self.data_variables:
            src_var = self.src_vars[data_variable]
            src_var_md = self.__get_metadata(src_var)
            self._chunksizes[data_variable] = src_var.chunking()
            self._var_chunk_caches[data_variable] = self._var_chunk_caches.get(data_variable,
                                                                               src_var.get_var_chunk_cache())
            def_scale_factor = self._scale_factors.get(data_variable, 1)
            self._scale_factors[data_variable] = src_var_md.get('scale_factor', def_scale_factor)
            def_offset = self._offsets.get(data_variable, 0)
            self._offsets[data_variable] = src_var_md.get('add_offset', def_offset)
            def_nodataval = self._nodatavals.get(data_variable, 127)
            self._nodatavals[data_variable] = src_var_md.get('_FillValue', def_nodataval)
            self._dtypes[data_variable] = self._dtypes.get(data_variable,
                                                           src_var.dtype)

    def _reset_from_ds(self, ds):
        if len(self.data_variables) == 0:
            self.data_variables = ds.data_vars

        for data_variable in ds.data_vars:
            dar = ds[data_variable]
            src_var_md = dar.attrs
            self._zlibs[data_variable] = self._zlibs.get(data_variable, False)
            self._complevels[data_variable] = self._complevels.get(data_variable, 0)
            self._chunksizes[data_variable] = self._chunksizes.get(data_variable, dar.chunks)
            self._var_chunk_caches[data_variable] = self._var_chunk_caches.get(data_variable, None)
            def_scale_factor = self._scale_factors.get(data_variable, 1)
            self._scale_factors[data_variable] = src_var_md.get('scale_factor', def_scale_factor)
            def_offset = self._offsets.get(data_variable, 0)
            self._offsets[data_variable] = src_var_md.get('add_offset', def_offset)
            def_nodataval = self._nodatavals.get(data_variable, 127)
            self._nodatavals[data_variable] = src_var_md.get('_FillValue', def_nodataval)
            self._dtypes[data_variable] = self._dtypes.get(data_variable,
                                                           dar.dtype.name)

    def _open(self, n_rows=None, n_cols=None):
        """
        Open file.

        Parameters
        ----------
        n_rows : int, optional
            Number rows.
        n_cols : int, optional
            Number of columns.

        """
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
                if 'GeoTransform' in self.src_vars[self.gm_name].attrs.keys():
                    self.geotrans = tuple(map(float, self.src_vars[self.gm_name].attrs['GeoTransform'].split(' ')))
                if 'spatial_ref' in self.src_vars[self.gm_name].attrs.keys():
                    self.sref_wkt = self.src_vars[self.gm_name].attrs['spatial_ref']

        if self.mode == "w":
            self.src = netCDF4.Dataset(self.filepath, mode="w",
                                       clobber=self.overwrite,
                                       format=self.nc_format)

            gm_name = None
            sref = None
            if self.sref_wkt is not None:
                sref = osr.SpatialReference()
                sref.ImportFromWkt(self.sref_wkt)
                gm_name = sref.GetAttrValue('PROJECTION')

            if gm_name is not None:
                self.gm_name = gm_name.lower()
                proj4_dict = {}
                for subset in sref.ExportToProj4().split(' '):
                    x = subset.split('=')
                    if len(x) == 2:
                        proj4_dict[x[0]] = x[1]

                false_e = float(proj4_dict['+x_0'])
                false_n = float(proj4_dict['+y_0'])
                lat_po = float(proj4_dict['+lat_0'])
                lon_po = float(proj4_dict['+lon_0'])
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

            self.src.createDimension('time', None)  # None means unlimited dimension
            chunksizes = self._chunksizes.get('time', None)
            zlib = self._zlibs.get('time', True)
            complevel = self._complevels.get('time', 2)
            self.src_vars['time'] = self.src.createVariable('time', np.float64, ('time',), chunksizes=chunksizes,
                                                            zlib=zlib, complevel=complevel)
            self.src_vars['time'].units = self.time_units

            self.src.createDimension('y', n_rows)
            attr = OrderedDict([
                ('standard_name', 'projection_y_coordinate'),
                ('long_name', 'y coordinate of projection'),
                ('units', 'm')])
            y = self.src.createVariable('y', 'float64', ('y',), )
            y.setncatts(attr)
            y[:] = self.geotrans[3] + \
                       (0.5 + np.arange(n_rows)) * self.geotrans[4] + \
                       (0.5 + np.arange(n_rows)) * self.geotrans[5]
            self.src_vars['y'] = y

            self.src.createDimension('x', n_cols)
            attr = OrderedDict([
                ('standard_name', 'projection_x_coordinate'),
                ('long_name', 'x coordinate of projection'),
                ('units', 'm')])
            x = self.src.createVariable('x', 'float64', ('x',))
            x.setncatts(attr)
            x[:] = self.geotrans[0] + \
                       (0.5 + np.arange(n_cols)) * self.geotrans[1] + \
                       (0.5 + np.arange(n_cols)) * self.geotrans[2]
            self.src_vars['x'] = x

            for data_variable in self.data_variables:
                zlib = self._zlibs.get(data_variable)
                complevel = self._complevels[data_variable]
                chunksizes = self._chunksizes[data_variable]
                var_chunk_cache = self._var_chunk_caches[data_variable]
                nodataval = self._nodatavals[data_variable]
                dtype = self._dtypes[data_variable]

                self.src_vars[data_variable] = self.src.createVariable(
                    data_variable, dtype, ('time', 'y', 'x'),
                    chunksizes=chunksizes, zlib=zlib,
                    complevel=complevel, fill_value=nodataval)
                self.src_vars[data_variable].set_auto_scale(self.auto_decode)

                if var_chunk_cache is not None:
                    self.src_vars[data_variable].set_var_chunk_cache(*var_chunk_cache[:3])

                if self.gm_name is not None:
                    self.src_vars[data_variable].setncattr('grid_mapping', self.gm_name)

    def read(self, row=0, col=0, n_rows=None, n_cols=None, data_variables=None, decoder=None,
             decoder_kwargs=None):
        """
        Read mosaic from netCDF4 file.

        Parameters
        ----------
        row : int, optional
            Row number/index.
            If None and `col` is not None, then `row_size` rows with the respective column number will be loaded.
        col : int, optional
            Column number/index.
            If None and `row` is not None, then `col_size` columns with the respective row number will be loaded.
        n_rows : int, optional
            Number of rows to read (default is 1).
        n_cols : int, optional
            Number of columns to read (default is 1).
        band : str or list of str, optional
            Band numbers/names. If None, all bands will be read.
        nodataval : tuple or list, optional
            List of no mosaic values for each band.
            Default: -9999 for each band.
        decoder : function, optional
            Decoding function expecting a NumPy array as input.
        decoder_kwargs : dict, optional
            Keyword arguments for the decoder.

        Returns
        -------
        mosaic : xarray.Dataset or netCDF4.variables
            Data stored in NetCDF file. Data type depends on read mode.

        """
        decoder_kwargs = decoder_kwargs or dict()
        n_cols = len(self.src_vars['x']) if n_cols is None else n_cols
        n_rows = len(self.src_vars['y']) if n_rows is None else n_rows
        data_variables = data_variables or self.data_variables

        ref_chunksize = self._chunksizes[data_variables[0]]
        data_xr = xr.open_dataset(xr.backends.NetCDF4DataStore(self.src), mask_and_scale=self.auto_decode,
                                  chunks={'time': ref_chunksize[0], 'y': ref_chunksize[1], 'x': ref_chunksize[2]})
        data = None
        for i, data_variable in enumerate(data_variables):
            data_sliced = data_xr[data_variable][..., row: row + n_rows, col: col + n_cols]
            if decoder:
                data_sliced = decoder(data_sliced,
                                      nodataval=self._nodatavals[data_variable],
                                      data_variable=data_variable, scale_factor=self._scale_factors[data_variable],
                                      offset=self._offsets[data_variable], dtype=self._dtypes[data_variable],
                                      **decoder_kwargs)
            if data is None:
                data = data_sliced.to_dataset()
            else:
                data = data.merge(data_sliced.to_dataset())

        return data

    def write(self, ds, encoder=None, encoder_kwargs=None, **kwargs):
        """
        Write mosaic into netCDF4 file.

        Parameters
        ----------
        ds : xarray.Dataset
            Data set containing dims ['time', 'y', 'x'].
        encoder : function
            Encoding function expecting a NumPy array as input.
        encoder_kwargs : dict, optional
            Keyword arguments for the encoder.

        """
        encoder_kwargs = encoder_kwargs or dict()
        data_variables = ds.data_vars
        self._reset_from_ds(ds)

        # open file and create dimensions and coordinates
        if self.src is None:
            self._open(n_rows=ds.dims['y'], n_cols=ds.dims['x'])

        if self.mode == 'a':
            # determine index where to append
            append_start = self.src_vars['time'].shape[0]
        else:
            append_start = 0

        # fill coordinate mosaic
        if ds['time'].dtype == "<M8[ns]":  # "<M8[ns]" is numpy datetime in ns # ToDo: solve this in a better way
            timestamps = netCDF4.date2num(ds['time'].to_index().to_pydatetime(),
                                          self.time_units, 'standard')
        else:
            timestamps = ds['time']
        self.src_vars['time'][append_start:] = timestamps
        if 'x' in ds.coords:
            self.src_vars['x'][:] = ds['x'].data
        if 'y' in ds.coords:
            self.src_vars['y'][:] = ds['y'].data

        for data_variable in data_variables:
            scale_factor = self._scale_factors[data_variable]
            offset = self._offsets[data_variable]
            dtype = self._dtypes[data_variable]
            nodataval = self._nodatavals[data_variable]
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
            self.src_vars[data_variable][append_start:, :, :] = data_write
            dar_md = ds[data_variable].attrs
            dar_md.pop('_FillValue', None)  # remove this attribute because it already exists
            self.src_vars[data_variable].setncatts(dar_md)

        for key, value in ds.attrs.items():
            self.src.setncattr(key, value)

        self.src.setncatts(self.metadata)

    def __to_dict(self, arg):
        """
        Assigns non-iterable object to a dictionary with mosaic variables as keys. If `arg` is already a dictionary the
        same object is returned.

        Parameters
        ----------
        arg : non-iterable or dict
            Non-iterable, which should be converted to a dict.

        Returns
        -------
        arg_dict : dict
            Dictionary mapping mosaic variables with the value of `arg`.

        """
        if not isinstance(arg, dict):
            arg_dict = dict()
            for data_variable in self.data_variables:
                arg_dict[data_variable] = arg
        else:
            arg_dict = arg

        return arg_dict

    def __to_iterable(self, arg):
        """
        Converts non-iterable object to a list. If `arg` is already a list or tuple the same object is returned.

        Parameters
        ----------
        arg : non-iterable or list or tuple
            Non-iterable, which should be converted to a list.

        Returns
        -------
        arg_list : list or tuple
            List containing `arg` as a value.

        """
        if not isinstance(arg, (list, tuple)):
            arg_list = [arg]
        else:
            arg_list = arg
        return arg_list

    def __get_metadata(self, src_var):
        attrs = dict()
        attrs_keys = src_var.ncattrs()
        for key in attrs_keys:
            attrs[key] = src_var.getncattr(key)
        return attrs

    def __get_gm_name(self):
        gm_var_name = None
        for var_name in self.src_vars.keys():
            if hasattr(self.src_vars[var_name], "attrs") and "grid_mapping" in self.src_vars[var_name].attrs:
                gm_var_name = var_name

        gm_name = None
        if gm_var_name is not None:
            gm_name = self.src_vars[gm_var_name].attrs["grid_mapping"]

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
    Wrapper for reading and writing netCDF4 files with xarray. It will create three
    predefined dimensions (time, x, y), with time as an unlimited dimension
    and x, y are defined by the shape of the mosaic.

    The arrays to be written should have the following dimensions: time, x, y

    Parameters
    ----------
    filepath : str
        File name.
    mode : str, optional
        File opening mode. Default: 'r' = xarray.open_dataset
        Other modes:
            'r'        ... reading with xarray.open_dataset
            'r_xarray' ... reading with xarray.open_dataset
            'r_netcdf' ... reading with netCDF4.Dataset
            'w'        ... writing with netCDF4.Dataset
            'a'        ... writing with netCDF4.Dataset
    complevel : int, optional
        Compression level (default 2)
    zlib : bool, optional
        If the optional keyword zlib is True, the mosaic will be compressed
        in the netCDF file using gzip compression (default True).
    geotrans : tuple or list, optional
        Geotransform parameters (default (0, 1, 0, 0, 0, 1)).
        0: Top left x
        1: W-E pixel resolution
        2: Rotation, 0 if image is "north up"
        3: Top left y
        4: Rotation, 0 if image is "north up"
        5: N-S pixel resolution (negative value if North up)
    sref : str, optional
        Coordinate Reference System (CRS) in Wkt form (default None).
    overwrite : boolean, optional
        Flag if file can be overwritten if it already exists (default True).
    nc_format : str, optional
        NetCDF format (default 'NETCDF4_CLASSIC' (because it is
        needed for netCDF4.mfdatasets))
    chunksizes : tuple, optional
        Chunksizes of dimensions. The right definition can increase read
        operations, depending on the access pattern (e.g. time series or
        images) (default None).
    time_units : str, optional
        Time unit of time stamps (default "days since 1900-01-01 00:00:00").
    var_chunk_cache : tuple, optional
        Change variable chunk cache settings. A tuple containing
        size, nelems, preemption (default None, using default cache size)
    auto_decode : bool, optional
        If true, when reading ds, "scale_factor" and "add_offset" is applied (default is True).
        ATTENTION: Also the xarray dataset may applies encoding/scaling what
                can mess up things
    """

    def __init__(self, filepath, mode='r', data_variables=None, time_variables='time',
                 scale_factors=1, offsets=0, nodatavals=127, dtypes='int8', compression=None,
                 chunksizes=None, geotrans=(0, 1, 0, 0, 0, 1), sref_wkt=None,
                 time_units="days since 1900-01-01 00:00:00", nc_format="NETCDF4_CLASSIC",
                 metadata=None, overwrite=True, auto_decode=False, engine='netcdf4'):

        self.src = None
        self.filepath = filepath
        self.mode = mode
        self.data_variables = self.__to_iterable(data_variables) if data_variables is not None else []
        self.sref_wkt = sref_wkt
        self.geotrans = geotrans
        self.metadata = metadata or dict()
        self.overwrite = overwrite
        self._engine = engine
        self.nc_format = nc_format
        self.auto_decode = auto_decode

        compressions = self.__to_dict(compression)
        chunksizes = self.__to_dict(chunksizes)
        scale_factors = self.__to_dict(scale_factors)
        offsets = self.__to_dict(offsets)
        nodatavals = self.__to_dict(nodatavals)
        dtypes = self.__to_dict(dtypes)

        self._compressions = dict()
        self._chunksizes = dict()
        self._scale_factors = dict()
        self._offsets = dict()
        self._nodatavals = dict()
        self._dtypes = dict()

        for data_variable in self.data_variables:
            self._compressions[data_variable] = compressions.get(data_variable, None)
            self._chunksizes[data_variable] = chunksizes.get(data_variable, None)
            self._scale_factors[data_variable] = scale_factors.get(data_variable, 1)
            self._offsets[data_variable] = offsets.get(data_variable, 0)
            self._nodatavals[data_variable] = nodatavals.get(data_variable, 127)
            self._dtypes[data_variable] = dtypes.get(data_variable, 'int8')

        self._time_units = dict()
        time_variables = self.__to_iterable(time_variables)
        for time_variable in time_variables:
            self._time_units[time_variable] = time_units

        if mode == 'r':
            self._open()

    def _open(self, ds=None):
        """
        Open file.

        Parameters
        ----------
        ds :


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
            n_rows, n_cols = len(ds['y']), len(ds['x'])
            ds['x'] = self.geotrans[0] + \
                   (0.5 + np.arange(n_cols)) * self.geotrans[1] + \
                   (0.5 + np.arange(n_cols)) * self.geotrans[2]
            ds['y'] = self.geotrans[3] + \
                      (0.5 + np.arange(n_rows)) * self.geotrans[4] + \
                      (0.5 + np.arange(n_rows)) * self.geotrans[5]
        else:
            err_msg = f"Mode '{self.mode}' not known."
            raise ValueError(err_msg)

    def _reset(self):
        if len(self.data_variables) == 0:
            self.data_variables = [self.src[dvar].name for dvar in self.src.data_vars
                                   if self.src[dvar].dims == ('time', 'y', 'x')]
        for data_variable in self.data_variables:
            ref_scale_factor = self._scale_factors.get(data_variable, 1)
            self._scale_factors[data_variable] = self.src[data_variable].attrs.get('scale_factor', ref_scale_factor)
            ref_offset = self._offsets.get(data_variable, 0)
            self._offsets[data_variable] = self.src[data_variable].attrs.get('add_offset', ref_offset)
            ref_nodataval = self._nodatavals.get(data_variable, 127)
            self._nodatavals[data_variable] = self.src[data_variable].attrs.get('_FillValue', ref_nodataval)
            ref_dtype = self._dtypes.get(data_variable, self.src[data_variable].dtype.name)
            self._dtypes[data_variable] = ref_dtype
            self._compressions[data_variable] = self._compressions.get(data_variable, None)
            ref_chunksizes = self.src[data_variable].encoding.get('chunksizes', None) # TODO fix this workaround
            self._chunksizes[data_variable] = self._chunksizes.get(data_variable, ref_chunksizes)

    def read(self, row=0, col=0, n_rows=None, n_cols=None, data_variables=None, decoder=None,
             decoder_kwargs=None):
        """
        Read mosaic from netCDF4 file.

        Parameters
        ----------
        row : int, optional
            Row number/index.
            If None and `col` is not None, then `row_size` rows with the respective column number will be loaded.
        col : int, optional
            Column number/index.
            If None and `row` is not None, then `col_size` columns with the respective row number will be loaded.
        n_rows : int, optional
            Number of rows to read (default is 1).
        n_cols : int, optional
            Number of columns to read (default is 1).
        data_variables : str or list of str, optional
            Data variable names. If None, all bands will be read.
        decoder : function, optional
            Decoding function expecting a NumPy array as input.
        decoder_kwargs : dict, optional
            Keyword arguments for the decoder.

        Returns
        -------
        mosaic : xarray.Dataset or netCDF4.variables
            Data stored in NetCDF file. Data type depends on read mode.

        """
        decoder_kwargs = decoder_kwargs or dict()
        n_cols = len(self.src['x']) if n_cols is None else n_cols
        n_rows = len(self.src['y']) if n_rows is None else n_rows
        data_variables = data_variables or self.data_variables

        data = None
        for i, data_variable in enumerate(data_variables):
            data_sliced = self.src[data_variable][..., row: row + n_rows, col: col + n_cols]
            chunksizes = self._chunksizes[data_variable]
            if chunksizes is not None:
                data_sliced = data_sliced.chunk({'time': chunksizes[0],
                                                 'y': chunksizes[1],
                                                 'x': chunksizes[2]})
            if decoder:
                data_sliced = decoder(data_sliced,
                                      nodataval=self._nodatavals[data_variable],
                                      data_variable=data_variable, scale_factor=self._scale_factors[data_variable],
                                      offset=self._offsets[data_variable], dtype=self._dtypes[data_variable],
                                      **decoder_kwargs)
            if data is None:
                data = data_sliced.to_dataset()
            else:
                data = data.merge(data_sliced.to_dataset())

        return data

    def write(self, ds, data_variables=None, encoder=None, encoder_kwargs=None, compute=True, unlimited_dims=None,
              **kwargs):
        """
        Write mosaic into a netCDF4 file.

        Parameters
        ----------
        ds : xarray.Dataset
            Data set containing dims ['time', 'y', 'x'].
        encoder : function
            Encoding function expecting a NumPy array as input.
        encoder_kwargs : dict, optional
            Keyword arguments for the encoder.

        """
        self._open(ds)
        data_variables = data_variables or self.data_variables
        encoder_kwargs = encoder_kwargs or dict()
        encoding = dict()
        ds_write = self.src
        for data_variable in data_variables:
            if encoder is not None:
                ds_write[data_variable] = encoder(ds_write[data_variable], data_variable=data_variable,
                                                  nodataval=self._nodatavals[data_variable],
                                                  scale_factor=self._scale_factors[data_variable],
                                                  offset=self._offsets[data_variable],
                                                  dtype=self._dtypes[data_variable],
                                                  **encoder_kwargs)
            encoding_dv = dict()
            compression = self._compressions[data_variable]
            if compression is not None:
                encoding_dv.update(compression)
            chunksizes = self._chunksizes[data_variable]
            if chunksizes is not None:
                encoding_dv['chunksizes'] = chunksizes
            encoding[data_variable] = encoding_dv

        for k, v in self._time_units.items():
            encoding.update({k: {'units': v}})
        ds_write.to_netcdf(self.filepath, mode=self.mode, format=self.nc_format, engine=self._engine,
                           encoding=encoding, compute=compute, unlimited_dims=unlimited_dims)

    def __to_dict(self, arg):
        """
        Assigns non-iterable object to a dictionary with mosaic variables as keys. If `arg` is already a dictionary the
        same object is returned.

        Parameters
        ----------
        arg : non-iterable or dict
            Non-iterable, which should be converted to a dict.

        Returns
        -------
        arg_dict : dict
            Dictionary mapping mosaic variables with the value of `arg`.

        """
        if not isinstance(arg, dict) or (isinstance(arg, dict) and not isinstance(arg[list(arg.keys())[0]], dict)):
            arg_dict = dict()
            for data_variable in self.data_variables:
                arg_dict[data_variable] = arg
        else:
            arg_dict = arg

        return arg_dict

    def __to_iterable(self, arg):
        """
        Converts non-iterable object to a list. If `arg` is already a list or tuple the same object is returned.

        Parameters
        ----------
        arg : non-iterable or list or tuple
            Non-iterable, which should be converted to a list.

        Returns
        -------
        arg_list : list or tuple
            List containing `arg` as a value.

        """
        if not isinstance(arg, (list, tuple)):
            arg_list = [arg]
        else:
            arg_list = arg
        return arg_list

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