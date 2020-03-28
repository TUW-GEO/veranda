# Copyright (c) 2017, Vienna University of Technology (TU Wien), Department
# of Geodesy and Geoinformation (GEO).
# All rights reserved.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL VIENNA UNIVERSITY OF TECHNOLOGY,
# DEPARTMENT OF GEODESY AND GEOINFORMATION BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from collections import OrderedDict

import netCDF4
import numpy as np
import xarray as xr
from osgeo import osr

# todo: enhance reading netcdf as NetCDF4 datasets
class NcFile(object):

    """
    Wrapper for reading and writing netCDF4 files. It will create three
    predefined dimensions (time, x, y), with time as an unlimited dimension
    and x, y are defined by the shape of the data.

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
        If the optional keyword zlib is True, the data will be compressed
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
    auto_scale : bool, optional
        should the data in variables automatically be encoded?
        that means: when reading ds, "scale_factor" and "add_offset" is applied.
        ATTENTION: Also the xarray dataset may applies encoding/scaling what
                can mess up things

    """

    def __init__(self, filepath, mode='r', geotrans=(0, 1, 0, 0, 0, 1), sref=None, overwrite=True,
                 complevel=2, zlib=True, nc_format="NETCDF4_CLASSIC", chunksizes=None,
                 time_units="days since 1900-01-01 00:00:00",
                 var_chunk_cache=None, auto_scale=True, shape=None):

        self.src = None
        self.src_var = {}
        self.filepath = filepath
        self.mode = mode
        self.shape = None
        self.sref = sref
        self.geotrans = geotrans
        self.overwrite = overwrite

        self.complevel = complevel
        self.zlib = zlib
        self.gm_name = None
        self.nc_format = nc_format
        self.chunksizes = chunksizes
        self.time_units = time_units
        self.var_chunk_cache = var_chunk_cache
        self.auto_scale = auto_scale

        if self.mode in ['r', 'r_xarray', 'r_netcdf']:
            self._open()

    @property
    def metadata(self):
        return self.get_global_atts()

    def _open(self, n_rows=None, n_cols=None, auto_scale=False):
        """
        Open file.

        Parameters
        ----------
        y_coords : int, optional
            Number rows or size of y dimension.
        x_coords : int, optional
            Number of columns or size of x dimension.
        """

        if self.mode in ['r', 'r_xarray']:
            self.src = xr.open_dataset(self.filepath, mask_and_scale=auto_scale)
            self.src_var = self.src

        if self.mode == 'r_netcdf':
            self.src = netCDF4.Dataset(self.filepath, mode='r')
            self.src_var = self.src.variables

            for var in self.src_var:
                if self.var_chunk_cache is not None:
                    self.src_var[var].set_var_chunk_cache(
                        self.var_chunk_cache[0], self.var_chunk_cache[1],
                        self.var_chunk_cache[2])

        if self.mode == 'a':
            self.src = netCDF4.Dataset(self.filepath, mode='a')
            self.src_var = self.src.variables

        if self.mode in ['r', 'a', 'r_netcdf', 'r_xarray']:

            # search for grid mapping attributes in the dataset
            gm_var_name = None
            for var_name in self.src_var.keys():
                if hasattr(self.src_var[var_name], "attrs") and "grid_mapping" in self.src_var[var_name].attrs:
                    gm_var_name = var_name

            # retrieve geotransform and spatial reference parameters
            if gm_var_name is not None:
                self.gm_name = self.src_var[gm_var_name].attrs["grid_mapping"]
                if 'GeoTransform' in self.src_var[self.gm_name].attrs.keys():
                    self.geotrans = tuple(map(float, self.src_var[self.gm_name].attrs['GeoTransform'].split(' ')))
                if 'spatial_ref' in self.src_var[self.gm_name].attrs.keys():
                    self.sref = self.src_var[self.gm_name].attrs['spatial_ref']

        if self.mode == 'w':
            self.src = netCDF4.Dataset(self.filepath, mode='w',
                                       clobber=self.overwrite,
                                       format=self.nc_format)

            if self.sref is not None:
                sref = osr.SpatialReference()
                sref.ImportFromWkt(self.sref)
                proj = sref.GetAttrValue('PROJECTION')

                if proj is not None:
                    self.gm_name = proj.lower()
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
                                        ('spatial_ref', self.sref),
                                        ('GeoTransform', geotrans)])
                else:
                    self.gm_name = None
                    geotrans = "{:} {:} {:} {:} {:} {:}".format(
                        self.geotrans[0], self.geotrans[1],
                        self.geotrans[2], self.geotrans[3],
                        self.geotrans[4], self.geotrans[5])
                    attr = OrderedDict([('grid_mapping_name',  self.gm_name),
                                        ('spatial_ref', self.sref),
                                        ('GeoTransform', geotrans)])

                crs = self.src.createVariable(self.gm_name, 'S1', ())
                crs.setncatts(attr)

            if 'time' not in self.src.dimensions:
                self.src.createDimension('time', None)  # None means unlimited dimension
                if self.chunksizes is not None:
                    chunksizes = (self.chunksizes[0],)
                else:
                    chunksizes = None

                self.src_var['time'] = self.src.createVariable('time', np.float64, ('time',), chunksizes=chunksizes,
                                                               zlib=self.zlib, complevel=self.complevel)
            else:
                self.src_var['time'] = self.src['time']

            if 'y' not in self.src.dimensions and n_rows is not None:
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
                self.src_var['y'] = y
            else:
                self.src_var['y'] = self.src['y']

            if 'x' not in self.src.dimensions and n_cols is not None:
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
                self.src_var['x'] = x
            else:
                self.src_var['x'] = self.src['x']

        if hasattr(self.src, 'dims'):
            if 'time' in self.src.dims.keys():
                self.shape = (self.src.dims['time'], self.src.dims['y'], self.src.dims['x'])
            else:
                self.shape = (self.src.dims['y'], self.src.dims['x'])
        else:
            if 'time' in self.src.dimensions.keys():
                self.shape = (self.src.dimensions['time'].size, self.src.dimensions['y'].size,
                              self.src.dimensions['x'].size)
            else:
                self.shape = (self.src.dimensions['y'].size, self.src.dimensions['x'].size)

    def write(self, ds, nodatavals=None, encoder=None, encoder_kwargs=None, auto_scale=False):
        """
        Write data into netCDF4 file.

        Parameters
        ----------
        ds : xarray.Dataset
            Data set containing dims ['time', 'y', 'x'].

        """
        encoder_kwargs = {} if encoder_kwargs is None else encoder_kwargs

        bands = list(set(ds.variables.keys()) - set(['time', 'x', 'y']))
        n_bands = len(bands)

        if nodatavals is not None and not isinstance(nodatavals, list):
            nodatavals = [nodatavals] * n_bands
        elif nodatavals is None:
            nodatavals = [-9999] * n_bands  # TODO: how should the default behaviour be here?

        # open file and create dimensions and coordinates
        if self.src is None:
            self._open(n_rows=ds.dims['y'], n_cols=ds.dims['x'])

        if self.mode == 'a':
            # determine index where to append
            append_start = self.src_var['time'].shape[0]
        else:
            append_start = 0

        # fill coordinate data
        coord_names = list(ds.coords.keys())
        if 'time' in coord_names:
            dates = netCDF4.date2num(ds['time'].to_index().to_pydatetime(),
                                     self.time_units, 'standard')
            self.src_var['time'][append_start:] = dates
        if 'x' in coord_names:
            self.src_var['x'][:] = ds['x'].data
        if 'y' in coord_names:
            self.src_var['y'][:] = ds['y'].data

        n_dims = len(ds.dims)
        for i, band in enumerate(bands):
            if band not in self.src.variables:
                # check if fill value is included in attributes
                if 'fill_value' in ds[band].attrs:
                    fill_value = ds[band].attrs['fill_value']
                else:
                    fill_value = None

                self.src_var[band] = self.src.createVariable(
                    band, ds[band].dtype.name, ds[band].dims,
                    chunksizes=self.chunksizes, zlib=self.zlib,
                    complevel=self.complevel, fill_value=fill_value)
                self.src_var[band].set_auto_scale(auto_scale)

                if self.var_chunk_cache is not None:
                    self.src_var[band].set_var_chunk_cache(
                        self.var_chunk_cache[0], self.var_chunk_cache[1],
                        self.var_chunk_cache[2])

                self.src_var[band].setncatts(ds[band].attrs)
                if self.gm_name is not None:
                    self.src_var[band].setncattr('grid_mapping', self.gm_name)

            encoded_data = encoder(ds[band].data, nodataval=nodatavals[i], **encoder_kwargs) \
                if encoder is not None else ds[band].data
            if n_dims == 3:
                self.src_var[band][append_start:, :, :] = encoded_data
            elif n_dims == 2:
                self.src_var[band][:, :] = encoded_data
            else:
                err_msg = "Data is only allowed to have 2 or 3 dimensions, but it has {} dimensions.".format(n_dims)
                raise ValueError(err_msg)

    def read(self, row=None, col=None, n_rows=1, n_cols=1, bands=None, nodatavals=None, decoder=None,
             decoder_kwargs=None):
        """
        Read data from netCDF4 file.

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
        bands : str or list of str, optional
            Band numbers/names. If None, all bands will be read.
        nodatavals : tuple or list, optional
            List of no data values for each band.
            Default: -9999 for each band.
        decoder : function, optional
            Decoding function expecting a NumPy array as input.
        decoder_kwargs : dict, optional
            Keyword arguments for the decoder.

        Returns
        -------
        data : xarray.Dataset or netCDF4.variables
            Data stored in NetCDF file. Data type depends on read mode.
        """
        if nodatavals is not None and not isinstance(nodatavals, list):
            nodatavals = [nodatavals]

        decoder_kwargs = {} if decoder_kwargs is None else decoder_kwargs

        if row is None and col is None:  # read whole dataset
            row = 0
            col = 0
            n_rows = self.shape[-2]
            n_cols = self.shape[-1]
        elif row is None and col is not None:  # read by row
            row = 0
            n_cols = self.shape[-1]
        elif row is not None and col is None:  # read by column
            col = 0
            n_rows = self.shape[-2]

        if len(self.shape) == 3:
            slices = (slice(None), slice(row, row+n_rows), slice(col, col+n_cols))
        else:
            slices = (slice(row, row+n_rows), slice(col, col+n_cols))

        if bands is None:
            bands = list(self.src_var.keys())
            bands = list(set(bands) - set(['time', 'y', 'x']))

        if self.mode == "r_netcdf":  # convert to xarray if necessary
            common_chunks = self.src[bands[0]].chunking()
            chunks = dict()
            if all([common_chunks == self.src[band].chunking() for band in bands]):  # check if all chunks are the same
                for i, dim_name in enumerate(list(self.src.dimensions.keys())):
                    chunks[dim_name] = common_chunks[i]
            src_var = xr.open_dataset(xr.backends.NetCDF4DataStore(self.src), chunks=chunks)
        else:
            src_var = self.src_var

        # convert float timestamps to datetime timestamps
        if 'time' in list(src_var.dims.keys()) and src_var['time'].dtype == 'float':
            timestamps = netCDF4.num2date(src_var.variables['time'], self.time_units)
            src_var = src_var.assign_coords({'time': timestamps})

        data = None
        for i, band in enumerate(bands):
            data_ar = src_var[band][slices]
            if decoder:
                data_ar.data = decoder(data_ar.data, nodatavals[i], **decoder_kwargs)
            if data is None:
                data = data_ar.to_dataset()
            else:
                data = data.merge(data_ar.to_dataset())

        return data

    def set_global_atts(self, atts):
        """
        Set global attributes.

        Parameters
        ----------
        atts : dict
            Global attributes stored as dict.
        """
        if self.src is not None:
            self.src.setncatts(atts)

    def get_global_atts(self):
        """
        Get global attributes.

        Returns
        ----------
        atts : dict
            Global attributes stored as dict.
        """
        if self.src is not None:
            if hasattr(self.src, "attrs"):
                return self.src.attrs
            else:
                attrs = dict()
                for attr_name in self.src.ncattrs():
                    attrs[attr_name] = self.src.getncattr(attr_name)
                return attrs
        else:
            return None

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
