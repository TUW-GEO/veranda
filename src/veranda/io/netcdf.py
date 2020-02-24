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


class NcFile(object):

    """
    Wrapper for reading and writing netCDF4 files. It will create three
    predefined dimensions (time, x, y), with time as an unlimited dimension
    and x, y are defined by the shape of the data.

    The arrays to be written should have the following dimensions: time, x, y

    Parameters
    ----------
    filename : str
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
    geotransform : tuple or list, optional
        Geotransform parameters (default (0, 1, 0, 0, 0, 1)).
        0: Top left x
        1: W-E pixel resolution
        2: Rotation, 0 if image is "north up"
        3: Top left y
        4: Rotation, 0 if image is "north up"
        5: N-S pixel resolution (negative value if North up)
    spatialref : str, optional
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

    def __init__(self, filename, mode='r', complevel=2, zlib=True,
                 geotransform=(0, 1, 0, 0, 0, 1), spatialref=None,
                 overwrite=True, nc_format="NETCDF4_CLASSIC", chunksizes=None,
                 time_units="days since 1900-01-01 00:00:00",
                 var_chunk_cache=None, auto_scale=True, shape=None, data_var_name=None):

        self.filename = filename
        self.mode = mode
        self.src = None
        self.src_var = {}
        self.complevel = complevel
        self.zlib = zlib

        self.spatialref = spatialref
        self.geotransform = geotransform
        self.gmn = None

        self.overwrite = overwrite
        self.nc_format = nc_format
        self.shape = shape
        self.chunksizes = chunksizes
        self.time_units = time_units
        self.var_chunk_cache = var_chunk_cache
        self.auto_scale = auto_scale

        if self.mode in ['r', 'r_xarray', 'r_netcdf']:
            self._open(data_var_name=data_var_name)

    @property
    def metadata(self):
        if self.src is not None:
            return self.src.ncattrs()
        else:
            return None

    def _open(self, n_x=None, n_y=None, data_var_name=None):
        """
        Open file.

        Parameters
        ----------
        n_x : int, optional
            Size of x dimension.
        n_y : int, optional
            Size of y dimension.
        data_var_name: str, optional
            Data variable name of netCDF4/xarray object.
        """

        if self.mode in ['r', 'r_xarray']:
            self.src = xr.open_dataset(
                self.filename, mask_and_scale=self.auto_scale)
            if 'time' in list(self.src.dims.keys()) and self.src.variables['time'].dtype == 'float':
                timestamps = netCDF4.num2date(self.src.variables['time'], self.time_units)
                self.src = self.src.assign_coords({'time': timestamps})
            self.src_var = self.src

        if self.mode == 'r_netcdf':
            self.src = netCDF4.Dataset(self.filename, mode='r')
            self.src_var = self.src.variables
            if 'time' in list(self.src.dimensions.keys()) and self.src.variables['time'].dtype == 'float':
                timestamps = netCDF4.num2date(np.array(self.src.variables['time']), self.time_units)
                self.src.variables['time'] = timestamps
            self.src_var = self.src.variables

            for var in self.src_var:
                if self.var_chunk_cache is not None:
                    self.src_var[var].set_var_chunk_cache(
                        self.var_chunk_cache[0], self.var_chunk_cache[1],
                        self.var_chunk_cache[2])

        if self.mode == 'a':
            self.src = netCDF4.Dataset(self.filename, mode=self.mode)
            self.src_var = self.src.variables

        if self.mode in ['r', 'a', 'r_netcdf', 'r_xarray']:
            if data_var_name is None:
                for var_name in self.src_var.keys():
                    if "grid_mapping" in self.src_var[var_name].attrs:
                        data_var_name = var_name

            if data_var_name is not None and "grid_mapping" in self.src_var[data_var_name].attrs:
                gmn = self.src_var[data_var_name].attrs["grid_mapping"]
                self.geotransform = tuple(map(float, self.src_var[gmn].attrs['GeoTransform'].split(' '))) \
                    if 'GeoTransform' in self.src_var[gmn].attrs.keys() else None
                self.spatialref = self.src_var[gmn].attrs['spatial_ref'] \
                    if 'spatial_ref' in self.src_var[gmn].attrs.keys() else None

        if self.mode == 'w':
            self.src = netCDF4.Dataset(self.filename, mode=self.mode,
                                       clobber=self.overwrite,
                                       format=self.nc_format)

            if self.spatialref is not None:
                spref = osr.SpatialReference()
                spref.ImportFromWkt(self.spatialref)
                proj = spref.GetAttrValue('PROJECTION')

                if proj is not None:
                    self.gmn = proj.lower()
                    proj4_dict = {}
                    for subset in spref.ExportToProj4().split(' '):
                        x = subset.split('=')
                        if len(x) == 2:
                            proj4_dict[x[0]] = x[1]

                    false_e = float(proj4_dict['+x_0'])
                    false_n = float(proj4_dict['+y_0'])
                    lat_po = float(proj4_dict['+lat_0'])
                    lon_po = float(proj4_dict['+lon_0'])
                    long_name = 'CRS definition'
                    # lon_pm = 0.
                    semi_major_axis = spref.GetSemiMajor()
                    inverse_flattening = spref.GetInvFlattening()
                    spatial_ref = self.spatialref
                    geotransform = "{:} {:} {:} {:} {:} {:}".format(
                        self.geotransform[0], self.geotransform[1],
                        self.geotransform[2], self.geotransform[3],
                        self.geotransform[4], self.geotransform[5])

                    attr = OrderedDict([('grid_mapping_name', self.gmn),
                                        ('false_easting', false_e),
                                        ('false_northing', false_n),
                                        ('latitude_of_projection_origin', lat_po),
                                        ('longitude_of_projection_origin', lon_po),
                                        ('long_name', long_name),
                                        ('semi_major_axis', semi_major_axis),
                                        ('inverse_flattening', inverse_flattening),
                                        ('spatial_ref', spatial_ref),
                                        ('GeoTransform', geotransform)])
                else:
                    self.gmn = 'proj_unknown'
                    spatial_ref = self.spatialref
                    geotransform = "{:} {:} {:} {:} {:} {:}".format(
                        self.geotransform[0], self.geotransform[1],
                        self.geotransform[2], self.geotransform[3],
                        self.geotransform[4], self.geotransform[5])
                    attr = OrderedDict([('grid_mapping_name', 'None'),
                                        ('spatial_ref', spatial_ref),
                                        ('GeoTransform', geotransform)])

                crs = self.src.createVariable(self.gmn, 'S1', ())
                crs.setncatts(attr)

            if 'time' not in self.src.dimensions:
                self.src.createDimension('time', None)

            if 'x' not in self.src.dimensions and n_x is not None:
                self.src.createDimension('x', n_x)
                attr = OrderedDict([
                    ('standard_name', 'projection_x_coordinate'),
                    ('long_name', 'x coordinate of projection'),
                    ('units', 'm')])
                x = self.src.createVariable('x', 'float64', ('x',))
                x.setncatts(attr)
                x[:] = self.geotransform[0] + \
                        (0.5 + np.arange(n_y)) * self.geotransform[1] + \
                        (0.5 + np.arange(n_y)) * self.geotransform[2]

            if 'y' not in self.src.dimensions and n_y is not None:

                self.src.createDimension('y', n_y)
                attr = OrderedDict([
                    ('standard_name', 'projection_y_coordinate'),
                    ('long_name', 'y coordinate of projection'),
                    ('units', 'm')])
                y = self.src.createVariable('y', 'float64', ('y',), )
                y.setncatts(attr)
                y[:] = self.geotransform[3] + \
                        (0.5 + np.arange(n_x)) * self.geotransform[4] + \
                        (0.5 + np.arange(n_x)) * self.geotransform[5]

        if hasattr(self.src, 'dims'):
            self.shape = (self.src.dims['y'], self.src.dims['x'])
        else:
            self.shape = (self.src.dimensions['y'].size, self.src.dimensions['x'].size)

    # TODO: add x and y dimensions and check updated implementation!
    def write(self, ds):
        """
        Write data into netCDF4 file.

        Parameters
        ----------
        ds : xarray.Dataset
            Data set containing dims ['time', 'x', 'y'].

        """
        # open file and create dimensions
        if self.src is None:
            n_x = None if 'x' not in ds.dims.keys() else ds.dims['x']
            n_y = None if 'y' not in ds.dims.keys() else ds.dims['y']
            self._open(n_x=n_x, n_y=n_y)

        # determine index where to append
        if self.src_var:
            append_start = self.src_var['time'].shape[0]
        else:
            append_start = 0

        # create variables
        for k in ds.variables:
            if not k in self.src.variables:
                if k == 'time':
                    if self.chunksizes is not None:
                        chunksizes = (self.chunksizes[0], )
                    else:
                        chunksizes = None

                    self.src_var[k] = self.src.createVariable(
                        k, np.float64, ds[k].dims, chunksizes=chunksizes,
                        zlib=self.zlib, complevel=self.complevel)
                else:
                    # check if fill value is included in attributes
                    if 'fill_value' in ds[k].attrs:
                        fill_value = ds[k].attrs['fill_value']
                    else:
                        fill_value = None

                    self.src_var[k] = self.src.createVariable(
                        k, ds[k].dtype.name, ds[k].dims,
                        chunksizes=self.chunksizes, zlib=self.zlib,
                        complevel=self.complevel, fill_value=fill_value)
                    self.src_var[k].set_auto_scale(self.auto_scale)

                    if self.var_chunk_cache is not None:
                        self.src_var[k].set_var_chunk_cache(
                            self.var_chunk_cache[0], self.var_chunk_cache[1],
                            self.var_chunk_cache[2])

                    self.src_var[k].setncatts(ds[k].attrs)
                    if self.gmn is not None:
                        self.src_var[k].setncattr('grid_mapping', self.gmn)

            if k not in self.src_var and k in self.src.variables:
                self.src_var[k] = self.src[k]

        # fill variables with data
        for k in ds.variables:
            if k == 'time':
                dates = netCDF4.date2num(ds[k].to_index().to_pydatetime(),
                                         self.time_units, 'standard')
                self.src_var[k][append_start:] = dates
            elif k in ['x', 'y']:
                self.src_var[k][:] = ds[k].data
            else:
                if len(ds.dims) == 3:
                    self.src_var[k][append_start:, :, :] = ds[k].data
                elif len(ds.dims) == 2:
                    self.src_var[k][:, :] = ds[k].data

    def read(self, col, row, col_size=1, row_size=1, band=None):
        """
        Read data from netCDF4 file.

        Returns
        -------
        data : xarray.Dataset or netCDF4.variables
            Data stored in NetCDF file. Data type depends on read mode.
        """
        # TODO: check band/data variable availability
        if col is None and row is None:
            if band is not None:
                return self.src_var[band].to_dataset()
            else:
                return self.src_var
        else:
            if band is not None:
                return self.src_var[band][row:(row + row_size), col:(col + col_size)].to_dataset()
            else:
                # TODO: what to do here?
                return self.src_var


    def decode_xarr_var(self, var):
        """
        Decodes the values in the xarray variables.

        BBM: If you find a better way to (safely!) control the decoding, please let me know.

        Parameters
        ----------
        var : str
            name of the variable in the xarray-Dataset

        Returns
        -------
        data
            decoded values of "var",
            # with applied fill_value
            # with applied scaling: data * scale_factor + add_offset

        """
        data = np.float32(self.src_var[var].data)

        # apply filling value
        try:
            fv = self.src_var[var].fill_value
            data[data == fv] = np.nan
        except:
            pass

        # get offset
        try:
            os = self.src_var[var].add_offset
        except:
            os = 0.0

        # apply scaling
        try:
            sf = self.src_var[var].scale_factor
            data = data * sf + os
        except:
            pass

        return data

    def set_global_atts(self, atts):
        """
        Set global attributes.

        Parameters
        ----------
        atts : dict
            Global attributes stored as dict.
        """
        self.src.setncatts(atts)

    def get_global_atts(self):
        """
        Get global attributes.

        Returns
        ----------
        atts : dict
            Global attributes stored as dict.
        """
        if hasattr(self.src, "attrs"):
            return self.src.attrs
        else:
            attrs = dict()
            for attr_name in self.src.ncattrs():
                attrs[attr_name] = self.src.getncattr(attr_name)
            return attrs

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
