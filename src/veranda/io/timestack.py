# Copyright (c) 2017, Vienna University of Technology (TU Wien), Department
# of Geodesy and Geoinformation (GEO).
# All rights reserved.
#
# All information contained herein is, and remains the property of Vienna
# University of Technology (TU Wien), Department of Geodesy and Geoinformation
# (GEO). The intellectual and technical concepts contained herein are
# proprietary to Vienna University of Technology (TU Wien), Department of
# Geodesy and Geoinformation (GEO). Dissemination of this information or
# reproduction of this material is forbidden unless prior written permission
# is obtained from Vienna University of Technology (TU Wien), Department of
# Geodesy and Geoinformation (GEO).

import os
import warnings
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime

from osgeo import gdal
import numpy as np
import pandas as pd
import xarray as xr
import netCDF4

from veranda.io.geotiff import GeoTiffFile, get_pixel_coords
from veranda.io.netcdf import NcFile

gdal_dtype = {"uint8": gdal.GDT_Byte,
              "int16": gdal.GDT_Int16,
              "int32": gdal.GDT_Int32,
              "uint16": gdal.GDT_UInt16,
              "uint32": gdal.GDT_UInt32,
              "float32": gdal.GDT_Float32,
              "float64": gdal.GDT_Float64,
              "complex64": gdal.GDT_CFloat32,
              "complex128": gdal.GDT_CFloat64}

r_gdal_dtype = {gdal.GDT_Byte: "byte",
                gdal.GDT_Int16: "int16",
                gdal.GDT_Int32: "int32",
                gdal.GDT_UInt16: "uint16",
                gdal.GDT_UInt32: "uint32",
                gdal.GDT_Float32: "float32",
                gdal.GDT_Float64: "float64",
                gdal.GDT_CFloat32: "cfloat32",
                gdal.GDT_CFloat64: "cfloat64"}

DECODING_ATTR = ["scale_factor", "add_offset"]


class GeoTiffRasterTimeStack(object):

    """
    A GeoTiffRasterTimeStack is a collection of GeoTiff files, which together
    represent a time series stack of raster data.

    Parameters
    ----------
    mode : str, optional
        File stack opening mode ('r' read, 'w' write) (default 'r').
    file_list : list of str, required if mode = 'r'
        Image stack files.
    out_path : str, required if mode = 'w'
        Output file path.
    compression : str or int, optional
        Set the compression to use. LZW ('LZW') and DEFLATE (a number between
        0 and 9) compressions can be used (default 'LZW').
    blockxsize : int, optional
        Set the block size for x dimension (default: 512).
    blockysize : int, optional
        Set the block size for y dimension (default: 512).
    geotransform : tuple or list
        Geotransform parameters (default (0, 1, 0, 0, 0, 1)).
        0: Top left x
        1: W-E pixel resolution
        2: Rotation, 0 if image is "north up"
        3: Top left y
        4: Rotation, 0 if image is "north up"
        5: N-S pixel resolution (negative value if North up)
    spatialref : str
        Coordinate Reference System (CRS) in Wkt form (default None).
    tags : dict, optional
        Meta data tags (default None).
    gdal_opt : dict, optional
        Driver specific control parameters (default None).
    file_band : int, optional
        When building the VRT stack the variable defines the band to be used.
    """

    def __init__(self, mode='r', file_ts=None, out_path=None,
                 compression='LZW', blockxsize=512, blockysize=512,
                 geotransform=(0, 1, 0, 0, 0, 1), spatialref=None,
                 tags=None, gdal_opt=None, file_band=1):

        self.mode = mode
        self.file_ts = file_ts
        self.file_band = file_band
        self.vrt = None

        self.out_path = out_path

        self.compression = compression
        self.blockxsize = blockxsize
        self.blockysize = blockysize
        self.geotransform = geotransform
        self.spatialref = spatialref
        self.tags = tags
        self.gdal_opt = gdal_opt

    def _build_stack(self):
        """
        Building vrt stack.
        """
        if self.file_ts is not None:
            path = tempfile.gettempdir()
            date_str = datetime.utcnow().strftime("%Y%m%d%H%M%S")
            tmp_filename = "{:}.vrt".format(date_str)
            vrt_filename = os.path.join(path, tmp_filename)
            create_vrt_file(vrt_filename, self.file_ts['filenames'],
                            band=self.file_band)
            self.vrt = gdal.Open(vrt_filename, gdal.GA_ReadOnly)
            self.spatialref = self.vrt.GetProjection()
            self.geotransform = self.vrt.GetGeoTransform()
        else:
            raise RuntimeError('Building VRT stack failed')

    def read(self, *args, **kwargs):
        """
        Read time series or image, depending on the number of arguments.
        If 1 argument (time stamp) is provided, a time series will be
        returned and if 2 arguments (col, row) are provided, an image
        will be returned.
        """
        if len(args) == 1:
            data = self.read_img(*args, **kwargs)
        if len(args) == 2:
            data = self.read_ts(*args, **kwargs)
        if len(args) == 4:
            data = self.read_ts_bbox(*args, **kwargs)

        return data

    def read_ts_bbox(self, xmin, xmax, ymin, ymax):
        """
        Read a spatial subset time series from a raster time stack.

        Parameters
        ----------
        xmin : float
            Min X coordinate.
        xmax : float
            Max X coordinate.
        ymin : float
            Min Y coordinate.
        ymax : float
            Max Y coordinate.

        Returns
        -------
        data : numpy.ndarray
            Data set.
        """
        if self.vrt is None:
            self._build_stack()

        col, row = get_pixel_coords(
            np.array([xmin, xmax]), np.array([ymin, ymax]),
            self.geotransform[1], self.geotransform[-1],
            self.geotransform[0], self.geotransform[3])

        col_size = col[1] - col[0]
        row_size = row[0] - row[1]

        return self.read_ts(col[0], row[1], col_size, row_size)

    def read_ts(self, col, row, col_size=1, row_size=1):
        """
        Read time series from raster time stack.

        Parameters
        ----------
        col : int
            Column position.
        row : int
            Row position.
        col_size : int
            Number of columns.
        row_size : int
            Number of rows.

        Returns
        -------
        data : numpy.ndarray
            Data set.
        """
        if self.vrt is None:
            self._build_stack()

        data = self.vrt.ReadAsArray(col, row, col_size, row_size)

        return data

    def read_img(self, time_stamp, subset=None):
        """
        Read raster data set.

        Parameters
        ----------
        time_stamp : datetime
            Time stamp.
        subset : list or tuple, optional
            The subset should be in pixels, like (xmin, ymin, xmax, ymax).
        """
        if self.vrt is None:
            self._build_stack()

        try:
            band = int(self.file_ts.loc[time_stamp]['band'])
        except KeyError:
            print('Time stamp not found: {:}'.format(time_stamp))
            data = None
        else:
            if subset is None:
                data = self.vrt.GetRasterBand(band).ReadAsArray()
            else:
                data = self.vrt.GetRasterBand(band).ReadAsArray(
                    subset[0], subset[1], subset[2], subset[3])

        return data

    def iter_img(self):
        """
        Iterate over image stack.
        """
        if self.vrt is None:
            self._build_stack()

        for band, time_stamp in zip(self.file_ts['band'], self.file_ts.index):
            yield time_stamp, self.vrt.GetRasterBand(int(band)).ReadAsArray()

    def write(self, ds):
        """
        Write data set into raster time stack.

        Parameters
        ----------
        ds : xarray.Dataset
            Input data set.
        """
        time_var = ds['time'].to_index()

        if time_var.dtype == np.float64:
            time_var = pd.to_datetime(time_var, unit='D',
                                      origin=pd.Timestamp('1900-01-01'))

        time_stamps = time_var.strftime('%Y%m%d%H%M%S')
        var_dict = {}

        for name in ds.variables:
            if name == 'time':
                continue

            filenames = []
            for i, time_stamp in enumerate(time_stamps):

                fn = '{:}_{:}'.format(time_stamp, name)
                filename = os.path.join(self.out_path, '{:}.tif'.format(fn))
                filenames.append(filename)

                with GeoTiffFile(filename, mode='w', count=1) as src:
                    src.write(ds[name].isel(time=i).values, band=1)

            band = np.arange(1, len(filenames) + 1)
            var_dict[name] = pd.DataFrame({'filenames': filenames,
                                           'band': band}, index=time_var)

        return var_dict

    def close(self):
        """
        """
        if self.vrt is not None:
            self.vrt = None
            self.vrt_filename = None

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    def export_to_nc(self, path, var_name='data', **kwargs):
        """
        Export to NetCDF files.

        Parameters
        ----------
        path : str
            Output path.

        Returns
        -------
        file_list : list
            NetCDF file names.
        """
        dims = ['time', 'x', 'y']
        time_stamps = []
        filenames = []

        with NcRasterTimeStack(out_path=path, mode='w', **kwargs) as rts:
            for time_stamp, data in self.iter_img():
                coords = {'time': [time_stamp]}
                ds = xr.Dataset({var_name:
                                 (dims, data[np.newaxis, :, :])},
                                coords=coords)
                file_ts = rts.write(ds)

                time_stamps.append(time_stamp)
                filenames.extend(file_ts['filenames'].tolist())

        return pd.DataFrame({'filenames': filenames}, index=time_stamps)


def create_vrt_file(vrt_filename, file_list, nodata=None, band=1):
    """
    Create a .VRT XML file. First file is used as master file.

    Parameters
    ----------
    vrt_filename : str
        VRT filename.
    file_list : list
        List of files to include in the VRT.
    nodata : float, optional
        No data value (default: None).
    band : int, optional
        Band of the input file (default: 1)
    """

    src = gdal.Open(file_list[0])
    proj = src.GetProjection()
    geot = src.GetGeoTransform()
    xsize = src.RasterXSize
    ysize = src.RasterYSize
    dtype = src.GetRasterBand(band).DataType
    src = None

    attrib = {"rasterXSize": str(xsize), "rasterYSize": str(ysize)}
    vrt_root = ET.Element("VRTDataset", attrib=attrib)

    geot_elem = ET.SubElement(vrt_root, "GeoTransform")
    geot_elem.text = ",".join(map(str, geot))

    geot_elem = ET.SubElement(vrt_root, "SRS")
    geot_elem.text = proj

    for i, filename in enumerate(file_list):

        attrib = {"dataType": r_gdal_dtype[dtype], "band": str(i + 1)}
        band_elem = ET.SubElement(vrt_root, "VRTRasterBand", attrib=attrib)
        simsrc_elem = ET.SubElement(band_elem, "SimpleSource")
        attrib = {"relativetoVRT": "0"}
        file_elem = ET.SubElement(simsrc_elem, "SourceFilename", attrib=attrib)
        file_elem.text = filename
        ET.SubElement(simsrc_elem, "SourceBand").text = str(band)

        attrib = {"RasterXSize": str(xsize), "RasterYSize": str(ysize),
                  "DataType": r_gdal_dtype[dtype], "BlockXSize": str(512),
                  "BlockYSize": str(512)}

        file_elem = ET.SubElement(simsrc_elem, "SourceProperties",
                                  attrib=attrib)

        if nodata:
            ET.SubElement(band_elem, "NodataValue").text = str(nodata)

    tree = ET.ElementTree(vrt_root)
    tree.write(vrt_filename, encoding="UTF-8")


class NcRasterTimeStack(object):

    """
    A NcRasterTimeStack is a collection of NetCDF files, which together
    represent a time series stack of raster data.

    Parameters
    ----------
    mode : str, optional
        File stack opening mode ('r' read, 'w' write). Default: read
    file_ts : pandas.DataFrame, required for mode = 'r'
        paImage stack files.
    out_path : str, required if mode = 'w'
        Output file path.
    stack_size : str, required if mode = 'w'
        Stack size specification. Default: '%Y%W', i.e. weekly stacks
        Other possibles: '%Y' yearly stacks, '%Y%m' monthly stacks,
        '%Y%m%d' daily stacks, '%Y%m%d%H' hourly stacks, etc.
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
    compression : int, optional
        Compression level (default 2)
    chunksizes : tuple, optional
        Chunksizes of dimensions. The right definition can increase read
        operations, depending on the access pattern (e.g. time series or
        images) (default None).
    chunks : int or dict, optional
        Chunk sizes along each dimension, e.g., 5 or {'x': 5, 'y': 5}
        (default: None).
    time_units : str, optional
        Time unit of time stamps (default "days since 1900-01-01 00:00:00").
    fn_prefix : str, optional
        File name prefix (default: '').
    fn_suffix : str, optional
        File name suffix (default: '.nc').
    auto_decode : bool, optional
        If true, when reading ds, "scale_factor" and "add_offset" is applied (default is True).

    """

    def __init__(self, mode='r', file_ts=None, out_path=None,
                 stack_size='%Y%W', geotransform=(0, 1, 0, 0, 0, 1),
                 spatialref=None, compression=2, chunksizes=None,
                 chunks=None, time_units="days since 1900-01-01 00:00:00",
                 fn_prefix='', fn_suffix='.nc', auto_decode=False):

        self.mode = mode
        self.file_ts = file_ts

        self.out_path = out_path
        self.stack_size = stack_size

        self.geotransform = geotransform
        self.spatialref = spatialref
        self.compression = compression
        self.chunksizes = chunksizes
        self.chunks = chunks
        self.time_units = time_units
        self.auto_decode = auto_decode

        self.fn_prefix = fn_prefix
        self.fn_suffix = fn_suffix

        self.mfdataset = None

    def _build_stack(self):
        """
        Building file stack and initialize netCDF4.mfdataset.
        """
        if self.file_ts is not None:
            self.mfdataset = xr.open_mfdataset(self.file_ts['filenames'].tolist(), chunks=self.chunks,
                                               combine='nested', concat_dim='time', mask_and_scale=self.auto_decode)
            if self.auto_decode:
                data_var_names = list(self.mfdataset.keys())

                for var_name in data_var_names:
                    for attr in DECODING_ATTR:
                        if attr not in self.mfdataset[var_name].attrs:
                            wrn_msg = "Automatic decoding is activated for variable '{}', " \
                                      "but attribute '{}' is missing!".format(var_name, attr)
                            warnings.warn(wrn_msg)
                            break
        else:
            raise RuntimeError('Building stack failed')

    def read(self):
        """
        Read time series or image from raster time stack.

        Returns
        -------
        data : xr.Dataset
            Data set.
        """
        if self.mfdataset is None:
            self._build_stack()

        return self.mfdataset

    def iter_img(self, var_name):
        """
        Iterate over image stack.

        Parameters
        ----------
        var_name : str
            Variable name.

        Yields
        ------
        time_stamp : datetime
            Time stamp.
        data : numpy.ndarray
            2d data set.
        """
        if self.mfdataset is None:
            self._build_stack()

        ds = self.read()

        for i in range(ds[var_name].shape[0]):
            time_stamp = netCDF4.num2date(ds['time'][i].values, self.time_units)
            yield time_stamp, ds[var_name][i, :, :]

    def write(self, ds):
        """
        Write data set into raster time stack.

        Parameters
        ----------
        ds : xarray.Dataset
            Input data set.
        """
        if self.stack_size == 'single':
            fn = '{:}{:}'.format(self.fn_prefix, self.fn_suffix)
            full_filename = os.path.join(self.out_path, fn)

            if os.path.exists(full_filename):
                mode = 'a'
            else:
                mode = 'w'

            with NcFile(full_filename, mode=mode,
                        complevel=self.compression,
                        geotransform=self.geotransform,
                        spatialref=self.spatialref,
                        chunksizes=self.chunksizes) as nc:
                nc.write(ds)

            filenames = [full_filename]
        else:
            dup_stack_filenames = ds[
                'time'].to_index().strftime(self.stack_size)
            stack_filenames, index = np.unique(dup_stack_filenames,
                                               return_index=True)
            index = np.hstack((index, len(dup_stack_filenames)))

            filenames = []
            for i, filename in enumerate(stack_filenames):
                time_sel = np.arange(index[i], index[i + 1])
                fn = '{:}{:}{:}'.format(self.fn_prefix, filename,
                                        self.fn_suffix)
                full_filename = os.path.join(self.out_path, fn)
                filenames.append(full_filename)

                if os.path.exists(full_filename):
                    mode = 'a'
                else:
                    mode = 'w'

                with NcFile(full_filename, mode=mode,
                            complevel=self.compression,
                            geotransform=self.geotransform,
                            spatialref=self.spatialref,
                            chunksizes=self.chunksizes) as nc:
                    nc.write(ds.isel(time=time_sel))

        return pd.DataFrame({'filenames': filenames})

    def close(self):
        """
        Close stack.
        """
        if self.mfdataset is not None:
            self.mfdataset.close()

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    def export_to_tif(self, path, var_name, **kwargs):
        """
        Export to Geo tiff files.

        Parameters
        ----------
        path : str
            Output path.
        var_name : str
            Variable name.

        Returns
        -------
        file_list : pandas.DataFrame
            Geotiff file names with time stamps as index.
        """
        dims = ['time', 'x', 'y']
        time_stamps = []
        filenames = []

        with GeoTiffRasterTimeStack(mode='w', out_path=path) as stack:
            for time_stamp, data in self.iter_img(var_name):
                coords = {'time': [time_stamp]}
                ds = xr.Dataset(
                    {var_name: (dims, data.values[np.newaxis, :, :])},
                    coords=coords)
                file_ts = stack.write(ds)

                time_stamps.append(time_stamp)
                filenames.extend(file_ts[var_name]['filenames'].tolist())

        band = np.arange(1, len(filenames) + 1)
        file_ts = pd.DataFrame({'filenames': filenames, 'band': band},
                               index=time_stamps)
        return file_ts
