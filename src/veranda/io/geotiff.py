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


"""
Read and write Geotiff files.
"""

import os
import math

import warnings
import numpy as np
from osgeo import gdal
from osgeo import gdal_array

_numpy2gdal_dtype = {"bool": 1,
                     "uint8": 1,
                     "int8": 1,
                     "uint16": 2,
                     "int16": 3,
                     "uint32": 4,
                     "int32": 5,
                     "float32": 6,
                     "float64": 7,
                     "complex64": 10,
                     "complex128": 11}

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


def read_tiff(src_file, sub_rect=None):
    """
    Reads a simple 1-band geotiff file and as output returns the stored raster
    as an array together with some tags-information as dictionary (tags_dict)

    CAUTION: the returned geotags corresponds to input file and not the
             returned src_arr (in case sub_rect is set)

    Parameters
    ----------
    src_file : string
        The full path of source dataset.
    sub_rect : list (optional)
        Set this keyword to a four-element array, [Xoffset, Yoffset, width,
        height], that specifies a rectangular region within the input raster
        to extract.

    Returns
    -------
    src_arr : numpy.ndarray
        Data stored in geotiff.
    tags_dict: dict
        Tags stored in geotiff.
    """
    src_data = gdal.Open(src_file)
    driver = src_data.GetDriver()

    if driver.ShortName != 'GTiff':
        raise IOError("input file is not a tiff file")

    # Fetch the number of raster bands on this dataset.
    raster_count = src_data.RasterCount

    if raster_count != 1:
        raise IOError("read_tiff can only handle 1-band tif files")

    src_band = src_data.GetRasterBand(int(1))
    no_data_val = src_band.GetNoDataValue()
    data_type = gdal_array.GDALTypeCodeToNumericTypeCode(src_band.DataType)

    # get parameters
    description = src_data.GetDescription()
    metadata = src_data.GetMetadata()

    # Fetch the affine transformation coefficients.
    geotransform = src_data.GetGeoTransform()
    spatialreference = src_data.GetProjection()
    gcps = src_data.GetGCPs()

    tags_dict = {'description': description,
                 'metadata': metadata,
                 'geotransform': geotransform,
                 'spatialreference': spatialreference,
                 'gcps': gcps,
                 'no_data_val': no_data_val,
                 'datatype': data_type,
                 'blockxsize': src_band.GetBlockSize()[0],
                 'blockysize': src_band.GetBlockSize()[1]}

    if sub_rect is None:
        src_arr = src_data.ReadAsArray()
    else:
        src_arr = src_data.ReadAsArray(
            sub_rect[0], sub_rect[1], sub_rect[2], sub_rect[3])

    return src_arr, tags_dict


def write_tiff(dst_file, src_arr=None, red=None, green=None, blue=None,
               tags_dict=None, tilesize=512, ct=None):
    """
    Write a 2D numpy array as a single band tiff file with tags.

    Parameters
    ----------
    dst_file : str
        The full path of output file.
    src_arr : numpy.ndarray (2d), optional
        The input image array to be written. It will be ignored if
        red, green, blue keywords are set.
    red : numpy.ndarray, optional
        If you are writing a palette color image, set these keywords equal
        to the color table vectors, scaled from 0 to 255.
    green : numpy.ndarray, optional
        If you are writing a palette color image, set these keywords equal
        to the color table vectors, scaled from 0 to 255.
    blue : numpy.ndarray, optional
        If you are writing a palette color image, set these keywords equal
        to the color table vectors, scaled from 0 to 255.
    tags_dict : dict, optional
        The tags need to be written in the tiff file.
    tilesize : int, optional
        The tile size of the tiled geotiff. Default: 512.
        None means no tiling.
    ct : gdal colortable
        If available then the colortable will be attached to geotiff file
    """
    if red is not None and green is not None and blue is not None:
        src_arr = red
        nband = 3
    else:
        nband = 1
        if src_arr is None:
            raise IOError("input data are None")

    # get gdal data type from numpy data type format, dtype is set according
    # to the src_arr (or red band) dtype
    gdal_dtype = _numpy2gdal_dtype[str(src_arr.dtype)]

    if src_arr.ndim != 2:
        raise IOError('the input data should be 2d')

    ncol = src_arr.shape[1]
    nrow = src_arr.shape[0]

    # geotiff driver
    opt = ["COMPRESS=LZW"]
    if tilesize:
        tilesize = int(tilesize)
        # make sure the tilesize is exponent of 2
        tilesize = 2 ** int(round(math.log(tilesize, 2)))
        opt.append("TILED=YES")
        opt.append("BLOCKXSIZE={:d}".format(tilesize))
        opt.append("BLOCKYSIZE={:d}".format(tilesize))

    driver = gdal.GetDriverByName('GTiff')
    dst_data = driver.Create(dst_file, ncol, nrow, nband, gdal_dtype, opt)

    # attach tags
    if tags_dict != None:
        if 'description' in tags_dict.keys():
            if tags_dict['description'] != None:
                dst_data.SetDescription(tags_dict['description'])
        if 'metadata' in tags_dict.keys():
            if tags_dict['metadata'] != None:
                dst_data.SetMetadata(tags_dict['metadata'])
        if 'no_data_val' in tags_dict.keys():
            if tags_dict['no_data_val'] != None:
                dst_data.GetRasterBand(1).SetNoDataValue(
                    float(tags_dict['no_data_val']))
        if 'geotransform' in tags_dict.keys():
            if tags_dict['geotransform'] != None:
                dst_data.SetGeoTransform(tags_dict['geotransform'])
        if 'spatialreference' in tags_dict.keys():
            if tags_dict['spatialreference'] != None:
                dst_data.SetProjection(tags_dict['spatialreference'])
        if 'gcps' in tags_dict.keys():
            if tags_dict['gcps'] != None:
                if len(tags_dict['gcps']) != 0:
                    dst_data.SetGCPs(tags_dict['gcps'],
                                     tags_dict['spatialreference'])

    # set color table
    if ct is not None:
        dst_data.GetRasterBand(1).SetRasterColorTable(ct)

    dst_data.GetRasterBand(1).WriteArray(src_arr)
    if nband == 3:
        dst_data.GetRasterBand(2).WriteArray(green)
        dst_data.GetRasterBand(3).WriteArray(blue)

    dst_data.FlushCache()

    # gdal lacks a close function, needs to be None
    dst_data = None


class GeoTiffFile(object):
    """
    GDAL wrapper for reading and writing raster files. A tiled (not stripped)
    GeoTiff file is created.

    Parameters
    ----------
    filepath : str
        File name.
    mode : str, optional
        File opening mode ('r' read, 'w' write). Default: read
    count : int, required if data is 2d
        Number of bands. Default: Not defined and raster data set
        assumed to be 3d [band, ncol, nrow], but if data is written for
        each band separately count needs to be set to the number of bands.
    compression : str or int, optional
        Set the compression to use. LZW ('LZW') and DEFLATE (a number between
        0 and 9) compressions can be used (default 'LZW').
    blockxsize : int, optional
        Set the block size for x dimension (default: 512).
    blockysize : int, optional
        Set the block size for y dimension (default: 512).
    geotrans : tuple or list, optional
        Geotransform parameters (default (0, 1, 0, 0, 0, 1)).
        0: Top left x
        1: W-E pixel resolution
        2: Rotation, 0 if image is "north up"
        3: Top left y
        4: Rotation, 0 if image is "north up"
        5: N-S pixel resolution (negative value if North up)
    sref : str, optional
        Coordinate Reference System (CRS) in Wkt form (default: None).
    tags : dict, optional
        Meta data tags (default: None).
    overwrite : boolean, optional
        Flag if file can be overwritten if it already exists (default: True).
    gdal_opt : dict, optional
        Driver specific control parameters (default: None).
    """

    def __init__(self, filepath, mode='r', compression='LZW',
                 blockxsize=512, blockysize=512, geotrans=(0, 1, 0, 0, 0, 1),
                 n_bands=None, sref=None, tags=None,
                 overwrite=True, gdal_opt=None, auto_decode=False):

        # mandatory variables for a RasterIOBase object
        self.src = None
        self.filepath = filepath
        self.mode = mode
        self.shape = None
        self.sref = sref
        self.geotrans = geotrans
        self.metadata = tags['metadata'] if tags is not None and 'metadata' in tags.keys() else None
        self.overwrite = overwrite
        self.auto_decode = auto_decode

        self.driver = gdal.GetDriverByName('GTiff')
        self.tags = tags
        self.n_bands = n_bands
        self.dtype = None
        self.chunks = None

        if self.mode == 'w':
            gdal_opt_default = {'COMPRESS': 'LZW', 'TILED': 'YES',
                                'BLOCKXSIZE': str(blockxsize),
                                'BLOCKYSIZE': str(blockysize)}

            if gdal_opt is None:
                gdal_opt = gdal_opt_default
            else:
                gdal_opt_default.update(gdal_opt)

            if not isinstance(compression, str):
                gdal_opt['COMPRESS'] = 'DEFLATE'
                gdal_opt['ZLEVEL'] = str(compression)

            self.gdal_opt = ['='.join((k, v)) for k, v in gdal_opt.items()]
        else:
            self.gdal_opt = []

        if self.mode == 'r':
            self._open()

    def _open(self, n_bands=None, n_rows=None, n_cols=None, dtype=None):
        """
        Open file.

        Parameters
        ----------
        n_bands : int, required for writing
            Maximum number of bands.
        n_rows : int, required for writing
            Number of rows.
        n_cols : int, required for writing
            Number of columns.
        dtype : str, required for writing
            Data type.
        """
        if self.mode == 'w':
            self.src = self.driver.Create(self.filepath, n_cols, n_rows,
                                          n_bands, gdal_dtype[dtype],
                                          self.gdal_opt)

            if self.geotrans is not None:
                self.src.SetGeoTransform(self.geotrans)

            if self.sref is not None:
                self.src.SetProjection(self.sref)

            if self.tags is not None:
                if 'description' in self.tags:
                    self.src.SetDescription(self.tags['description'])
                if 'gcps' in self.tags:
                    if len(self.tags['gcps']) != 0:
                        self.src.SetGCPs(self.tags['gcps'],
                                         self.tags['spatialreference'])

            if self.metadata is not None:
                self.src.SetMetadata(self.metadata)

            if n_bands == 1:
                self.shape = (n_rows, n_cols)
            else:
                self.shape = (n_bands, n_rows, n_cols)

        if self.mode == 'r':
            self.src = gdal.Open(self.filepath, gdal.GA_ReadOnly)

            n_rasters = self.src.RasterCount
            if n_rasters == 1:
                self.shape = (self.src.RasterYSize, self.src.RasterXSize)
            else:
                self.shape = (n_rasters, self.src.RasterYSize, self.src.RasterXSize)

            self.geotrans = self.src.GetGeoTransform()
            self.sref = self.src.GetProjection()

            self.metadata = self.src.GetMetadata()

            # assume all bands are like the first
            self.dtype = gdal.GetDataTypeName(self.src.GetRasterBand(1).DataType)

            self.chunks = list(self.src.GetRasterBand(1).GetBlockSize())[::-1]

        if self.src is None:
            raise IOError("Open failed: %s".format(self.filepath))

    def read(self, row=None, col=None, n_rows=1, n_cols=1, band=None, nodataval=None,
             decoder=None, decoder_kwargs=None):
        """
        Read data from raster file.

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
        band : int or list of int, optional
            Band numbers (starting with 1). If None, all bands will be read.
        nodataval : tuple or list, optional
            List of no data values for each band.
            Default: -9999 for each band.
        decoder : function, optional
            Decoding function expecting a NumPy array as input.
        decoder_kwargs : dict, optional
            Keyword arguments for the decoder.

        Returns
        -------
        data : numpy.ndarray
            Data set.
        """
        if nodataval is not None and not isinstance(nodataval, list):
            nodataval = [nodataval]

        decoder_kwargs = {} if decoder_kwargs is None else decoder_kwargs

        if row is None and col is not None:
            row = 0
            n_cols = self.shape[-1]
        elif row is not None and col is None:
            col = 0
            n_rows = self.shape[-2]

        if band is None:
            if row is None and col is None:
                data = self.src.ReadAsArray()
            else:
                data = self.src.ReadAsArray(col, row, n_cols, n_rows)

            if decoder is not None:
                if nodataval is None:
                    nodataval = [-9999]
                data = decoder(data, nodataval=nodataval[0], **decoder_kwargs)
        else:
            if not isinstance(band, list):
                band = [band]

            if nodataval is None:
                nodataval = [-9999] * len(band)

            data_list = []
            for i, band in enumerate(band):
                gdal_band = self.src.GetRasterBand(int(band))
                if gdal_band is None:
                    raise IOError("Reading band {:} failed.".format(band))
                if row is None and col is None:
                    data = gdal_band.ReadAsArray()
                else:
                    data = gdal_band.ReadAsArray(col, row, n_cols, n_rows)

                tags = self.read_tags(band)

                if self.auto_decode:
                    if (tags['scale_factor'] != 1.) or (tags['add_offset'] != 0.):
                        data = data.astype(float)
                        data[data == tags['nodata']] = np.nan
                        data = data * tags['scale_factor'] + tags['add_offset']
                    else:
                        wrn_msg = "Automatic decoding is activated for band '{}', " \
                                  "but attribute 'scale' and 'offset' are missing!".format(band)
                        warnings.warn(wrn_msg)
                else:
                    if decoder is not None:
                        data = decoder(data, nodataval=nodataval[i], **decoder_kwargs)
                data_list.append(data)

            data = np.vstack(data_list)

        return data

    def read_tags(self, band):
        """
        Read tags from raster file for a band.

        Parameters
        ----------
        band : int
            Band number (starting with 1).

        Returns
        -------
        tags : dict
            Tags in the raster file.
        """
        band = int(band)
        description = self.src.GetDescription()
        metadata = self.src.GetMetadata()
        geotransform = self.src.GetGeoTransform()
        spatialreference = self.src.GetProjection()
        gcps = self.src.GetGCPs()
        blockxsize, blockysize = self.src.GetRasterBand(band).GetBlockSize()
        nodata = self.src.GetRasterBand(band).GetNoDataValue()
        scale_factor = self.src.GetRasterBand(band).GetScale()
        offset = self.src.GetRasterBand(band).GetOffset()
        dtype = gdal_array.GDALTypeCodeToNumericTypeCode(
            self.src.GetRasterBand(band).DataType)

        tags = {'description': description,
                'metadata': metadata,
                'geotransform': geotransform,
                'spatialreference': spatialreference,
                'gcps': gcps,
                'nodata': nodata,
                'scale_factor': scale_factor,
                'add_offset': offset,
                'datatype': dtype,
                'blockxsize': blockxsize,
                'blockysize': blockysize}

        return tags

    def write(self, data, band=None, encoder=None, nodataval=None, encoder_kwargs=-9999, scale_factor=1., add_offset=0, ct=None):
        """
        Write data into raster file.

        Parameters
        ----------
        data : numpy.ndarray (2d or 3d)
            Raster data set, either as single image (2d) or as stack (3d).
            Dimensions [band, x, y]
        band : int, optional
            Band number (starting with 1). If band number is provided,
            raster data set has to be 2d.
            Default: Raster data set is 3d and all bands a written at
            the same time.
        encoder : function
            Encoding function expecting a NumPy array as input.
        encoder_kwargs : dict, optional
            Keyword arguments for the encoder.
        nodataval : tuple or list, optional
            List of no data values for each band.
            Default: -9999 for each band.
        scale_factor : number or tuple or list, optional
            Number or list of no data values for each band.
            Default: 1. for each band.
        add_offset : number or tuple or list, optional
            Number or list of no data values for each band.
            Default: 0. for each band.
        ct : tuple or list, optional
            List of color tables for each band.
            Default: No color tables set.

        """

        if not self.overwrite and os.path.exists(self.filepath):
            raise RuntimeError("File exists {:}".format(self.filepath))

        encoder_kwargs = {} if encoder_kwargs is None else encoder_kwargs

        # get data shape properties
        if data.ndim == 3:
            n_data_layers, n_data_rows, n_data_cols = data.shape
        elif data.ndim == 2:
            n_data_layers = 1
            n_data_rows, n_data_cols = data.shape
        else:
            err_msg = "Only 2d or 3d arrays are supported"
            raise ValueError(err_msg)

        if not isinstance(nodataval, (tuple, list)):
            nodataval = [-9999] * n_data_layers
        if not isinstance(scale_factor, (tuple, list)):
            scale_factor = [scale_factor] * n_data_layers
        if not isinstance(add_offset, (tuple, list)):
            add_offset = [add_offset] * n_data_layers

        if not isinstance(ct, list):
            ct = [ct] * n_data_layers

        # always set band to first band if a band number is not provided and the data is 2D
        if band is None and data.ndim == 2:
            band = 1

        if self.src is None:
            if self.n_bands is None:
                # if the band number is larger than the layer number, reset the layer number to the band number
                if band is not None and band > n_data_layers:
                    self.n_bands = band
                else:
                    self.n_bands = n_data_layers
            self._open(n_bands=self.n_bands, n_rows=n_data_rows, n_cols=n_data_cols, dtype=data.dtype.name)
        else:
            if len(self.shape) == 3:
                _, n_rows, n_cols = self.shape
            else:
                n_rows, n_cols = self.shape

            if (n_rows, n_cols) != (n_data_rows, n_data_cols):
                err_msg = "GeoTIFF dimensions ({},{}) and data dimensions ({},{}) mismatch.".format(n_rows, n_cols,
                                                                                                    n_data_rows,
                                                                                                    n_data_cols)
                raise ValueError(err_msg)

        if band is None:
            for b in range(n_data_layers):
                band = int(b + 1)
                if encoder is not None:
                    self.src.GetRasterBand(band).WriteArray(encoder(data[b, :, :],
                                                                    nodataval=nodataval[b],
                                                                    **encoder_kwargs))
                else:
                    self.src.GetRasterBand(band).WriteArray(data[b, :, :])
                self.src.GetRasterBand(band).SetNoDataValue(nodataval[b])
                self.src.GetRasterBand(band).SetScale(scale_factor[b])
                self.src.GetRasterBand(band).SetOffset(add_offset[b])
                if ct[b] is not None:
                    self.src.GetRasterBand(band).SetRasterColorTable(ct[b])
                else:
                    # helps to view layers e.g. as quicklooks  #TODO: @bbm why 2? Document it.
                    self.src.GetRasterBand(band).SetRasterColorInterpretation(2)
        else:
            if band > self.n_bands:
                err_msg = "Band number ({}) is larger than number of bands ({})".format(band, self.n_bands)
                raise ValueError(err_msg)
            band = int(band)
            if data.ndim != 2:
                err_msg = "Array needs to have 2 dimensions [height, width]"
                raise ValueError(err_msg)
            else:
                if encoder is not None:
                    # nodataval only first element is needed, because only one band is given
                    self.src.GetRasterBand(band).WriteArray(encoder(data, nodataval=nodataval[0], **encoder_kwargs))
                else:
                    self.src.GetRasterBand(band).WriteArray(data)

                self.src.GetRasterBand(band).SetScale(scale_factor[0])
                self.src.GetRasterBand(band).SetOffset(add_offset[0])
                if nodataval is not None:
                    self.src.GetRasterBand(band).SetNoDataValue(nodataval[0])
                if ct[0] is not None:
                    self.src.GetRasterBand(band).SetRasterColorTable(ct[0])
                else:
                    # helps to view layers e.g. as quicklooks  #TODO: @bbm why 2? Document it.
                    self.src.GetRasterBand(band).SetRasterColorInterpretation(2)

    def flush(self):
        """
        Flush data on disk.
        """
        if self.src is not None:
            self.src.FlushCache()

    def close(self):
        """
        Close the dataset
        """
        self.src = None

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()
