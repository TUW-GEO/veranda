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

import numpy as np
from osgeo import gdal
from osgeo import gdal_array

# gdal.SetCacheMax(1024**3)

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


def get_pixel_coords(x, y, xres, yres, xmin, ymax):
    """
    Translate x, y coordinates to cols, rows.

    Example:
        col, row = map_pixel(x, y, geotransform[1],
            geotransform[-1], geotransform[0], geotransform[3])

    Parameters
    ----------
    x : float, numpy.ndarray
        X coordinates.
    y : float, numpy.ndarray
        Y coordinates.
    xres : float
        X resolution.
    yres : float
        Y resolution.

    Returns
    -------
    col : int, numpy.ndarray
        Column coordinates.
    row : int, numpy.ndarray
        Row coordinates.
    """
    col = np.around((x - xmin) / xres).astype(int)
    row = np.around((y - ymax) / yres).astype(int)

    return col, row


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
    filename : str
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
    geotransform : tuple or list, optional
        Geotransform parameters (default (0, 1, 0, 0, 0, 1)).
        0: Top left x
        1: W-E pixel resolution
        2: Rotation, 0 if image is "north up"
        3: Top left y
        4: Rotation, 0 if image is "north up"
        5: N-S pixel resolution (negative value if North up)
    spatialref : str, optional
        Coordinate Reference System (CRS) in Wkt form (default: None).
    tags : dict, optional
        Meta data tags (default: None).
    overwrite : boolean, optional
        Flag if file can be overwritten if it already exists (default: True).
    gdal_opt : dict, optional
        Driver specific control parameters (default: None).
    """

    def __init__(self, filename, mode='r', count=None, compression='LZW',
                 blockxsize=512, blockysize=512,
                 geotransform=(0, 1, 0, 0, 0, 1), spatialref=None, tags=None,
                 overwrite=True, gdal_opt=None):

        self.src = None
        self.filename = filename
        self.mode = mode
        self.count = count
        self.driver = gdal.GetDriverByName('GTiff')
        self.tags = tags
        self.spatialref = spatialref
        self.geotransform = geotransform
        self.overwrite = overwrite

        self.n_layer = count
        self.shape = None
        self.ndim = None
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

    @property
    def metadata(self):
        if self.src is not None:
            return self.src.GetMetadata()
        else:
            return None

    def _open(self, xsize=None, ysize=None, dtype=None):
        """
        Open file.

        Parameters
        ----------
        xsize : int, required for writing
            Number of columns.
        ysize : int, required for writing
            Number of rows.
        dtype : str, required for writing
            Data type.
        """
        if self.mode == 'w':
            self.src = self.driver.Create(self.filename, ysize, xsize,
                                          self.count, gdal_dtype[dtype],
                                          self.gdal_opt)

            if self.geotransform:
                self.src.SetGeoTransform(self.geotransform)

            if self.spatialref:
                self.src.SetProjection(self.spatialref)

            if self.tags is not None:
                if 'description' in self.tags:
                    self.src.SetDescription(self.tags['description'])
                if 'metadata' in self.tags:
                    self.src.SetMetadata(self.tags['metadata'])
                if 'gcps' in self.tags:
                    if len(self.tags['gcps']) != 0:
                        self.src.SetGCPs(self.tags['gcps'],
                                         self.tags['spatialreference'])

        if self.mode == 'r':
            self.src = gdal.Open(self.filename, gdal.GA_ReadOnly)

            self.n_layers = self.src.RasterCount

            if self.n_layers == 1:
                self.shape = (self.src.RasterYSize, self.src.RasterXSize)
            else:
                self.shape = (self.src.RasterCount, self.src.RasterYSize,
                              self.src.RasterXSize)

            self.ndim = len(self.shape)
            self.geotransform = self.src.GetGeoTransform()
            self.spatialref = self.src.GetProjection()

            # assume all bands are like the first
            self.dtype = gdal.GetDataTypeName(
                self.src.GetRasterBand(1).DataType)

            self.chunks = list(self.src.GetRasterBand(1).GetBlockSize())[::-1]

        if self.src is None:
            raise IOError("Open failed: %s".format(self.filename))

    def read(self, band=1, return_tags=True, encode_func=None, sub_rect=None):
        """
        Read data from raster file.

        CAUTION: the returned geotags corresponds to input file and not the
                 returned src_arr (in case sub_rect is set)

        Parameters
        ----------
        band : int, optional
            Band number (starting with 1). Default: 1
        return_tags : bool, optional
            If set tags will be returned as well
        encode_func : func (optional)
            function of type "encode_func(data, tags)" to encode the array
            in-place.
        sub_rect : list (optional)
            Set this keyword to a four-element array, [Xoffset, Yoffset,
            width, height], that specifies a rectangular region within the
            input raster to extract.

        Returns
        -------
        data : numpy.ndarray
            Data set.
        tags : dict
            Tag information for band

        """
        if band is None:
            if sub_rect is None:
                data = self.src.ReadAsArray()
            else:
                data = self.src.ReadAsArray(
                    sub_rect[0], sub_rect[1], sub_rect[2], sub_rect[3])
        else:
            gdal_band = self.src.GetRasterBand(int(band))
            if gdal_band is None:
                raise IOError("Reading band {:} failed.".format(band))
            if sub_rect is None:
                data = gdal_band.ReadAsArray()
            else:
                data = gdal_band.ReadAsArray(
                    sub_rect[0], sub_rect[1], sub_rect[2], sub_rect[3])

        tags = self.read_tags(band)

        if encode_func is not None:
            data = encode_func(data, tags)

        if return_tags:
            return data, tags
        else:
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
        dtype = gdal_array.GDALTypeCodeToNumericTypeCode(
            self.src.GetRasterBand(band).DataType)

        tags = {'description': description,
                'metadata': metadata,
                'geotransform': geotransform,
                'spatialreference': spatialreference,
                'gcps': gcps,
                'nodata': nodata,
                'datatype': dtype,
                'blockxsize': blockxsize,
                'blockysize': blockysize}

        return tags

    def write(self, data, band=None, nodata=None, ct=None):
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
        nodata : tuple or list, optional
            List of no data values for each band.
            Default: -9999 for each band.
        ct : tuple or list, optional
            List of color tables for each band.
            Default: No color tables set.
        """
        if not self.overwrite and os.path.exists(self.filename):
            raise RuntimeError("File exists {:}".format(self.filename))

        if self.src is None:
            if data.ndim == 3:
                self.count = data.shape[0]
                self._open(data.shape[1], data.shape[2], data.dtype.name)
            elif data.ndim == 2:
                if self.count is None:
                    raise ValueError("Number of band (counts) not defined")
                self._open(data.shape[0], data.shape[1], data.dtype.name)
            else:
                raise ValueError("Only 2d or 3d array supported")

        if nodata is None:
            nodata = [-9999] * self.count
        if ct is None:
            ct = [None] * self.count

        if band is None:
            if data.ndim != 3:
                msg = "Array needs to have 3 dimensions [band, width, height]"
                raise ValueError(msg)
            else:
                for b in range(self.count):
                    bandn = int(b + 1)
                    self.src.GetRasterBand(bandn).WriteArray(data[b, :, :])
                    self.src.GetRasterBand(bandn).SetNoDataValue(nodata[b])
                    if ct[b] is not None:
                        self.src.GetRasterBand(bandn).SetRasterColorTable(
                            ct[b])
                    else:
                        # BBM: not completely sure about this: helps to view
                        # layers e.g. in Irfanview
                        self.src.GetRasterBand(
                            bandn).SetRasterColorInterpretation(2)
        else:
            band = int(band)
            if data.ndim != 2:
                msg = "Array needs to have 2 dimensions [width, height]"
                raise ValueError(msg)
            else:
                self.src.GetRasterBand(band).WriteArray(data)
                if nodata is not None:
                    self.src.GetRasterBand(band).SetNoDataValue(
                        nodata[band - 1])
                if ct[band - 1] is not None:
                    self.src.GetRasterBand(band).SetRasterColorTable(
                        ct[band - 1])

    def __setitem__(self, idx, data):
        """
        Write band data.

        Parameters
        ----------
        idx : slice
            Index band.
        data : numpy.ndarray
            2d or 3d array.
        """
        if self.src is None:
            if data.ndim == 3:
                self.count = data.shape[0]
                self._open(data.shape[1], data.shape[2], data.dtype.name)
            elif data.ndim == 2:
                if self.count is None:
                    raise ValueError("Number of band (counts) not defined")
                self._open(data.shape[0], data.shape[1], data.dtype.name)
            else:
                raise ValueError("Only 2d or 3d array supported")

        if isinstance(idx, int):
            slcz = slice(idx, idx + 1)
        else:
            slcz = idx

        if slcz.step:
            step = slcz.step
        else:
            step = 1

        for i, nband in enumerate(range(slcz.start, slcz.stop, step)):
            if data.ndim == 2:
                self.src.GetRasterBand(nband).WriteArray(data)
            if data.ndim == 3:
                self.src.GetRasterBand(nband).WriteArray(data[i, :, :])

    def __getitem__(self, idx):
        """
        Read band data.

        Parameters
        ----------
        idx : slice
           Index [band, x, y].

        Returns
        -------
        data : numpy.ndarray
            Data.
        """
        if len(idx) == 2:
            slcy, slcx = idx
            slcz = None
        elif len(idx) == 3:
            slcz, slcy, slcx = idx
            if isinstance(slcz, int):
                slcz = slice(slcz, slcz + 1)
        else:
            raise ValueError('Unsupported getitem call', idx)

        xoff = slcx.start if slcx.start is not None else 0
        yoff = slcy.start if slcy.start is not None else 0

        if slcx.start is None:
            xstart = 0
        else:
            xstart = slcx.start

        if slcx.stop is None:
            xstop = self.src.RasterYSize
        else:
            xstop = slcx.stop

        if slcy.start is None:
            ystart = 0
        else:
            ystart = slcy.start

        if slcx.stop is None:
            ystop = self.src.RasterXSize
        else:
            ystop = slcy.stop

        xs = xstop - xstart
        ys = ystop - ystart

        if slcz is None:
            # 2D case
            data = self.src.ReadAsArray(xoff, yoff, xs, ys)
        else:
            # 3D case
            band_data = []

            if slcz.step:
                step = slcz.step
            else:
                step = 1

            for i in range(slcz.start, slcz.stop, step):
                band_chunk = self.src.GetRasterBand(
                    i).ReadAsArray(xoff, yoff, xs, ys)
                band_data.append(band_chunk[np.newaxis, ...])

            data = np.vstack(band_data)

        return data

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
