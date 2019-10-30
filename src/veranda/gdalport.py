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
A sample class and Some handy functions for using gdal library to read/write
data.
"""

import os
import subprocess
from itertools import cycle

import numpy as np
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
from gdalconst import GA_Update

gdal_datatype = {"uint8": gdal.GDT_Byte,
                 "int16": gdal.GDT_Int16,
                 "int32": gdal.GDT_Int32,
                 "uint16": gdal.GDT_UInt16,
                 "uint32": gdal.GDT_UInt32,
                 "float32": gdal.GDT_Float32,
                 "float64": gdal.GDT_Float64,
                 "complex64": gdal.GDT_CFloat32,
                 "complex128": gdal.GDT_CFloat64
                 }


gdal_resample_type = {"nearst": gdal.GRA_NearestNeighbour,
                      "bilinear": gdal.GRA_Bilinear,
                      "cubic": gdal.GRA_Cubic,
                      "cubicspline": gdal.GRA_CubicSpline,
                      "lanczos": gdal.GRA_Lanczos,
                      "average": gdal.GRA_Average,
                      "mode": gdal.GRA_Mode
                      }


class GdalImage:

    """
    A sample class to access a image with GDAL library.
    """

    def __init__(self, gdaldataset, filepath):
        """

        Parameters
        ----------
        dataset : str
            Data set name.
        filepath : str
            File path.
        """
        self.dataset = gdaldataset
        self.filepath = filepath

    def close(self):
        """
        Close the dataset
        """
        self.dataset = None

    def read_band(self, band_idx, subset=None):
        """
        Read data from given band.

        Parameters
        ----------
        band_idx : int
            The band index starting from 1.
        subset : list or tuple, optional
            The subset should be in pixels, like this (xmin, ymin, xmax, ymax).
            Default: None

        Returns
        -------
        data : numpy.ndarray
            2d array including data reading from given band.
        """
        if band_idx < 1 or band_idx > self.dataset.RasterCount:
            raise IndexError("band index is out of range")

        band = self.dataset.GetRasterBand(band_idx)

        if subset is None:
            data = band.ReadAsArray(0, 0, band.XSize, band.YSize)
        else:
            data = band.ReadAsArray(subset[0], subset[1],
                                    subset[2], subset[3])

        return data

    def XSize(self):
        """
        Get the width of the image.

        Returns
        -------
        img_xsize : int
            Image width.
        """
        return self.dataset.RasterXSize if self.dataset else None

    def YSize(self):
        """
        Get the height of the image.

        Returns
        -------
        img_ysize : int
            Image width.
        """
        return self.dataset.RasterYSize if self.dataset else None

    @property
    def shape(self):
        """
        Get the shape of the image.

        Returns
        -------
        img_shape : tuple of ints
            Image shape (height, width).
        """
        return (self.YSize(), self.XSize())

    def get_band_nodata(self, band_idx=1):
        """
        Get band nodata value.

        Parameters
        ----------
        band_idx : int, optional
            The band index starting from 1. Default: 1

        Returns
        -------
        nodata : int
            nodata value if it's available, otherwise it will return None
        """
        if band_idx < 1 or band_idx > self.dataset.RasterCount:
            raise IndexError("band index is out of range")
        return self.dataset.GetRasterBand(band_idx).GetNoDataValue()

    def get_raster_nodata(self):
        """
        Get the nodata value for all bands in a list this is compatible with
        the write_image's nodata parameter.

        Returns
        -------
        nodata
            no data values in a list if it's available, otherwise it
            will return None.
        """
        nodata = []
        for i in range(0, self.dataset.RasterCount):
            nodata.append(self.dataset.GetRasterBand(i + 1).GetNoDataValue())

        return nodata if len(nodata) >= 0 and not all(d is None for d in nodata) else None

    def read_all_band(self):
        """
        Read the data of all the bands.

        Returns
        -------
        m : numpy.ndarray
            Data from all bands.
        """
        m = np.full((self.band_count(), self.YSize(), self.XSize()), 0.0)

        for bandIdx in range(self.band_count()):
            m[bandIdx] = self.read_band(bandIdx + 1)

        return m

    def get_band_dtype(self, band_idx=1):
        """
        Get the data type of given band.

        Parameters
        ----------
        band_idx : int, optional
            The band index starting from 1. Default: 1

        Returns
        -------
        dtype : str
            Data type.
        """
        if band_idx < 1 or band_idx > self.dataset.RasterCount:
            raise IndexError("band index is out of range")
        return self.dataset.GetRasterBand(band_idx).DataType

    def geotransform(self):
        """
        Get the geotransform data.

        Returns
        -------
        geo_data : numpy.ndarray
            Geotransfrom data.
        """
        return self.dataset.GetGeoTransform() if self.dataset else None

    def projection(self):
        """
        Get the projection string in wkt format.

        Returns
        -------
        geo_data : numpy.ndarray
            Geotransfrom data.
        """
        return self.dataset.GetProjection() if self.dataset else None

    def colormap(self, band_idx=1):
        """
        Get the colormap of given band.

        Returns
        -------
        colormap : list
            Colormap.
        """
        if band_idx < 1 or band_idx > self.dataset.RasterCount:
            raise IndexError("band index is out of range")
        ct = self.dataset.GetRasterBand(band_idx).GetColorTable()
        if ct is None:
            return None

        colormap = []
        for i in range(ct.GetCount()):
            colormap.append(ct.GetColorEntry(i))

        return colormap

    def band_count(self):
        """
        Get the band count.

        Returns
        -------
        bc : int
            Number of bands.
        """
        return self.dataset.RasterCount if self.dataset else None

    def get_extent(self):
        """
        Get the extent of the image as.

        Returns
        -------
        img_extend : list
            Image extend (xmin, ymin, xmax, ymax).
        """
        geot = self.geotransform()
        return (geot[0], geot[3] + self.YSize() * geot[5],
                geot[0] + self.XSize() * geot[1], geot[3])

    def pixel2coords(self, x, y):
        """
        Get global coordinates from pixel x, y coordinates.

        Parameters
        ----------
        x : int
           x-coordinate.
        y : int
           y-coordinate.

        Returns
        -------
        xp, yp : tuple
           pixel x-coordinate.p, y-coordinate.
        """
        xoff, a, b, yoff, d, e = self.geotransform()

        xp = a * x + b * y + xoff
        yp = d * x + e * y + yoff

        return (xp, yp)

    def coords2pixel(self, l1, l2):
        """
        Translate coordinates to pixel location.

        Parameters
        ----------
        l1 : int
            x-coordinate
        l2 : int
            y-coordinate

        Returns
        -------
        row, col : list
            Row and column position
        """
        gt = self.geotransform()
        col = int((l1 - gt[0]) / gt[1])
        row = int((l2 - gt[3]) / gt[5])

        if col < 0 or col >= self.XSize() or row < 0 or row >= self.YSize():
            return None

        return [row, col]

    def inside(self, l1, l2):
        """
        Checks if a pixel is in this image.

        Parameters
        ----------
        l1 : int
            x-coordinate
        l2 : int
            y-coordinate

        Returns
        -------
        flag : boolean
            True if pixel is inside image, False if not.
        """
        x, y = self.coords2pixel(l1, l2)

        return x >= 0 and x < self.XSize() and y >= 0 and y < self.YSize()


def open_image(filename):
    """
    Open an image file

    Parameters
    ----------
    filename : str
        Full path string of input file.

    Returns
    -------
    gdal_img : GdalImage object
        GdalImage object if successful, otherwise None

    Raise
    -----
    IOError
        if fail to open the image file
    """
    dataset = gdal.Open(filename, gdal.GA_ReadOnly)

    if dataset is None:
        raise IOError("cannot open %s" % filename)

    return GdalImage(dataset, filename)


def dtype_np2gdal(datatype):
    """
    Get gdal data type from datatype.

    Parameters
    ----------
    datatype :  str
        Data type string in python such as "uint8", "float32" and so forth.

    Returns
    -------
    gdal_datatype : str
        Gdal data type
    """
    datatype = datatype.lower()
    return gdal_datatype[datatype] if datatype in gdal_datatype else None


def rtype_str2gdal(resampling_method):
    """
    Get gdal resample type from resampling method string

    Parameters
    ----------
    resampling_method : str
        Resampling method.

    Returns
    -------
    gdal_resam :
        Gdal resampling.
    """
    mthd = resampling_method.lower()
    return gdal_resample_type[mthd] if mthd in gdal_resample_type else None


def create_dataset(filename, datatype, dims, frmt="GTiff", geotransform=None,
                   projection=None, option=None):
    """
    Create GDAL dataset.

    Parameters
    ----------
    filename : string
        full path of output filename
    datatype : string
        data type string like numpy's dtpye
    dims : tuple
        Dimension of the dataset in the format of (bands, XSize, YSize)
    frmt :  string
        The format of output image should be a string that gdal supported
    geotransform : array like
        It contains six geotransform parameters
    projection : string
        projection definition string

    Returns
    -------
    GDAL dataset

    Raise
    -----
    IOError
        if fail to obtain driver with specific format or to create the output
        dataset
    """
    driver = gdal.GetDriverByName(frmt)
    gdaldatatype = dtype_np2gdal(datatype)
    if driver is None:
        raise IOError("cannot get driver for {}".format(frmt))
    band_count, xsize, ysize = dims
    if option is None:
        out_ds = driver.Create(
            filename, xsize, ysize, band_count, gdaldatatype)
    else:
        out_ds = driver.Create(filename, xsize, ysize, band_count,
                               gdaldatatype, option)
    if out_ds is None:
        raise IOError("cannot create file of {}".format(filename))
    if geotransform is not None:
        out_ds.SetGeoTransform(geotransform)
    if projection is not None:
        out_ds.SetProjection(projection)

    return out_ds


def write_image(image, filename, frmt="GTiff", nodata=None,
                geotransform=None, projection=None, option=None,
                colormap=None, compress=True, overwrite=True,
                ref_image=None, dtype=None):
    """
    Output image into filename with specific format

    Parameters
    ----------
    image : array like
        two dimension array containing data that will be stored
    filename : string
        full path of output filename
    frmt :  string
        the format of output image should be a string that gdal supported.
    nodata : list, optional
        a list contian the nodata values of each band
    geotransform : array like
        contain six geotransform parameters
    projection : string
        projection definition string
    dtype
        Datatype that we want to use, per default dtype = dtype.image

    Raise
    -----
        IOError
            if IO error happens
        ValueError
            if some invalid parameters are given
    """
    if ref_image is not None:
        if geotransform is None:
            geotransform = ref_image.geotransform()

        if projection is None:
            projection = ref_image.projection()

        if nodata is None:
            nodata = ref_image.get_raster_nodata()

    if overwrite is False and os.path.exists(filename):
        return None

    # to make sure dim of image is 2 or 3
    if image is None or image.ndim < 2 or image.ndim > 3:
        raise ValueError(
            "The image is None or it's dimension isn't in two or three.")
    dims = (1, image.shape[1], image.shape[0]) if image.ndim == 2 \
        else (image.shape[0], image.shape[2], image.shape[1])

    # Enable compression of the file

    if compress:
        if option is None:
            option = ['COMPRESS=LZW']
        else:
            if not filter(lambda x: x.upper().startswith("COMPRESS"), option):
                option.append("COMPRESS=LZW")
            else:
                print("Info: use compression method set by option!")

    # create dataset
    ds = create_dataset(filename, str(image.dtype), dims, frmt, geotransform,
                        projection, option)
    # write data
    if image.ndim == 2:
        ds.GetRasterBand(1).WriteArray(image, 0, 0)
    else:
        for i in range(ds.RasterCount):
            ds.GetRasterBand(i + 1).WriteArray(
                image[i] if dtype is None else image[i].astype(dtype), 0, 0)

    # set nodata for each band
    if nodata is not None:
        assert ds.RasterCount == len(nodata) or len(
            nodata) == 1, "Mismatch of nodata values and RasterCount"
        for i, val in zip(range(ds.RasterCount), cycle(nodata)):
            ds.GetRasterBand(i + 1).SetNoDataValue(val)

    # colormaps are only supported for 1 band rasters
    if colormap is not None and ds.RasterCount == 1:
        ct = gdal.ColorTable()
        for i, color in enumerate(colormap):
            if len(color) == 3:
                color = list(color) + [0, ]
            ct.SetColorEntry(i, tuple(color))

        for i in range(ds.RasterCount):
            ds.GetRasterBand(i + 1).SetRasterColorTable(ct)

    ds.FlushCache()
    ds = None


def write_geometry(geom, fname, format="shapefile"):
    """
    Write an geometry to a vector file.

    parameters
    ----------
    geom : Geometry
        geometry object
    fname : string
        full path of the output file name
    format : string
        format name. currently only shape file is supported
    """
    drv = ogr.GetDriverByName("ESRI Shapefile")
    dst_ds = drv.CreateDataSource(fname)
    srs = geom.GetSpatialReference()

    dst_layer = dst_ds.CreateLayer("out", srs=srs)
    fd = ogr.FieldDefn('DN', ogr.OFTInteger)
    dst_layer.CreateField(fd)

    feature = ogr.Feature(dst_layer.GetLayerDefn())
    feature.SetField("DN", 1)
    feature.SetGeometry(geom)
    dst_layer.CreateFeature(feature)

    feature.Destroy()
    dst_ds.Destroy()


def extent2polygon(extent, wkt=None):
    """
    Create a polygon geometry from extent.

    Parameters
    ----------
    extent : list
        extent in terms of [xmin, ymin, xmax, ymax]
    wkt : string
        project string in well known text format

    Returns
    -------
    geom_area : ogr.Geometry
        Polygon geometry.
    """
    area = [(extent[0], extent[1]), (extent[2], extent[1]),
            (extent[2], extent[3]), (extent[0], extent[3])]

    edge = ogr.Geometry(ogr.wkbLinearRing)
    [edge.AddPoint(x, y) for x, y in area]
    edge.CloseRings()
    geom_area = ogr.Geometry(ogr.wkbPolygon)
    geom_area.AddGeometry(edge)
    if wkt:
        geo_sr = osr.SpatialReference()
        geo_sr.ImportFromWkt(wkt)
        geom_area.AssignSpatialReference(geo_sr)

    return geom_area


def call_gdal_util(util_name, gdal_path=None, src_files=None, dst_file=None,
                   options={}):
    """
    Call gdal utility to run the operation.
    http://www.gdal.org/gdal_utilities.html

    Parameters
    ----------
    util_name : string
        pre-defined name of the utility
        (e.g. "gdal_translate": convert raster data between different formats,
        potentially performing some operations like subsettings, resampling,
        and rescaling pixels in the process.)
    src_files : string
        The source dataset name. It can be either file name,
        URL of data source or subdataset name for multi-dataset files.
    dst_file : string
        The destination file name.
    gdal_path : string
        It the path where your gdal installed. If gpt command can not found by
        the os automatically, you should give this path.
    options : dict
        A dictionary of options. You can find all options at
        http://www.gdal.org/gdal_utilities.html

    Returns
    -------
    succeed :

    output :

    """
    # define specific options
    _opt_2b_in_quote = ["-mo", "-co"]

    # get the gdal installed path if it is set in system environmental variable
    if not gdal_path:
        gdal_path = _find_gdal_path()
    if not gdal_path:
        raise OSError("gdal utility not found in system environment!")

    # prepare the command string
    cmd = []
    gdal_cmd = os.path.join(gdal_path, util_name) if gdal_path else util_name
    # put gdal_cmd in double quotation
    cmd.append('"%s"' % gdal_cmd)

    for k, v in options.items():
        is_iterable = isinstance(v, (tuple, list))
        if k in _opt_2b_in_quote:
            if (k == "-mo" or k == "-co") and is_iterable:
                for i in range(len(v)):
                    cmd.append(" ".join((k, '"%s"' % v[i])))
            else:
                cmd.append(" ".join((k, '"%s"' % v)))
        else:
            cmd.append(k)
            if is_iterable:
                cmd.append(' '.join(map(str, v)))
            else:
                cmd.append(str(v))

    # add source files and destination file (in double quotation)
    if isinstance(src_files, (tuple, list)):
        src_files_str = " ".join(src_files)
    else:
        src_files_str = '"%s"' % src_files
    cmd.append(src_files_str)
    if dst_file is not None:
        cmd.append('"%s"' % dst_file)

    output = subprocess.check_output(" ".join(cmd), shell=True, cwd=gdal_path)
    succeed = _analyse_gdal_output(str(output))

    return succeed, output


def _find_gdal_path():
    """
    Find the gdal installed path from the system enviroment variables.

    Returns
    -------
    path : str
        GDAL install path.
    """
    evn_name = "GDAL_UTIL_HOME"
    return os.environ[evn_name] if evn_name in os.environ else None


def _analyse_gdal_output(output):
    """
    Analyse the output from gpt to find if it executes successfully.

    Parameters
    ----------
    output : str
        Ouptut from gpt.

    Returns
    -------
    flag : boolean
        False if "Error" is found and True if not.
    """
    # return false if "Error" is found.
    if 'error' in output.lower():
        return False
    # return true if "100%" is found.
    elif '100 - done' in output.lower():
        return True
    # otherwise return false.
    else:
        return False


def gen_qlook(src_file, dst_file=None, src_nodata=None, gdal_path=None,
              min_stretch=None, max_stretch=None, resize_factor=('3%', '3%'), ct=None, scale=True,
              output_format="GTiff"):
    '''
    Parameters
    ----------

    src_file: string (required)
        The source dataset name. It can be either file name (full path)
        URL of data source or subdataset name for multi-dataset files.
    dst_file: string (optional)
        The destination file name (full path).
        if not provided the default of dst_file is src_file name+ '_qlook'
    src_nodata: string (optional)
        Set nodata masking values for input image
    gdal_path : string (optional)
        The path where your gdal installed. If gdal path is found by
        the os automatically, you should give this path.
    min_stretch, max_stretch: (optional)
        rescale the input pixels values from
        the range min_stretch to max_stretch to the range 0 to 255.
        If omitted pixel values will be scaled to minimum and maximum
    resize_factor: (optional)
        Set the size of the output file. Outsize is in pixels and lines
        unless '%' is attached in which case it is as a fraction of the
        input image size. The default is ('3%', '3%')
    ct: gdal ColorTable class (optional)
    output_format: str
        Output format of the quicklook. At the moment we support 'GTiff'
        and 'jpeg'

    Returns
    -------
    a tuple of (succeed, output)
        succeed: results of process as True/False
        output: cmd output of gdal utility
    '''

    output_format = output_format.lower()
    f_ext = {'gtiff': 'tif',
             'jpeg': 'jpeg'}

    # get the gdal installed path if it is set in system environmental variable
    if not gdal_path:
        gdal_path = _find_gdal_path()
    if not gdal_path:
        raise OSError("gdal utility not found in system environment!")

    # check if destination file name is given. if not use the source directory
    if dst_file is None:
        dst_file = os.path.join(os.path.dirname(src_file),
                                os.path.basename(src_file).split('.')[0]
                                + '_qlook.{}'.format(f_ext[output_format]))


    # prepare options for gdal_transalte
    options_dict = {'gtiff': {'-of': 'GTiff', '-co': 'COMPRESS=LZW',
                              '-mo': ['parent_data_file="%s"' % os.path.basename(src_file)], '-outsize': resize_factor,
                              '-ot': 'Byte'},
                    'jpeg':  {'-of': 'jpeg', '-co': 'QUALITY=95',
                              '-mo': ['parent_data_file="%s"' % os.path.basename(src_file)], '-outsize': resize_factor}}

    options = options_dict[output_format]

    if scale:
        options["-scale"] = ' '
    if scale and (min_stretch is not None)and(max_stretch is not None):
        options['-scale'] = (min_stretch, max_stretch, 0, 255)
        # stretching should be done differently if nodata value exist.
        # depending on nodatavalue, first or last position is reserved for nodata value
        if src_nodata is not None:
            if src_nodata < min_stretch:
                options['-scale'] = (min_stretch, max_stretch, 1, 255)
                options['-a_nodata'] = 0
            elif src_nodata > max_stretch:
                options['-scale'] = (min_stretch, max_stretch, 0, 254)
                options['-a_nodata'] = 255


    # call gdal_translate to resize input file
    succeed, output = call_gdal_util('gdal_translate', src_files=src_file,
                                     dst_file=dst_file, gdal_path=gdal_path,
                                     options=options)


    if (output_format == 'gtiff') and (ct is not None):
        # Update quick look image, attach color table this does not so easily
        # work with JPEG so we keep the colortable of the original file
        ds = gdal.Open(dst_file, GA_Update)
        # set the new color table
        if ds.RasterCount == 1:
            ds.GetRasterBand(1).SetRasterColorTable(ct)

    return succeed, output

if __name__ == '__main__':
    pass