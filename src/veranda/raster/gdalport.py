"""
Some handy functions using GDAL to manage geospatial raster mosaic.

"""

import os
import subprocess

from osgeo import gdal
from osgeo.gdalconst import GA_Update


NUMPY_TO_GDAL_DTYPE = {"bool": gdal.GDT_Byte,
                       "uint8": gdal.GDT_Byte,
                       "int8": gdal.GDT_Byte,
                       "uint16": gdal.GDT_UInt16,
                       "int16": gdal.GDT_Int16,
                       "uint32": gdal.GDT_UInt32,
                       "int32": gdal.GDT_Int32,
                       "float32": gdal.GDT_Float32,
                       "float64": gdal.GDT_Float64,
                       "complex64": gdal.GDT_CFloat32,
                       "complex128": gdal.GDT_CFloat64}

GDAL_TO_NUMPY_DTYPE = {gdal.GDT_Byte: "uint8",
                       gdal.GDT_Int16: "int16",
                       gdal.GDT_Int32: "int32",
                       gdal.GDT_UInt16: "uint16",
                       gdal.GDT_UInt32: "uint32",
                       gdal.GDT_Float32: "float32",
                       gdal.GDT_Float64: "float64",
                       gdal.GDT_CFloat32: "cfloat32",
                       gdal.GDT_CFloat64: "cfloat64"}

GDAL_RESAMPLE_TYPE = {"nearst": gdal.GRA_NearestNeighbour,
                      "bilinear": gdal.GRA_Bilinear,
                      "cubic": gdal.GRA_Cubic,
                      "cubicspline": gdal.GRA_CubicSpline,
                      "lanczos": gdal.GRA_Lanczos,
                      "average": gdal.GRA_Average,
                      "mode": gdal.GRA_Mode
                      }


def dtype_np2gdal(datatype):
    """
    Get GDAL mosaic type from NumPy-style datatype.

    Parameters
    ----------
    datatype :  str
        Data type string in python such as "uint8", "float32" and so forth.

    Returns
    -------
    gdal_datatype : str
        Gdal mosaic type.

    """
    return NUMPY_TO_GDAL_DTYPE.get(datatype.lower())


def rtype_str2gdal(resampling_method):
    """
    Get GDAL resample type from resampling method string.

    Parameters
    ----------
    resampling_method : str
        Resampling method.

    Returns
    -------
    object :
        GDAL resampling.

    """
    return GDAL_RESAMPLE_TYPE.get(resampling_method.lower())


def call_gdal_util(util_name, gdal_path=None, src_files=None, dst_file=None,
                   options={}):
    """
    Call gdal utility to run the operation.
    http://www.gdal.org/gdal_utilities.html

    Parameters
    ----------
    util_name : string
        pre-defined name of the utility
        (e.g. "gdal_translate": convert raster mosaic between different formats,
        potentially performing some operations like subsettings, resampling,
        and rescaling pixels in the process.)
    src_files : string
        The source dataset name. It can be either file name,
        URL of mosaic source or subdataset name for multi-dataset files.
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
    succeed : bool
        True if process was successful.
    output : str
        Console output.

    """
    # define specific options
    _opt_2b_in_quote = ["-mo", "-co"]

    # get the gdal installed path if it is set in system environmental variable
    if not gdal_path:
        gdal_path = _find_gdal_path()
    if not gdal_path:
        err_msg = "GDAL utility not found in system environment!"
        raise OSError(err_msg)

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
    """
    Parameters
    ----------

    src_file: string (required)
        The source dataset name. It can be either file name (full path)
        URL of mosaic source or subdataset name for multi-dataset files.
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
    succeed : bool
        True if process was successful.
    output : str
        Console output.

    """

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