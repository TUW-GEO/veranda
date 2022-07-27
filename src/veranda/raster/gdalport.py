""" Some handy functions using GDAL to manage geospatial raster data. """

import os
import subprocess
from typing import Tuple
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

GDAL_RESAMPLE_TYPE = {"nearest": gdal.GRA_NearestNeighbour,
                      "bilinear": gdal.GRA_Bilinear,
                      "cubic": gdal.GRA_Cubic,
                      "cubicspline": gdal.GRA_CubicSpline,
                      "lanczos": gdal.GRA_Lanczos,
                      "average": gdal.GRA_Average,
                      "mode": gdal.GRA_Mode
                      }


def dtype_np2gdal(np_dtype) -> object:
    """
    Get GDAL data type from a NumPy-style data type.

    Parameters
    ----------
    np_dtype :  str
        NumPy-style data type, e.g. "uint8".

    Returns
    -------
    str :
        GDAL data type.

    """
    return NUMPY_TO_GDAL_DTYPE.get(np_dtype.lower())


def rtype_str2gdal(resampling_method) -> object:
    """
    Get GDAL resample type from resampling method.

    Parameters
    ----------
    resampling_method : str
        Resampling method, e.g. "cubic".

    Returns
    -------
    object :
        GDAL resampling.

    """
    return GDAL_RESAMPLE_TYPE.get(resampling_method.lower())


def try_get_gdal_installation_path(gdal_path=None) -> str:
    """
    Tries to find the system path to the GDAL utilities if not already provided.

    Parameters
    ----------
    gdal_path : str, optional
        The path where your GDAL utilities are installed. By default, this function tries to look up this path in the
        environment variable "GDAL_UTIL_HOME". If this variable is not set, then `gdal_path` must be provided as a
        key-word argument.

    Returns
    -------
    gdal_path : str
        Path to installed GDAL utilities.

    """
    if not gdal_path:
        gdal_path = _find_gdal_path()
    if not gdal_path:
        err_msg = "GDAL utility not found in system environment!"
        raise OSError(err_msg)

    return gdal_path


def convert_gdal_options_to_command_list(options) -> list:
    """
    Prepares command line arguments for a GDAL utility.

    Parameters
    ----------
    options : dict, optional
        A dictionary storing additional settings for the process. You can find all options at
        http://www.gdal.org/gdal_utilities.html

    Returns
    -------
    cmd_options : list of str
        Command line options for a GDAL utility given as a list of strings.

    """
    # define specific options, which need to be quoted
    _opt_2b_in_quote = ["-mo", "-co"]

    cmd_options = []
    for k, v in options.items():
        is_iterable = isinstance(v, (tuple, list))
        if k in _opt_2b_in_quote:
            if (k == "-mo" or k == "-co") and is_iterable:
                for i in range(len(v)):
                    cmd_options.append(" ".join((k, string2cli_arg(v[i]))))
            else:
                cmd_options.append(" ".join((k, string2cli_arg(v))))
        else:
            cmd_options.append(k)
            if is_iterable:
                cmd_options.append(' '.join(map(str, v)))
            else:
                cmd_options.append(str(v))

    return cmd_options


def string2cli_arg(string) -> str:
    """ Decorates the given string with double quotes. """
    return f'"{string}"'


def call_gdal_util(util_name, src_files, dst_file=None, options=None, gdal_path=None) -> Tuple[bool, str]:
    """
    Call a GDAL utility to run a GDAL operation (see http://www.gdal.org/gdal_utilities.html).

    Parameters
    ----------
    util_name : str
        Pre-defined name of the utility, e.g. "gdal_translate", which converts raster data between different formats,
        potentially performing some operations like sub-settings, resampling, or rescaling pixels.
    src_files : str or list of str
        The name(s) of the source data. It can be either a file path, URL to the data source or a sub-dataset name for
        multi-dataset file.
    dst_file : str, optional
        Full system path to the output file. Defaults to None, i.e. the GDAL utility itself deals with defining a file
        path.
    options : dict, optional
        A dictionary storing additional settings for the process. You can find all options at
        http://www.gdal.org/gdal_utilities.html
    gdal_path : str, optional
        The path where your GDAL utilities are installed. By default, this function tries to look up this path in the
        environment variable "GDAL_UTIL_HOME". If this variable is not set, then `gdal_path` must be provided as a
        key-word argument.

    Returns
    -------
    successful : bool
        True if process was successful.
    output : str
        Console output.

    """
    options = options or dict()

    gdal_path = try_get_gdal_installation_path(gdal_path)

    # prepare the command string
    cmd = []
    gdal_cmd = os.path.join(gdal_path, util_name) if gdal_path else util_name
    cmd.append(string2cli_arg(gdal_cmd))
    cmd.extend(convert_gdal_options_to_command_list(options))

    # add source files and destination file (in double quotation)
    if isinstance(src_files, (tuple, list)):
        src_files_str = " ".join(src_files)
    else:
        src_files_str = string2cli_arg(src_files)
    cmd.append(src_files_str)
    if dst_file is not None:
        cmd.append(string2cli_arg(dst_file))

    output = subprocess.check_output(" ".join(cmd), shell=True, cwd=gdal_path)
    successful = _analyse_gdal_output(str(output))

    return successful, output


def _find_gdal_path() -> str:
    """
    Looks-up the GDAL installation path from the system environment variables, i.e. if "GDAL_UTIL_HOME" is set.

    Returns
    -------
    path : str
        GDAL installation path, where all GDAL utilities are located.

    """
    env_name = "GDAL_UTIL_HOME"
    return os.environ[env_name] if env_name in os.environ else None


def _analyse_gdal_output(output) -> bool:
    """
    Analyses console output from a GDAL utility, i.e. it tries to determine if a process was successful or not.

    Parameters
    ----------
    output : str
        Console output.

    Returns
    -------
    bool :
        True if the process completed success, else False.

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


def _add_scale_option(options, stretch=None, src_nodata=None) -> dict:
    """
    Adds scale (and no data value) settings to dictionary containing command line options for gdal_translate.

    Parameters
    ----------
    options : dict
        Dictionary with gdal_translate command line options.
    src_nodata: int, optional
        No data value of the input image. Defaults to None. If it is not None, then it will be transformed to 0 if it
        is smaller than minimum value range or to 255 if it is larger than the maximum value range.
    stretch : 2-tuple of number, optional
        Minimum and maximum input pixel value to consider for scaling pixels to 0-255. If omitted (default), pixel
        values will be scaled to the minimum and maximum value.

    Returns
    -------
    options : dict
        Dictionary with gdal_translate command line options.

    """
    stretch = stretch or (None, None)
    min_stretch, max_stretch = stretch

    options["-scale"] = ' '
    if (min_stretch is not None) and (max_stretch is not None):
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

    return options


def gen_qlook(src_file, dst_file=None, src_nodata=None, stretch=None, resize_factor=('3%', '3%'),
              ct=None, scale=True, output_format="GTiff", gdal_path=None) -> Tuple[bool, str]:
    """
    Generates a quick-look image.

    Parameters
    ----------
    src_file: str
        The name(s) of the source data. It can be either a file path, URL to the data source or a sub-dataset name for
        multi-dataset file.
    dst_file: str, optional
        Full system path to the output file. If not provided, the output file path will be a combination of `src_file`
        + '_qlook'.
    src_nodata: int, optional
        No data value of the input image. Defaults to None. If it is not None, then it will be transformed to 0 if it
        is smaller than minimum value range or to 255 if it is larger than the maximum value range.
    stretch : 2-tuple of number, optional
        Minimum and maximum input pixel value to consider for scaling pixels to 0-255. If omitted (default), pixel
        values will be scaled to the minimum and maximum value.
    resize_factor : 2-tuple of str, optional
        Size of the output file in pixels and lines, unless '%' is attached in which case it is as a fraction of the
        input image size. The default is ('3%', '3%').
    ct : gdal.ColorTable, optional
        GDAL's color table used for displaying the quick-look data. Defaults to None.
    scale : bool, optional
        If True, pixel values will be scaled to 0-255 if `stretch` is given.
    output_format : str
        Output format of the quick-look. At the moment, 'GTiff' (default) and 'JPEG' are supported.
    gdal_path : str, optional
        The path where your GDAL utilities are installed. By default, this function tries to look up this path in the
        environment variable "GDAL_UTIL_HOME". If this variable is not set, then `gdal_path` must be provided as a
        key-word argument.

    Returns
    -------
    successful : bool
        True if process was successful.
    output : str
        Console output.

    """
    output_format = output_format.lower()
    f_ext = {'gtiff': 'tif',
             'jpeg': 'jpeg'}

    gdal_path = try_get_gdal_installation_path(gdal_path)

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
        options = _add_scale_option(options, stretch=stretch, src_nodata=src_nodata)

    # call gdal_translate to resize input file
    successful, output = call_gdal_util('gdal_translate', src_files=src_file,
                                     dst_file=dst_file, gdal_path=gdal_path,
                                     options=options)

    if (output_format == 'gtiff') and (ct is not None):
        # Update quick look image, attach color table this does not so easily
        # work with JPEG so we keep the colortable of the original file
        ds = gdal.Open(dst_file, GA_Update)
        # set the new color table
        if ds.RasterCount == 1:
            ds.GetRasterBand(1).SetRasterColorTable(ct)

    return successful, output


if __name__ == '__main__':
    pass
