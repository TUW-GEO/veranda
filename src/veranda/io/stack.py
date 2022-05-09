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
import copy
import warnings
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime

from osgeo import gdal
import numpy as np
import pandas as pd
import xarray as xr
import netCDF4

from veranda.io.geotiff import GeoTiffFile
from veranda.raster.driver.netcdf import NcFile

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


# TODO: define abstract base IO class

class GeoTiffRasterStack:

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
    geotrans : tuple or list
        Geotransform parameters (default (0, 1, 0, 0, 0, 1)).
        0: Top left x
        1: W-E pixel resolution
        2: Rotation, 0 if image is "north up"
        3: Top left y
        4: Rotation, 0 if image is "north up"
        5: N-S pixel resolution (negative value if North up)
    sref : str
        Coordinate Reference System (CRS) in Wkt form (default None).
    tags : dict, optional
        Meta data tags (default None).
    gdal_opt : dict, optional
        Driver specific control parameters (default None).
    auto_decode : bool, optional
        If true, when reading ds, "scale_factor" and "add_offset" is applied (default is False).
        ATTENTION: Also the xarray dataset may applies encoding/scaling what
                can mess up things
    """

    def __init__(self, inventory=None, mode='r',
                 compression='LZW', blockxsize=512, blockysize=512,
                 geotrans=(0, 1, 0, 0, 0, 1), sref=None,
                 tags=None, gdal_opt=None, auto_decode=False):

        self.inventory = inventory
        self.mode = mode
        self.geotrans = geotrans
        self.sref = sref
        self.shape = None
        self.metadata = tags['metadata'] if tags is not None and 'metadata' in tags.keys() else None

        self.vrt = None
        self.compression = compression
        self.blockxsize = blockxsize
        self.blockysize = blockysize
        self.tags = tags
        self.gdal_opt = gdal_opt
        self.auto_decode = auto_decode

        self._open()

    def _open(self):

        if self.inventory is not None:
            ref_filepath = self.inventory[self.inventory.notnull()]['filepath'][0]
            with GeoTiffFile(ref_filepath, mode='r') as geotiff:
                self.sref = geotiff.sref
                self.geotrans = geotiff.geotrans
                self.metadata = geotiff.metadata
                self.dtype = geotiff.dtype
                self.shape = (len(self.inventory), geotiff.shape[-2], geotiff.shape[-1])

            self.vrt = self._build_stack(self.inventory)

    def _build_stack(self, inventory):
        """
        Building vrt stack.
        """
        if inventory is not None:
            path = tempfile.gettempdir()
            date_str = datetime.utcnow().strftime("%Y%m%d%H%M%S")
            tmp_filename = "{:}.vrt".format(date_str)
            vrt_filepath = os.path.join(path, tmp_filename)
            filepaths = inventory.dropna()['filepath'].to_list()
            bands = 1 if "band" not in inventory.keys() else inventory.dropna()['band'].to_list()  # take first band as a default
            nodatavals = -9999 if "nodataval" not in inventory.keys() else inventory.dropna()['nodataval'].to_list() # take -9999 as a default no data value
            scale_factors = 1 if "scale_factor" not in inventory.keys() else inventory.dropna()['scale_factor'].to_list()
            add_offsets = 1 if "add_offset" not in inventory.keys() else inventory.dropna()['add_offset'].to_list()
            create_vrt_file(vrt_filepath, filepaths, bands=bands, geotrans=self.geotrans,
                            shape=(self.shape[1], self.shape[2]), dtype=self.dtype, sref=self.sref,
                            nodataval=nodatavals, scale_factor=scale_factors, add_offset=add_offsets)
            return gdal.Open(vrt_filepath, gdal.GA_ReadOnly)
        else:
            raise RuntimeError('Building VRT stack failed')

    def read(self, row=None, col=None, n_rows=1, n_cols=1, band=1, nodataval=-9999,
             decoder=None, decoder_kwargs=None, idx=None):
        """
        Read data from a VRT raster file stack.

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
        bands : int or list of int, optional
            Band numbers (starting with 1). If None, all bands will be read.
        nodatavals : tuple or list, optional
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
        decoder_kwargs = {} if decoder_kwargs is None else decoder_kwargs

        inventory = copy.deepcopy(self.inventory)
        rebuild_vrt = False
        if band != 1:
            inventory['band'] = band
            rebuild_vrt = True

        if idx is not None:
            inventory = inventory.loc[[idx]]
            rebuild_vrt = True

        if rebuild_vrt:  # things have changed in the inventory, recreated vrt file
            vrt = self._build_stack(inventory)
        else:
            vrt = self.vrt

        if row is None and col is not None:
            row = 0
            n_cols = self.shape[2]
        elif row is not None and col is None:
            col = 0
            n_rows = self.shape[1]

        if row is None and col is None:
            data = vrt.ReadAsArray()
        else:
            data = vrt.ReadAsArray(col, row, n_cols, n_rows)

        if self.auto_decode:
            data = self._auto_decode(data, band, vrt)
        else:
            if decoder is not None:
                data = decoder(data, nodataval=nodataval, **decoder_kwargs)

        return self._fill_nan(data)

    def _auto_decode(self, data, band=1, vrt=None):
        """
        Applies auto-decoding (if activated) to data related to a specific band.

        Parameters
        ----------
        data : np.array
            Data related to `band`.
        band : int, optional
            Band number (defaults to 1).

        Returns
        -------
        data : np.array
            Decoded data if auto-decoding is activated.

        """
        vrt = vrt if vrt is not None else self.vrt
        if self.auto_decode:
            scale_factor = vrt.GetRasterBand(band).GetScale()
            offset = vrt.GetRasterBand(band).GetOffset()
            nodata = vrt.GetRasterBand(band).GetNoDataValue()
            if (scale_factor != 1.) and (offset != 0.):
                data = data.astype(float)
                data[data == nodata] = np.nan
                data = data * scale_factor + offset
            else:
                wrn_msg = "Automatic decoding is activated for band '{}', but attribute 'scale' and 'offset' " \
                          "are missing!".format(band)
                warnings.warn(wrn_msg)
        return data

    def _fill_nan(self, data):
        """
        Extends data set with nan values where no file paths are available in the inventory.

        Parameters
        ----------
        data : numpy.ndarray
            3D NumPy data set.

        Returns
        -------
        numpy.ndarray
            3D NumPy data set and NaN values where no file path is given in the inventory.
        """
        if None in self.inventory['filepath'].to_list():
            n_entries = len(self.inventory)
            ext_data = np.ones((n_entries, data.shape[-2], data.shape[-1]))*np.nan
            # get indexes of non nan/None data layers
            idxs = np.arange(n_entries)[self.inventory['filepath'].notnull()]
            ext_data[idxs, :, :] = data
            return ext_data
        else:
            return data

    def write(self, data, filepath, band=None, encoder=None, nodataval=None, encoder_kwargs=None, ct=None):
        """
        Write data into raster file.

        Parameters
        ----------
        data : numpy.ndarray (3d)
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
        ct : tuple or list, optional
            List of color tables for each band.
            Default: No color tables set.
        """
        if data.ndim != 3:
            err_msg = "Array needs to have 3 dimensions [band, width, height]"
            raise ValueError(err_msg)

        with GeoTiffFile(filepath, mode='w') as geotiff:
            geotiff.write(data, band=band, encoder=encoder, nodataval=nodataval, encoder_kwargs=encoder_kwargs,
                          ct=ct)

    def write_from_xr(self, ds, dir_path):
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

        timestamps = time_var.strftime('%Y%m%d%H%M%S')
        var_dict = {}

        for name in ds.variables:
            if name == 'time':
                continue
            ds_attrs = ds[name].attrs.keys()
            nodata = [ds[name].attrs["fill_value"]] if "fill_value" in ds_attrs else [-9999]
            scale_factor = [ds[name].attrs["scale_factor"]] if "scale_factor" in ds_attrs else [1.]
            add_offset = [ds[name].attrs["add_offset"]] if "add_offset" in ds_attrs else [0.]

            filepaths = []
            for i, timestamp in enumerate(timestamps):
                filename = '{:}_{:}'.format(timestamp, name)
                filepath = os.path.join(dir_path, '{:}.tif'.format(filename))
                filepaths.append(filepath)

                with GeoTiffFile(filepath, mode='w', n_bands=1) as src:
                    src.write(ds[name].isel(time=i).values, band=1, nodataval=nodata,
                              scale_factor=scale_factor, add_offset=add_offset)

            bands = [1] * len(filepaths)
            var_dict[name] = pd.DataFrame({'filepath': filepaths,
                                           'band': bands,
                                           'nodataval': nodata,
                                           'scale_factor': scale_factor,
                                           'add_offset': add_offset}, index=ds['time'].to_index())

        return var_dict

    def iter_img(self):
        """
        Iterate over image stack.
        """

        for i, time_stamp in enumerate(self.inventory.index):
            data = self.vrt.GetRasterBand(i+1).ReadAsArray()
            band = 1
            if 'band' in self.inventory.columns:
                band = self.inventory.iloc[i]['band']

            yield time_stamp, self._auto_decode(data, band)

    def export_to_nc(self, path, band='data', **kwargs):
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

        dims = ['time', 'y', 'x']
        timestamps = []
        filepaths = []

        with NcRasterStack(mode='w', geotrans=self.geotrans, sref=self.sref, **kwargs) as rts:
            for timestamp, data in self.iter_img():
                coords = {'time': [timestamp]}
                ds = xr.Dataset({band:
                                     (dims, data[np.newaxis, :, :])},
                                coords=coords)
                inventory = rts.write_netcdfs(ds, path)

                timestamps.append(timestamp)
                filepaths.extend(inventory['filepath'].tolist())

        return pd.DataFrame({'filepath': filepaths, 'time': timestamps})

    def close(self):
        """
        """
        if self.vrt is not None:
            self.vrt = None

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()


def create_vrt_file(vrt_filepath, filepaths, nodataval=None, scale_factor=None, add_offset=None, bands=1,
                    sref=None, geotrans=None, shape=None, dtype=None):
    """
    Create a .VRT XML file. First file is used as master file.

    Parameters
    ----------
    vrt_filepath : str
        VRT filename.
    filepaths : list
        List of files to include in the VRT.
    nodata : float, optional
        No data value (default: None).
    band : int, optional
        Band of the input file (default: 1)
    """
    n_filepaths = len(filepaths)
    if not isinstance(bands, list):
        bands = [bands] * n_filepaths
    else:
        n_bands = len(bands)
        if n_bands != n_filepaths:
            err_msg = "Number of bands ({}) and number of file paths ({}) mismatch.".format(n_bands, n_filepaths)
            raise ValueError(err_msg)

    # if one of these attributes is None take the first file to get the metadata from
    load_md = any([elem is None for elem in [sref, geotrans, shape, dtype,
                                             nodataval, scale_factor, add_offset]])
    if load_md:
        src = gdal.Open(filepaths[0])
        if sref is None:
            sref = src.GetProjection()
        if geotrans is None:
            geotrans = src.GetGeoTransform()
        if shape is None:
            shape = (src.RasterYSize, src.RasterXSize)
        if dtype is None:
            dtype = r_gdal_dtype[src.GetRasterBand(bands[0]).DataType]
        if scale_factor:
            scale_factor = src.GetRasterBand(bands[0]).GetScale()
        if add_offset is None:
            add_offset = src.GetRasterBand(bands[0]).GetOffset()
        if nodataval is None:
            nodataval = src.GetRasterBand(bands[0]).GetNoDataValue()
        src = None

    n_rows, n_cols = shape

    if not isinstance(nodataval, list):
        nodataval = [nodataval] * n_filepaths

    if not isinstance(scale_factor, list):
        scale_factor = [scale_factor] * n_filepaths

    if not isinstance(add_offset, list):
        add_offset = [add_offset] * n_filepaths

    attrib = {"rasterXSize": str(n_cols), "rasterYSize": str(n_rows)}
    vrt_root = ET.Element("VRTDataset", attrib=attrib)

    geot_elem = ET.SubElement(vrt_root, "GeoTransform")
    geot_elem.text = ",".join(map(str, geotrans))

    geot_elem = ET.SubElement(vrt_root, "SRS")
    geot_elem.text = sref

    for i in range(n_filepaths):
        filepath = filepaths[i]
        attrib = {"dataType": dtype.lower(), "band": str(i + 1)}
        band_elem = ET.SubElement(vrt_root, "VRTRasterBand", attrib=attrib)
        simsrc_elem = ET.SubElement(band_elem, "SimpleSource")
        attrib = {"relativetoVRT": "0"}
        file_elem = ET.SubElement(simsrc_elem, "SourceFilename", attrib=attrib)
        file_elem.text = filepath
        ET.SubElement(simsrc_elem, "SourceBand").text = str(bands[i])

        attrib = {"RasterXSize": str(n_cols), "RasterYSize": str(n_rows),
                  "DataType": dtype.lower(), "BlockXSize": str(512),
                  "BlockYSize": str(512)}

        file_elem = ET.SubElement(simsrc_elem, "SourceProperties",
                                  attrib=attrib)

        ET.SubElement(band_elem, "NodataValue").text = str(nodataval[i])
        ET.SubElement(band_elem, "Scale").text = str(scale_factor[i])
        ET.SubElement(band_elem, "Offset").text = str(add_offset[i])

    tree = ET.ElementTree(vrt_root)
    tree.write(vrt_filepath, encoding="UTF-8")


class NcRasterStack:

    """
    A NcRasterTimeStack is a collection of NetCDF files, which together
    represent a time series stack of raster data.

    Parameters
    ----------
    mode : str, optional
        File stack opening mode ('r' read, 'w' write). Default: read
    inventory : pandas.DataFrame, required for mode = 'r'
        paImage stack files.
    out_path : str, required if mode = 'w'
        Output file path.
    stack_size : str, required if mode = 'w'
        Stack size specification. Default: '%Y%W', i.e. weekly stacks
        Other possibles: '%Y' yearly stacks, '%Y%m' monthly stacks,
        '%Y%m%d' daily stacks, '%Y%m%d%H' hourly stacks, etc.
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
    """

    def __init__(self, mode='r', inventory=None, geotrans=(0, 1, 0, 0, 0, 1),
                 sref=None, compression=2, chunksizes=None,
                 chunks=None, time_units="days since 1900-01-01 00:00:00", auto_decode=False):

        self.mfdataset = None
        self.shape = None
        self._dims = None
        self.mode = mode
        self.inventory = inventory
        if self.inventory is not None and self.inventory.index.name is None:
            self.inventory.index.name = 'time'

        self.geotrans = geotrans
        self.sref = sref
        self.compression = compression
        self.chunksizes = chunksizes
        self.chunks = chunks
        self.time_units = time_units
        self.auto_decode = auto_decode

        self._open()

    def _open(self):

        if self.inventory is not None:
            ref_filepath = self.inventory[self.inventory.notnull()]['filepath'][0]
            with NcFile(ref_filepath, mode='r') as netcdf:
                self.sref = netcdf.sref
                self.geotrans = netcdf.geotrans
                self.metadata = netcdf.metadata
                self.shape = (len(self.inventory), netcdf.shape[-2], netcdf.shape[-1])
                self._dims = len(netcdf.shape)

            self._build_stack()

    def _build_stack(self):
        """
        Building file stack and initialize netCDF4.mfdataset.
        """
        if self.inventory is not None:
            if self._dims == 2:
                self.mfdataset = xr.open_mfdataset(self.inventory.dropna()['filepath'].tolist(),
                                                   chunks=self.chunks,
                                                   combine="nested",
                                                   concat_dim=self.inventory.index.name,
                                                   mask_and_scale=self.auto_decode, use_cftime=False)
                self.mfdataset = self.mfdataset.assign_coords({self.inventory.index.name: self.inventory.index})
                gm_name = NcFile.get_gm_name(self.mfdataset)
                if gm_name is not None:
                    self.mfdataset[gm_name] = self.mfdataset[gm_name].sel(**{self.inventory.index.name: 0}, drop=True)
            else:
                self.mfdataset = xr.open_mfdataset(
                    self.inventory.dropna()['filepath'].tolist(), chunks=self.chunks, combine='by_coords',
                    mask_and_scale=self.auto_decode, use_cftime=False)
        else:
            raise RuntimeError('Building stack failed')

    def read(self, row=None, col=None, n_rows=1, n_cols=1, band="1", nodataval=-9999, decoder=None,
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
        band : str or list of str, optional
            Band numbers/names. If None, all bands will be read.
        nodataval : tuple or list, optional
            List of no data values for each band.
            Default: -9999 for each band.
        decoder : function, optional
            Decoding function expecting a NumPy array as input.
        decoder_kwargs : dict, optional
            Keyword arguments for the decoder.

        Returns
        -------
        data : xarray.Dataset
            Data set with the dimensions [time, y, x] and one data variable.
        """

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
            slices = (slice(None), slice(row, row + n_rows), slice(col, col + n_cols))
        else:
            slices = (slice(row, row + n_rows), slice(col, col + n_cols))

        data_ar = self.mfdataset[band][slices]
        if decoder:
           data_ar.data = decoder(data_ar.data, nodataval, **decoder_kwargs)
        data = data_ar.to_dataset()

        if 'time' in list(data.dims.keys()) and data.variables['time'].dtype == 'float':
            timestamps = netCDF4.num2date(data['time'], self.time_units, only_use_cftime_datetimes=False)
            data = data.assign_coords({'time': timestamps})

        # add projection informations again
        gm_name = NcFile.get_gm_name(self.mfdataset)
        if gm_name is not None:
            data[gm_name] = self.mfdataset[gm_name]
        
        #add attributes
        data.attrs=self.mfdataset.attrs

        return self._fill_nan(data)

    # ToDO: rechunk?
    def _fill_nan(self, data):
        """
        Extends data set with nan values where data and inventory time stamps mismatch.

        Parameters
        ----------
        data : xarray.Dataset
            Data set with the dimensions [time, y, x].

        Returns
        -------
        xarray.Dataset
            Data set with the dimensions [time, y, x] and NaN values where no filepath is given in the inventory.

        Notes
        -----
        Sorting along time is applied.
        """
        if None in self.inventory['filepath'].to_list():
            timestamps = data['time'].to_index().tolist()
            nan_timestamps = self.inventory.index[~self.inventory['filepath'].notnull()].to_list()  # get all timestamps which do not contain data
            timestamps.extend(nan_timestamps)
            return data.reindex(time=sorted(timestamps))
        else:
            return data

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

        for i in range(self.mfdataset[var_name].shape[0]):
            timestamp = netCDF4.num2date(self.mfdataset['time'][i].values, self.time_units,
                                         only_use_cftime_datetimes=False)
            yield timestamp, self.mfdataset[var_name][i, :, :]

    def write_netcdfs(self, ds, dir_path, stack_size="%Y%m", fn_prefix='', fn_suffix='.nc'):
        
        #inclusive left, exclusive right
        #stacks are smaller than 1D
        if any(x in stack_size for x in ['H','min','T']):
            dup_stack_filenames = ds['time'].to_index().floor(stack_size)
        else:
            dup_stack_filenames = ds['time'].to_index().strftime(stack_size)

        stack_filenames, index = np.unique(dup_stack_filenames, return_index=True)
        index = np.hstack((index, len(dup_stack_filenames)))

        filepaths = []
        timestamps = []
        for i, stack_filename in enumerate(stack_filenames):
            time_sel = np.arange(index[i], index[i + 1])

            if any(x in stack_size for x in ['H','min','T']):
                timestamp = ds['time'][[index[i]]].to_index().floor(stack_size)[0].to_datetime64()
                stack_filename = pd.to_datetime(str(stack_filename)).strftime('%Y%m%d_%H%M%S')
            else:
                timestamp = datetime.strptime(ds['time'][[index[i]]].to_index().strftime(stack_size)[0], stack_size)
            timestamps.append(timestamp)
            filename = '{:}{:}{:}'.format(fn_prefix, stack_filename, fn_suffix)
            filepath = os.path.join(dir_path, filename)
            filepaths.append(filepath)

            if os.path.exists(filepath):
                mode = 'a'
            else:
                mode = 'w'

            with NcFile(filepath, mode=mode,
                        complevel=self.compression,
                        geotrans=self.geotrans,
                        sref=self.sref,
                        chunksizes=self.chunksizes) as nc:
                nc.write(ds.isel(time=time_sel))

        return pd.DataFrame({'filepath': filepaths}, index=timestamps)

    def write(self, ds, filepath, band=None, encoder=None, nodataval=None, encoder_kwargs=None, auto_scale=False):
        """
        Write data set into raster time stack.

        Parameters
        ----------
        ds : xarray.Dataset
            Input data set.
        """
        if os.path.exists(filepath):
            mode = 'a'
        else:
            mode = 'w'

        with NcFile(filepath, mode=mode,
                    complevel=self.compression,
                    geotrans=self.geotrans,
                    sref=self.sref,
                    chunksizes=self.chunksizes) as nc:
            nc.write(ds, band=band, nodataval=nodataval, encoder=encoder,  encoder_kwargs=encoder_kwargs)

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

    def export_to_tif(self, path, band, **kwargs):
        """
        Export to Geo tiff files.

        Parameters
        ----------
        path : str
            Output path.
        band : str
            Variable name.

        Returns
        -------
        file_list : pandas.DataFrame
            Geotiff file names with time stamps as index.
        """
        dims = ['time', 'y', 'x']
        timestamps = []
        filepaths = []

        with GeoTiffRasterStack(mode='w', geotrans=self.geotrans, sref=self.sref, **kwargs) as stack:
            for timestamp, data in self.iter_img(band):
                coords = {'time': [timestamp]}
                ds = xr.Dataset(
                    {band: (dims, data.values[np.newaxis, :, :])},
                    coords=coords)
                inventory = stack.write_from_xr(ds, path)

                timestamps.append(timestamp)
                filepaths.extend(inventory[band]['filepath'].tolist())

        bands = [1]*len(filepaths)
        return pd.DataFrame({'filepath': filepaths, 'band': bands}, index=timestamps)
