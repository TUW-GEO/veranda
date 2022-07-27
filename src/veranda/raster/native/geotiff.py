""" Manages I/O for a GeoTIFF file. """

import os
import struct
import numpy as np
from osgeo import gdal
from typing import List
from collections import defaultdict
import xml.etree.ElementTree as ET

from veranda.utils import to_list
from veranda.raster.gdalport import NUMPY_TO_GDAL_DTYPE, GDAL_TO_NUMPY_DTYPE


class GeoTiffFile:
    """ GDAL wrapper for reading or writing a GeoTIFF file. """
    def __init__(self, filepath, mode='r', geotrans=(0, 1, 0, 0, 0, 1), sref_wkt=None, raster_shape=None,
                 compression='LZW', metadata=None, is_bigtiff=False, is_tiled=True, blocksize=(512, 512),
                 n_bands=1, dtypes='uint8', scale_factors=1, offsets=0, nodatavals=255, color_tbls=None,
                 color_intprs=None, overwrite=False, auto_decode=False):
        """
        Constructor of `GeoTiffFile`.

        Parameters
        ----------
        filepath : str
            Full file path to a GeoTIFF file.
        mode : str, optional
            File opening mode :
                - 'r' : read (default)
                - 'w' : write
        geotrans : 6-tuple or list, optional
            Geo-transformation parameters with the following entries:
                0: Top-left X coordinate
                1: W-E pixel sampling
                2: Rotation, 0 if image is axis-parallel
                3: Top-left Y coordinate
                4: Rotation, 0 if image is axis-parallel
                5: N-S pixel sampling (negative value if North-up)
            Defaults to (0, 1, 0, 0, 0, 1).
        sref_wkt : str, optional
            Coordinate Reference System (CRS) information in WKT format. Defaults to None.
        raster_shape : 2-tuple, optional
            2D raster_shape of the raster. Defaults to None, i.e. (1, 1).
        compression : str, optional
            Set the compression to use. Defaults to 'LZW'.
        metadata : dict, optional
            Dictionary representing the metadata of the GeoTIFF file. Defaults to None.
        is_bigtiff : bool, optional
            True if the GeoTIFF file should be managed as a 'BIGTIFF' (required if the file will be above 4 GB).
            Defaults to False.
        is_tiled : bool, optional
            True if the data should be tiled (default). False if the data should be stripped.
        blocksize : 2-tuple, optional
            Blocksize of the data blocks in the GeoTIFF file. Defaults to (512, 512).
        n_bands : int, optional
            Number of bands in the GeoTIFF file (relevant when creating a new GeoTIFF file). Defaults to 1.
        dtypes : dict or str, optional
            Data types used for de- or encoding (NumPy-style). Defaults to 'uint8'. It can either be one value (will
            be used for all bands), or a dictionary mapping the band number with the respective data type.
        scale_factors : dict or number, optional
            Scale factor used for de- or encoding. Defaults to 1. It can either be one value (will be used for all
            bands), or a dictionary mapping the band number with the respective scale factor.
        offsets : dict or number, optional
            Offset used for de- or encoding. Defaults to 0. It can either be one value (will be used for all bands),
            or a dictionary mapping the band number with the respective offset.
        nodatavals : dict or int, optional
            No data value used for de- or encoding. Defaults to 255. It can either be one value (will be used for all
            bands), or a dictionary mapping the band number with the respective no data value.
        color_tbls : dict or gdal.ColorTable, optional
            GDAL color tables. Defaults to None. It can either be one value (will be used for all bands), or a
            dictionary mapping the band number with the respective color table.
        color_intprs : dict or gdal.ColorInterp, optional
            GDAL color interpretation value. Defaults to None. It can either be one value (will be used for all bands),
            or a dictionary mapping the band number with the respective color interpretation value.
        overwrite : bool, optional
            Flag if the file can be overwritten if it already exists (defaults to False).
        auto_decode : bool, optional
            True if data should be decoded according to the information available in its header.
            False if not (default).

        """
        self.src = None
        self._driver = gdal.GetDriverByName('GTiff')
        self.filepath = filepath
        self.mode = mode
        self.geotrans = geotrans
        self.sref_wkt = sref_wkt
        self.raster_shape = raster_shape
        self.compression = compression
        self.metadata = dict() if metadata is None else metadata
        self.is_bigtiff = is_bigtiff
        self.is_tiled = is_tiled
        self.blocksize = blocksize
        self.overwrite = overwrite
        self.auto_decode = auto_decode
        self.bands = list(range(1, n_bands + 1))

        dtypes = self.__to_dict(dtypes)
        scale_factors = self.__to_dict(scale_factors)
        offsets = self.__to_dict(offsets)
        nodatavals = self.__to_dict(nodatavals)
        color_tbls = self.__to_dict(color_tbls)
        color_intprs = self.__to_dict(color_intprs)
        self.__set_coding_info_from_input(nodatavals, scale_factors, offsets, dtypes, color_tbls, color_intprs)

        if raster_shape is not None or self.mode == 'r':
            self._open()

    @property
    def n_bands(self) -> int:
        """ Number of bands. """
        return len(self.bands)

    @staticmethod
    def is_file_bigtiff(filepath) -> bool:
        """
        Determines if the given GeoTIFF is a BigTIFF file or not.

        Parameters
        ----------
        filepath : str
            Full file path to a GeoTIFF file.

        Returns
        -------
        bool :
            True if the given file is a BigTIFF, else False.

        """
        with open(filepath, 'rb') as f:
            header = f.read(4)
        byteorder = {b'II': '<', b'MM': '>', b'EP': '<'}[header[:2]]
        version = struct.unpack(byteorder + "H", header[2:4])[0]
        return version == 43

    @property
    def scale_factors(self) -> List[float]:
        """ Scale factors of the different bands. """
        return list(self._scale_factors.values())

    @property
    def offsets(self) -> List[float]:
        """ Offsets of the different bands. """
        return list(self._offsets.values())

    @property
    def nodatavals(self) -> List[int]:
        """ No data values of the different bands. """
        return list(self._nodatavals.values())

    @property
    def color_interps(self) -> List[int]:
        """ Color interpretation values of the different bands. """
        return list(self._color_intprs.values())

    @property
    def color_tables(self) -> List[gdal.ColorTable]:
        """ Color tables of the different bands. """
        return list(self._color_tbls.values())

    @property
    def dtypes(self) -> List[str]:
        """ Data types in NumPy-style format. """
        return [GDAL_TO_NUMPY_DTYPE[dtype] for dtype in self._dtypes.values()]

    def _open(self):
        """
        Helper function supporting the different file modes, i.e. either opening an existing file or creating a new
        file source.

        """
        if self.mode == 'r':
            if not os.path.exists(self.filepath):
                err_msg = f"File '{self.filepath}' does not exist."
                raise FileNotFoundError(err_msg)
            self.src = gdal.Open(self.filepath, gdal.GA_ReadOnly)
            self.raster_shape = self.src.RasterYSize, self.src.RasterXSize
            self.geotrans = self.src.GetGeoTransform()
            self.sref_wkt = self.src.GetProjection()
            self.metadata = self.src.GetMetadata()
            self.blocksize = self.src.GetRasterBand(1).GetBlockSize()  # block seems to be band-independent, because no set function is available per band
            self.compression = self.src.GetMetadata('IMAGE_STRUCTURE').get('COMPRESSION')
            self.is_bigtiff = self.is_file_bigtiff(self.filepath)
            self.is_tiled = self.blocksize[1] == 1
            self.__set_coding_info_from_file()
        elif self.mode == 'w':
            if os.path.exists(self.filepath):
                if self.overwrite:
                    os.remove(self.filepath)
                else:
                    err_msg = f"File '{self.filepath}' exists."
                    raise FileExistsError(err_msg)
            self.__create_driver()
            self.src.SetGeoTransform(self.geotrans)
            if self.sref_wkt is not None:
                self.src.SetProjection(self.sref_wkt)

            if self.metadata is not None:
                self.src.SetMetadata(self.metadata)

            self.__set_bands()
        else:
            err_msg = f"Mode '{self.mode}' not known."
            raise ValueError(err_msg)

        if self.src is None:
            err_msg = f"Open failed: {self.filepath}"
            raise IOError(err_msg)

    def read(self, row=0, col=0, n_rows=None, n_cols=None, bands=None, decoder=None, decoder_kwargs=None) -> dict:
        """
        Read data from a GeoTIFF file.

        Parameters
        ----------
        row : int, optional
            Row number/index. Defaults to 0.
        col : int, optional
            Column number/index. Defaults to 0.
        n_rows : int, optional
            Number of rows of the reading window (counted from `row`). If None (default), then the full extent of
            the raster is used.
        n_cols : int, optional
            Number of columns of the reading window (counted from `col`). If None (default), then the full extent of
            the raster is used.
        bands : int or tuple or list, optional
            Band numbers of the GeoTIFF file to read data from. Defaults to None, i.e. all available bands will be
            used.
        decoder : callable, optional
            Decoding function expecting a NumPy array as input.
        decoder_kwargs : dict, optional
            Keyword arguments for the decoder.

        Returns
        -------
        data : dict
            Dictionary mapping band numbers to NumPy arrays read from disk.

        """

        decoder_kwargs = decoder_kwargs or dict()
        n_rows = self.raster_shape[0] if n_rows is None else n_rows
        n_cols = self.raster_shape[1] if n_cols is None else n_cols
        bands = bands or self.bands
        bands = to_list(bands)

        data = {band: self._read_band(band, col, row, n_cols, n_rows, decoder, decoder_kwargs) for band in bands}

        return data

    def write(self, data, row=0, col=0, encoder=None, encoder_kwargs=None):
        """
        Writes a NumPy array to a GeoTIFF file.

        Parameters
        ----------
        data : dict of np.array or np.array
            Dictionary mapping band numbers with a 2D NumPy array or 3D numpy array, where the first dimension must
            correspond to the number of bands.
        row : int, optional
            Offset row number/index (defaults to 0).
        col : int, optional
            Offset column number/index (defaults to 0).
        encoder : callable, optional
            Function allowing to encode a NumPy array before writing it to disk.
        encoder_kwargs : dict, optional
            Keyword arguments for the encoder.

        """
        if self.mode != 'w':
            err_msg = "Wrong mode for writing a GeoTIFF file (use 'w')."
            raise IOError(err_msg)

        encoder_kwargs = encoder_kwargs or dict()
        if not isinstance(data, dict):
            data_write = dict()
            for i, band in enumerate(self.bands):
                data_write[band] = data[i, ...]
        else:
            data_write = data

        data_bands = list(data_write.keys())
        data_shape = data_write[data_bands[0]].shape
        if self.src is None:
            self.raster_shape = data_shape
            self._open()

        for band in data_bands:
            self._write_band(band, col, row, data_write[band], encoder, encoder_kwargs)

    def _read_band(self, band, col, row, n_cols, n_rows, decoder, decoder_kwargs) -> np.ndarray:
        """
        Reads and decodes data per band.

        Parameters
        ----------
        band : int
            Band number.
        row : int
            Row number/index
        col : int
            Column number/index
        n_rows : int
            Number of rows of the reading window (counted from `row`).
        n_cols : int
            Number of columns of the reading window (counted from `col`).
        decoder : callable
            Decoding function expecting a NumPy array as input.
        decoder_kwargs : dict
            Keyword arguments for the decoder.

        Returns
        -------
        band_data : np.ndarray
            Decoded band data.

        """
        band = int(band)
        band_data = self.src.GetRasterBand(band).ReadAsArray(col, row, n_cols, n_rows)
        scale_factor = self._scale_factors[band]
        nodataval = self._nodatavals[band]
        offset = self._offsets[band]
        if self.auto_decode:
            band_data = band_data.astype(float)
            band_data[band_data == nodataval] = np.nan
            band_data = band_data * scale_factor + offset
        else:
            if decoder is not None:
                dtype = GDAL_TO_NUMPY_DTYPE(self._dtypes[band])
                band_data = decoder(band_data, nodataval=nodataval, band=band, scale_factor=scale_factor,
                                    offset=offset, dtype=dtype, **decoder_kwargs)

        return band_data

    def _write_band(self, band, col, row, data, encoder, encoder_kwargs):
        """
        Writes and encodes data per band.

        Parameters
        ----------
        band : int
            Band number.
        col : int
            Offset column number/index.
        row : int
            Offset row number/index.
        data : np.ndarray
            2D NumPy array.
        encoder : callable, optional
            Function allowing to encode a NumPy array before writing it to disk.
        encoder_kwargs : dict, optional
            Keyword arguments for the encoder.

        """
        nodataval = self._nodatavals[band]
        scale_factor = self._scale_factors[band]
        offset = self._offsets[band]
        if encoder is not None:
            dtype = GDAL_TO_NUMPY_DTYPE(self._dtypes[band])
            self.src.GetRasterBand(band).WriteArray(encoder(data,
                                                            band=band,
                                                            nodataval=nodataval,
                                                            scale_factor=scale_factor,
                                                            offset=offset,
                                                            dtype=dtype,
                                                            **encoder_kwargs),
                                                    xoff=col, yoff=row)
        else:
            self.src.GetRasterBand(band).WriteArray(data, xoff=col, yoff=row)

    def __set_coding_info_from_input(self, nodatavals, scale_factors, offsets, dtypes, color_tbls, color_intprs):
        """
        Sets/overwrites internal dictionaries used to store all the coding information applied during en- or decoding
        with externally provided arguments.

        Parameters
        ----------
        nodatavals : dict
            Maps the band number with the respective no data value used for de- or encoding.
        scale_factors : dict
            Maps the band number with the respective scale factor used for de- or encoding.
        offsets : dict
            Maps the band number with the respective offset used for de- or encoding.
        dtypes : dict
            Maps the band number with the respective data type used for de- or encoding (NumPy-style).
        color_tbls : dict
           Maps the band number with the respective GDAL color table (`gdal.ColorTable`).
        color_intprs : dict or gdal.ColorInterp, optional
            Maps the band number with the respective GDAL color interpretation value (`gdal.ColorInterp`).

        """
        self._scale_factors = dict()
        self._offsets = dict()
        self._nodatavals = dict()
        self._color_tbls = dict()
        self._color_intprs = dict()
        self._dtypes = dict()
        for band in self.bands:
            self._dtypes[band] = NUMPY_TO_GDAL_DTYPE[dtypes.get(band, 'uint8')]
            self._scale_factors[band] = scale_factors.get(band, 1)
            self._offsets[band] = offsets.get(band, 0)
            self._nodatavals[band] = nodatavals.get(band, 255)
            self._color_tbls[band] = color_tbls.get(band, None)
            self._color_intprs[band] = color_intprs.get(band, None)

    def __set_coding_info_from_file(self):
        """
        Sets/overwrites internal dictionaries used to store all the coding information applied during en- or decoding
        with coding information retrieved from a GDAL dataset.

        """
        self.bands = []
        self._scale_factors = dict()
        self._offsets = dict()
        self._nodatavals = dict()
        self._color_tbls = dict()
        self._color_intprs = dict()
        self._dtypes = dict()
        for band in range(1, self.src.RasterCount + 1):
            self.bands.append(band)
            scale_factor = self.src.GetRasterBand(band).GetScale()
            offset = self.src.GetRasterBand(band).GetOffset()
            self._scale_factors[band] = scale_factor or 1
            self._offsets[band] = offset or 0
            self._nodatavals[band] = self.src.GetRasterBand(band).GetNoDataValue()
            self._color_tbls[band] = self.src.GetRasterBand(band).GetColorTable()
            self._color_intprs[band] = self.src.GetRasterBand(band).GetColorInterpretation()
            self._dtypes[band] = self.src.GetRasterBand(band).DataType

    def __create_driver(self):
        """ Creates a new GDAL dataset/driver. """
        gdal_opt = dict()
        gdal_opt['COMPRESS'] = self.compression
        gdal_opt['TILED'] = 'YES' if self.is_tiled else 'NO'
        gdal_opt['BLOCKXSIZE'] = str(self.blocksize[0])
        gdal_opt['BLOCKYSIZE'] = str(self.blocksize[1])
        gdal_opt['BIGTIFF'] = 'YES' if self.is_bigtiff else 'NO'
        gdal_opt = ['='.join((k, v)) for k, v in gdal_opt.items()]
        self.src = self._driver.Create(self.filepath, self.raster_shape[1], self.raster_shape[0],
                                       self.n_bands, NUMPY_TO_GDAL_DTYPE[self.dtypes[0]],
                                       options=gdal_opt)

    def __set_bands(self):
        """ Sets band attributes, i.e. default/fill value, no data value, scale factor, and offset. """
        for band in self.bands:
            self.src.GetRasterBand(band).Fill(int(self._nodatavals[band]))
            self.src.GetRasterBand(band).SetNoDataValue(float(self._nodatavals[band]))
            self.src.GetRasterBand(band).SetScale(self._scale_factors[band])
            self.src.GetRasterBand(band).SetOffset(self._offsets[band])

    def __to_dict(self, arg) -> dict:
        """
        Assigns non-iterable object to a dictionary with band numbers as keys. If `arg` is already a dictionary the
        same object is returned.

        Parameters
        ----------
        arg : non-iterable or dict
            Non-iterable, which should be converted to a dict.

        Returns
        -------
        arg_dict : dict
            Dictionary mapping band numbers with the value of `arg`.

        """
        return {band: arg for band in self.bands} if not isinstance(arg, dict) else arg

    def flush(self):
        """
        Flush data on disk.
        """
        if self.src is not None:
            self.src.FlushCache()

    def close(self):
        """
        Close the dataset.
        """
        self.src = None

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()


def create_vrt_file(filepaths, vrt_filepath, shape, sref_wkt, geotrans, bands=1):
    """
    Creates a VRT file stack from a list of file paths.

    Parameters
    ----------
    filepaths : list of str
        Full system path to the files to stack.
    vrt_filepath : str
        Full system path to the VRT file to create.
    shape : 2-tuple
        Shape (rows, columns) of the raster stack.
    sref_wkt : str
        Coordinate Reference System (CRS) information in WKT format.
    geotrans : 6-tuple or list, optional
        Geo-transformation parameters with the following entries:
            0: Top-left X coordinate
            1: W-E pixel sampling
            2: Rotation, 0 if image is axis-parallel
            3: Top-left Y coordinate
            4: Rotation, 0 if image is axis-parallel
            5: N-S pixel sampling (negative value if North-up)
    bands : tuple or list or int, optional
        Band number(s). Defaults to 1.

    """
    n_filepaths = len(filepaths)
    bands = to_list(bands)
    n_bands = len(bands)
    n_rows, n_cols = shape

    ref_filepath = filepaths[0]
    band_attr_dict = defaultdict(list)
    with GeoTiffFile(ref_filepath, 'r') as gt_file:
        for band in bands:
            band_attr_dict = _read_band_attributes(gt_file, band, band_attr_dict)

    attrib = {"rasterXSize": str(n_cols), "rasterYSize": str(n_rows)}
    vrt_root = ET.Element("VRTDataset", attrib=attrib)

    geot_elem = ET.SubElement(vrt_root, "GeoTransform")
    geot_elem.text = ",".join(map(str, geotrans))

    geot_elem = ET.SubElement(vrt_root, "SRS")
    geot_elem.text = sref_wkt

    entry_idx = 1
    for f_idx in range(n_filepaths):
        filepath = filepaths[f_idx]
        for band_idx in range(n_bands):
            _fill_vrt_file_per_band(vrt_root, filepath, bands, band_idx, entry_idx, n_cols, n_rows, band_attr_dict)
            entry_idx += 1

    tree = ET.ElementTree(vrt_root)
    tree.write(vrt_filepath, encoding="UTF-8")


def _read_band_attributes(gt_file, band, attr_dict) -> dict:
    """
    Updates a GeoTIFF attribute dictionary for a specific band with information on no data value, scale factor,
    offset, data type, and block size.

    Parameters
    ----------
    gt_file : GeoTiffFile
        Pointer to an open GeoTIFF file.
    band : int
        Band number.
    attr_dict :
        Attributes dictionary to fill/update.

    Returns
    -------
    attr_dict : dict
        Updated attributes dictionary.

    """
    b_idx = list(gt_file.bands).index(band)
    attr_dict['nodataval'].append(gt_file.nodatavals[b_idx])
    attr_dict['scale_factor'].append(gt_file.scale_factors[b_idx])
    attr_dict['offset'].append(gt_file.offsets[b_idx])
    attr_dict['dtype'].append(gdal.GetDataTypeName(NUMPY_TO_GDAL_DTYPE[gt_file.dtypes[b_idx]]))
    attr_dict['blocksize'].append(gt_file.blocksize)
    return attr_dict


def _fill_vrt_file_per_band(et_root, filepath, bands, band_idx, entry_idx, n_cols, n_rows, band_attr_dict):
    """
    Writes all band-relevant information to a new entry in the element tree representing the VRT file, in-place.

    Parameters
    ----------
    et_root : ET.Element
        Root element of element tree/VRT file.
    filepath : str
        Full system file path to GeoTIFF file.
    bands : list of int
        Band numbers.
    band_idx : int
        Index for the current band.
    entry_idx : int
        Overall index keeping track of the number of entries in the VRT file.
    n_cols : int
        Number of columns of the GeoTIFF file.
    n_rows : int
        Number of rows of the GeoTIFF file.
    band_attr_dict : dict
        Dictionary storing coding information for a GeoTIFF band.

    """
    band = bands[band_idx]
    attrib = {"dataType": band_attr_dict['dtype'][band_idx], "band": str(entry_idx)}
    band_elem = ET.SubElement(et_root, "VRTRasterBand", attrib=attrib)
    simsrc_elem = ET.SubElement(band_elem, "SimpleSource")
    attrib = {"relativetoVRT": "0"}
    file_elem = ET.SubElement(simsrc_elem, "SourceFilename", attrib=attrib)
    file_elem.text = filepath
    ET.SubElement(simsrc_elem, "SourceBand").text = str(band)

    attrib = {"RasterXSize": str(n_cols), "RasterYSize": str(n_rows),
              "DataType": band_attr_dict['dtype'][band_idx],
              "BlockXSize": str(band_attr_dict['blocksize'][band_idx][0]),
              "BlockYSize": str(band_attr_dict['blocksize'][band_idx][1])}

    file_elem = ET.SubElement(simsrc_elem, "SourceProperties", attrib=attrib)

    scale_factor = band_attr_dict['scale_factor'][band_idx]
    scale_factor = 1 if scale_factor is None else scale_factor
    ET.SubElement(band_elem, "NodataValue").text = str(band_attr_dict['nodataval'][band_idx])
    ET.SubElement(band_elem, "Scale").text = str(scale_factor)
    ET.SubElement(band_elem, "Offset").text = str(band_attr_dict['offset'][band_idx])


if __name__ == '__main__':
    pass
