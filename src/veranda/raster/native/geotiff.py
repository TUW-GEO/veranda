import os
import struct
import numpy as np
from osgeo import gdal
import xml.etree.ElementTree as ET

from veranda.utils import to_list
from veranda.raster.gdalport import NUMPY_TO_GDAL_DTYPE, GDAL_TO_NUMPY_DTYPE


class GeoTiffFile:
    """ GDAL wrapper for reading and writing GeoTIFF files. """
    def __init__(self, filepath, mode='r', geotrans=(0, 1, 0, 0, 0, 1), sref_wkt=None, shape=None,
                 compression='LZW', metadata=None, is_bigtiff=False, is_tiled=True, blocksize=(512, 512),
                 n_bands=1,  dtypes='uint8', scale_factors=1, offsets=0, nodatavals=255, color_tbls=None,
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
                0: Top left x
                1: W-E pixel resolution
                2: Rotation, 0 if image is "north up"
                3: Top left y
                4: Rotation, 0 if image is "north up"
                5: N-S pixel resolution (negative value if North up)
            Defaults to (0, 1, 0, 0, 0, 1).
        sref_wkt : str, optional
            Coordinate Reference System (CRS) in WKT format. Defaults to none.
        shape : 2-tuple, optional
            2D shape of the raster. Defaults to (1, 1).
        compression : str, optional
            Set the compression to use. Defaults to 'LZW'.
        metadata : dict, optional
            Dictionary representing the metadata of the GeoTIFF file. Defaults to none.
        is_bigtiff : bool, optional
            True if GeoTIFF file should be managed as a 'BIGTIFF' (required if the file will be above 4 GB).
            Defaults to false.
        is_tiled : bool, optional
            True if the mosaic should be tiled (default). False if the mosaic should be stripped.
        blocksize : 2-tuple, optional
            Blocksize of the mosaic blocks in the GeoTIFF file. Defaults to (512, 512).
        n_bands : int, optional
            Number of bands in the GeoTIFF file. Defaults to 1.
        dtypes : dict or str, optional
            Data types used for de- or encoding (NumPy-style). Defaults to 'uint8'. It can either be one value (will
            be used for all bands), or a dictionary mapping the band number with the respective mosaic type.
        scale_factors : dict or number, optional
            Scale factor used for de- or encoding. Defaults to 1. It can either be one value (will be used for all
            bands), or a dictionary mapping the band number with the respective scale factor.
        offsets : dict or number, optional
            Offset used for de- or encoding. Defaults to 0. It can either be one value (will be used for all bands),
            or a dictionary mapping the band number with the respective offset.
        nodatavals : dict or int, optional
            No mosaic value used for de- or encoding. Defaults to 255. It can either be one value (will be used for all
            bands), or a dictionary mapping the band number with the respective no mosaic value.
        color_tbls : dict or gdal.ColorTable, optional
            GDAL color tables. Defaults to None. It can either be one value (will be used for all bands), or a
            dictionary mapping the band number with the respective color table.
        color_intprs : dict or gdal.ColorInterp, optional
            GDAL color interpretation value. Defaults to None. It can either be one value (will be used for all bands),
            or a dictionary mapping the band number with the respective color interpretation value.
        overwrite : bool, optional
            Flag if the file can be overwritten if it already exists (defaults to false).
        auto_decode : bool, optional
            True if mosaic should be decoded according to the information available in its metadata.
            False if not (default).

        """
        self.src = None
        self._driver = gdal.GetDriverByName('GTiff')
        self.filepath = filepath
        self.mode = mode
        self.geotrans = geotrans
        self.sref_wkt = sref_wkt
        self.shape = shape
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

        if shape is not None or self.mode == 'r':
            self._open()

    @property
    def n_bands(self):
        """ int : Number of bands. """
        return len(self.bands)

    @staticmethod
    def _is_bigtiff(filepath):
        """
        Determines if the given GeoTIFF is a BigTIFF file or not.

        Parameters
        ----------
        filepath : str
            Full file path to a GeoTIFF file.

        Returns
        -------
        bool :
            True if the given file is a BigTIFF, else false.

        """
        with open(filepath, 'rb') as f:
            header = f.read(4)
        byteorder = {b'II': '<', b'MM': '>', b'EP': '<'}[header[:2]]
        version = struct.unpack(byteorder + "H", header[2:4])[0]
        return version == 43

    @property
    def scale_factors(self):
        """ list of str : Scale factors of the different bands. """
        return list(self._scale_factors.values())

    @property
    def offsets(self):
        """ list of numbers : Offsets of the different bands. """
        return list(self._offsets.values())

    @property
    def nodatavals(self):
        """ list of numbers : No mosaic values of the different bands. """
        return list(self._nodatavals.values())

    @property
    def color_interps(self):
        """ list of numbers : Color interpretation values of the different bands. """
        return list(self._color_intprs.values())

    @property
    def color_tables(self):
        """ list : Color tables of the different bands. """
        return list(self._color_tbls.values())

    @property
    def dtypes(self):
        """ list of str : Data types in NumPy-style format. """
        return [GDAL_TO_NUMPY_DTYPE[dtype] for dtype in self._dtypes.values()]

    def _open(self):
        """
        Helper function supporting the different mosaic modes, i.e. either opening existing mosaic or creating a new
        mosaic source.

        """
        if self.mode == 'r':
            if not os.path.exists(self.filepath):
                err_msg = f"File '{self.filepath}' does not exist."
                raise FileNotFoundError(err_msg)
            self.src = gdal.Open(self.filepath, gdal.GA_ReadOnly)
            self.shape = self.src.RasterYSize, self.src.RasterXSize
            self.geotrans = self.src.GetGeoTransform()
            self.sref_wkt = self.src.GetProjection()
            self.metadata = self.src.GetMetadata()
            self.blocksize = self.src.GetRasterBand(1).GetBlockSize()  # block seems to be band-independent, because no set function is available per band
            self.compression = self.src.GetMetadata('IMAGE_STRUCTURE').get('COMPRESSION')
            self.is_bigtiff = self._is_bigtiff(self.filepath)
            self.is_tiled = self.blocksize[1] == 1

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
        elif self.mode == 'w':
            if os.path.exists(self.filepath):
                if self.overwrite:
                    os.remove(self.filepath)
                else:
                    err_msg = f"File '{self.filepath}' exists."
                    raise FileExistsError(err_msg)

            gdal_opt = dict()
            gdal_opt['COMPRESS'] = self.compression
            gdal_opt['TILED'] = 'YES' if self.is_tiled else 'NO'
            gdal_opt['BLOCKXSIZE'] = str(self.blocksize[0])
            gdal_opt['BLOCKYSIZE'] = str(self.blocksize[1])
            gdal_opt['BIGTIFF'] = 'YES' if self.is_bigtiff else 'NO'
            gdal_opt = ['='.join((k, v)) for k, v in gdal_opt.items()]
            self.src = self._driver.Create(self.filepath, self.shape[1], self.shape[0],
                                           self.n_bands, NUMPY_TO_GDAL_DTYPE[self.dtypes[0]],
                                           options=gdal_opt)

            self.src.SetGeoTransform(self.geotrans)
            if self.sref_wkt is not None:
                self.src.SetProjection(self.sref_wkt)

            if self.metadata is not None:
                self.src.SetMetadata(self.metadata)

            for band in self.bands:
                self.src.GetRasterBand(band).Fill(int(self._nodatavals[band]))
                self.src.GetRasterBand(band).SetNoDataValue(float(self._nodatavals[band]))
                self.src.GetRasterBand(band).SetScale(self._scale_factors[band])
                self.src.GetRasterBand(band).SetOffset(self._offsets[band])
        else:
            err_msg = f"Mode '{self.mode}' not known."
            raise ValueError(err_msg)

        if self.src is None:
            err_msg = f"Open failed: {self.filepath}"
            raise IOError(err_msg)

    def read(self, row=0, col=0, n_rows=None, n_cols=None, bands=None, decoder=None, decoder_kwargs=None):
        """
        Read mosaic from a GeoTIFF file.

        Parameters
        ----------
        row : int, optional
            Row number/index (defaults to 0).
        col : int, optional
            Column number/index (defaults to 0).
        n_rows : int, optional
            Number of rows to read (default is 1).
        n_cols : int, optional
            Number of columns to read (default is 1).
        bands : int or tuple or list, optional
            Band numbers of the GeoTIFF file to read mosaic from. Defaults to none, i.e. all available bands will be
            used.
        decoder : function, optional
            Decoding function expecting a NumPy array as input.
        decoder_kwargs : dict, optional
            Keyword arguments for the decoder.

        Returns
        -------
        mosaic : dict
            Dictionary mapping band numbers to NumPy arrays read from disk.

        """

        decoder_kwargs = decoder_kwargs or dict()
        n_cols = self.shape[1] if n_cols is None else n_cols
        n_rows = self.shape[0] if n_rows is None else n_rows
        bands = bands or self.bands
        bands = to_list(bands)

        data = dict()
        for band in bands:
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
            data[band] = band_data

        return data

    def write(self, data, row=0, col=0, encoder=None, encoder_kwargs=None):
        """
        Writes mosaic to a GeoTIFF file.

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
            Function allowing to encode mosaic before writing it to disk.
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
            self.shape = data_shape
            self._open()

        for band in data_bands:
            nodataval = self._nodatavals[band]
            scale_factor = self._scale_factors[band]
            offset = self._offsets[band]
            if encoder is not None:
                dtype = GDAL_TO_NUMPY_DTYPE(self._dtypes[band])
                self.src.GetRasterBand(band).WriteArray(encoder(data_write[band],
                                                        band=band,
                                                        nodataval=nodataval,
                                                        scale_factor=scale_factor,
                                                        offset=offset,
                                                        dtype=dtype,
                                                        **encoder_kwargs),
                                                        xoff=col, yoff=row)
            else:
                self.src.GetRasterBand(band).WriteArray(data_write[band], xoff=col, yoff=row)

    def __to_dict(self, arg):
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
        if not isinstance(arg, dict):
            arg_dict = dict()
            for band in self.bands:
                arg_dict[band] = arg
        else:
            arg_dict = arg

        return arg_dict

    def flush(self):
        """
        Flush mosaic on disk.
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


def create_vrt_file(filepaths, vrt_filepath, shape, sref_wkt, geotrans, bands=(1,)):
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
        Coordinate reference system in WKT format.
    geotrans : 6-tuple
        GDAL's geotransformation parameters.
    bands : tuple, optional
        Band numbers. Defaults to (1,).

    """
    n_filepaths = len(filepaths)
    n_bands = len(bands)
    n_rows, n_cols = shape

    ref_filepath = filepaths[0]
    band_attr_dict = dict()
    band_attr_dict['nodataval'] = []
    band_attr_dict['scale_factor'] = []
    band_attr_dict['offset'] = []
    band_attr_dict['dtype'] = []
    band_attr_dict['blocksize'] = []
    with GeoTiffFile(ref_filepath, 'r') as gt_driver:
        for band in bands:
            b_idx = list(gt_driver.bands).index(band)
            band_attr_dict['nodataval'].append(gt_driver.nodatavals[b_idx])
            band_attr_dict['scale_factor'].append(gt_driver.scale_factors[b_idx])
            band_attr_dict['offset'].append(gt_driver.offsets[b_idx])
            band_attr_dict['dtype'].append(gdal.GetDataTypeName(NUMPY_TO_GDAL_DTYPE[gt_driver.dtypes[b_idx]]))
            band_attr_dict['blocksize'].append(gt_driver.blocksize)

    attrib = {"rasterXSize": str(n_cols), "rasterYSize": str(n_rows)}
    vrt_root = ET.Element("VRTDataset", attrib=attrib)

    geot_elem = ET.SubElement(vrt_root, "GeoTransform")
    geot_elem.text = ",".join(map(str, geotrans))

    geot_elem = ET.SubElement(vrt_root, "SRS")
    geot_elem.text = sref_wkt

    i = 1
    for f_idx in range(n_filepaths):
        filepath = filepaths[f_idx]
        for b_idx in range(n_bands):
            band = bands[b_idx]
            attrib = {"dataType": band_attr_dict['dtype'][b_idx], "band": str(i)}
            band_elem = ET.SubElement(vrt_root, "VRTRasterBand", attrib=attrib)
            simsrc_elem = ET.SubElement(band_elem, "SimpleSource")
            attrib = {"relativetoVRT": "0"}
            file_elem = ET.SubElement(simsrc_elem, "SourceFilename", attrib=attrib)
            file_elem.text = filepath
            ET.SubElement(simsrc_elem, "SourceBand").text = str(band)

            attrib = {"RasterXSize": str(n_cols), "RasterYSize": str(n_rows),
                      "DataType": band_attr_dict['dtype'][b_idx],
                      "BlockXSize": str(band_attr_dict['blocksize'][b_idx][0]),
                      "BlockYSize": str(band_attr_dict['blocksize'][b_idx][1])}

            file_elem = ET.SubElement(simsrc_elem, "SourceProperties", attrib=attrib)

            scale_factor = band_attr_dict['scale_factor'][b_idx]
            scale_factor = 1 if scale_factor is None else scale_factor
            ET.SubElement(band_elem, "NodataValue").text = str(band_attr_dict['nodataval'][b_idx])
            ET.SubElement(band_elem, "Scale").text = str(scale_factor)
            ET.SubElement(band_elem, "Offset").text = str(band_attr_dict['offset'][b_idx])
            i += 1

    tree = ET.ElementTree(vrt_root)
    tree.write(vrt_filepath, encoding="UTF-8")


if __name__ == '__main__':
    pass

