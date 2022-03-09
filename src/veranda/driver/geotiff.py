import os
import numpy as np
import xml.etree.ElementTree as ET
from osgeo import gdal

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


# TODO: align no data values to be of type integer
class GeoTiffDriver:
    def __init__(self, filepath, mode='r', geotrans=(0, 1, 0, 0, 0, 1), sref_wkt=None, shape=(1, 1), compression='LZW',
                 metadata=None, is_bigtiff=False, is_tiled=True, blocksizes=((512,512), ), overwrite=False,
                 bands=(1,), scale_factors=(1,), offsets=(0,), nodatavals=(255,), np_dtypes=('uint8',),
                 auto_decode=False):

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
        self.blocksizes = blocksizes
        self.overwrite = overwrite
        self.auto_decode = auto_decode

        self.bands = bands
        self.scale_factors = scale_factors
        self.offsets = offsets
        self.nodatavals = nodatavals
        self.dtypes = tuple([_numpy2gdal_dtype[np_dtype] for np_dtype in np_dtypes])

        self._open()

    @property
    def n_bands(self):
        return len(self.bands)

    @property
    def dtype_names(self):
        return [gdal.GetDataTypeName(dtype) for dtype in self.dtypes]

    def _open(self):

        if self.mode == 'r':
            if not os.path.exists(self.filepath):
                err_msg = f"File '{self.filepath}' does not exist."
                raise FileNotFoundError(err_msg)
            self.src = gdal.Open(self.filepath, gdal.GA_ReadOnly)
            self.shape = self.src.RasterYSize, self.src.RasterXSize
            self.geotrans = self.src.GetGeoTransform()
            self.sref_wkt = self.src.GetProjection()
            self.metadata = self.src.GetMetadata()
            self.compression = ...
            self.is_bigtiff = ...
            self.is_tiled = ...

            self.bands = []
            self.scale_factors = []
            self.offsets = []
            self.nodatavals = []
            self.dtypes = []
            self.blocksizes = []
            for band in range(1, self.src.RasterCount + 1):
                self.bands.append(band)
                self.scale_factors.append(self.src.GetRasterBand(band).GetScale())
                self.offsets.append(self.src.GetRasterBand(band).GetOffset())
                self.nodatavals.append(self.src.GetRasterBand(band).GetNoDataValue())
                self.dtypes.append(self.src.GetRasterBand(band).DataType)
                self.blocksizes.append(self.src.GetRasterBand(band).GetBlockSize())
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
            gdal_opt['BLOCKXSIZE'] = str(self.blocksizes[0][0])
            gdal_opt['BLOCKYSIZE'] = str(self.blocksizes[0][1])
            gdal_opt = ['='.join((k, v)) for k, v in gdal_opt.items()]
            self.src = self._driver.Create(self.filepath, self.shape[1], self.shape[0],
                                           self.n_bands, self.dtypes[0],
                                           options=gdal_opt)

            self.src.SetGeoTransform(self.geotrans)
            self.src.SetProjection(self.sref_wkt)

            if self.metadata is not None:
                self.src.SetMetadata(self.metadata)

            # set fill value for each band to the given no data values
            for band in self.bands:
                band_idx = self.bands.index(band)
                self.src.GetRasterBand(band).Fill(int(self.nodatavals[band_idx]))
        else:
            err_msg = f"Mode '{self.mode}' not known."
            raise ValueError(err_msg)

    def read(self, row=0, col=0, n_rows=None, n_cols=None, bands=None, decoder=None, decoder_kwargs=None):
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

        decoder_kwargs = {} if decoder_kwargs is None else decoder_kwargs
        n_cols = self.shape[1] if n_cols is None else n_cols
        n_rows = self.shape[0] if n_rows is None else n_rows
        bands = self.bands if bands is None else bands

        data = dict()
        for band in bands:
            band_data = self.src.GetRasterBand(band).ReadAsArray(col, row, n_cols, n_rows)
            band_idx = self.bands.index(band)
            scale_factor = self.scale_factors[band_idx]
            nodataval = self.nodatavals[band_idx]
            offset = self.offsets[band_idx]
            if self.auto_decode:
                band_data = band_data.astype(float)
                band_data[band_data == nodataval] = np.nan
                band_data = band_data * scale_factor + offset
            else:
                if decoder is not None:
                    dtype = self.dtypes[band_idx]
                    np_dtype = r_gdal_dtype[dtype]
                    band_data = decoder(band_data, nodataval=nodataval, band=band, scale_factor=scale_factor, offset=offset,
                                   dtype=np_dtype, **decoder_kwargs)
            data[band] = band_data

        return data

    def write(self, data, bands=(1,), row=0, col=0, encoder=None, encoder_kwargs=None):
        encoder_kwargs = {} if encoder_kwargs is None else encoder_kwargs
        if data.ndim == 2:
            data = data[None, ...]

        n_bands = len(bands)
        n_data_layers = data.shape[0]
        if n_data_layers != n_bands:
            err_msg = f"Number data layers and number of bands do not match: {n_data_layers} != {n_bands}"
            raise ValueError(err_msg)

        for band in bands:
            nodataval = self.nodatavals[band - 1]
            scale_factor = self.scale_factors[band - 1]
            offset = self.offsets[band - 1]
            if encoder is not None:
                dtype = self.dtypes[band - 1]
                np_dtype = r_gdal_dtype[dtype]
                self.src.GetRasterBand(band).WriteArray(encoder(data[band, ...],
                                                        band=band,
                                                        nodataval=nodataval,
                                                        scale_factor=scale_factor,
                                                        offset=offset,
                                                        dtype=np_dtype,
                                                        **encoder_kwargs),
                                                        xoff=col, yoff=row)
            else:
                self.src.GetRasterBand(band).WriteArray(data[band - 1, ...], xoff=col, yoff=row)
            self.src.GetRasterBand(band).SetNoDataValue(float(nodataval))
            self.src.GetRasterBand(band).SetScale(scale_factor)
            self.src.GetRasterBand(band).SetOffset(offset)

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


def create_vrt_file(filepaths, vrt_filepath, shape, sref_wkt, geotrans, bands=(1,)):
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
    with GeoTiffDriver(ref_filepath, 'r') as gt_driver:
        for band in bands:
            b_idx = list(gt_driver.bands).index(band)
            band_attr_dict['nodataval'].append(gt_driver.nodatavals[b_idx])
            band_attr_dict['scale_factor'].append(gt_driver.scale_factors[b_idx])
            band_attr_dict['offset'].append(gt_driver.offsets[b_idx])
            band_attr_dict['dtype'].append(gt_driver.dtype_names[b_idx])
            band_attr_dict['blocksize'].append(gt_driver.blocksizes[b_idx])

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


