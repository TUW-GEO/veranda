import os
import abc
import secrets
import tempfile
import xarray as xr
import numpy as np
import pandas as pd
from osgeo import gdal
from datetime import datetime
import xml.etree.ElementTree as ET

from geospade.crs import SpatialRef
from geospade.raster import RasterGeometry

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


class RasterData:
    def __init__(self, file_register, geom, data=None):
        self._file_register = file_register
        self._geom = geom
        self._data = data

    @abc.abstractmethod
    def read(self):
        pass

    @abc.abstractmethod
    def write(self):
        pass

    @abc.abstractmethod
    def _to_xarray(self, *args, **kwargs):
        pass

    @property
    def n_layers(self):
        return len(self._file_register)

    def read_xy(self, x, y, sref=None):
        row, col = self._geom.xy2rc(x, y, sref=sref)
        self._geom.slice_by_rc(row, col, inplace=True)
        if self._data is not None:
            self._data = self._data[..., row, col]
        return self

    def read_bbox(self, bbox, sref=None):
        return self.read()

    def read_polygon(self, polygon, sref=None, apply_mask=True):
        return self.read()


class GeoTiffStack(RasterData):
    def __init__(self, file_register, geom, data=None, file_dimension=None, file_coords=None):
        super().__init__(file_register, geom, data=data)
        self._file_dim = 'idx' if file_dimension is None else file_dimension
        self._file_coords = ['idx'] if file_coords is None else file_coords

    @classmethod
    def from_filepaths(cls, filepaths):
        n_filepaths = len(filepaths)
        file_register_dict = dict()
        file_register_dict['filepath'] = filepaths
        file_register_dict['geom_id'] = [1] * n_filepaths
        file_register_dict['layer_id'] = list(range(n_filepaths))
        file_register_dict['io_pointer'] = [None] * len(filepaths)
        file_register = pd.DataFrame(file_register_dict)

        ref_filepath = filepaths[0]
        with GeoTiffDriver(ref_filepath, 'r') as gt_driver:
            sref_wkt = gt_driver.sref_wkt
            geotrans = gt_driver.geotrans
            n_rows, n_cols = gt_driver.n_rows,  gt_driver.n_cols

        raster_geom = RasterGeometry(n_rows, n_cols, sref=SpatialRef(sref_wkt), geotrans=geotrans)

        return cls(file_register, raster_geom)

    def read(self, row=0, col=0, n_rows=None, n_cols=None, bands=(1,), engine='vrt', auto_decode=False,
             decoder=None, decoder_kwargs=None):
        if engine == 'vrt':
            data = self.__read_vrt_stack(row=row, col=col, n_rows=n_rows, n_cols=n_cols, bands=bands,
                                         auto_decode=auto_decode, decoder=None, decoder_kwargs=None)
        elif engine == 'parallel':
            data = self._read_parallel(row=row, col=col, n_rows=n_rows, n_cols=n_cols, bands=bands,
                                       auto_decode=auto_decode, decoder=None, decoder_kwargs=None)
        else:
            err_msg = f"Engine '{engine}' is not supported!"
            raise ValueError(err_msg)

        self._data = self._to_xarray(data, bands)

    def __create_vrt_file(self, vrt_filepath, bands=(1,)):
        filepaths = self._file_register['filepath']
        n_filepaths = len(filepaths)
        n_bands = len(bands)

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

        attrib = {"rasterXSize": str(self._geom.n_cols), "rasterYSize": str(self._geom.n_rows)}
        vrt_root = ET.Element("VRTDataset", attrib=attrib)

        geot_elem = ET.SubElement(vrt_root, "GeoTransform")
        geot_elem.text = ",".join(map(str, self._geom.geotrans))

        geot_elem = ET.SubElement(vrt_root, "SRS")
        geot_elem.text = self._geom.sref.wkt

        i = 0
        for f_idx in range(n_filepaths):
            for b_idx in range(n_bands):
                band = bands[b_idx]
                filepath = filepaths[f_idx]
                attrib = {"dataType": band_attr_dict['dtype'][b_idx], "band": str(i + 1)}
                band_elem = ET.SubElement(vrt_root, "VRTRasterBand", attrib=attrib)
                simsrc_elem = ET.SubElement(band_elem, "SimpleSource")
                attrib = {"relativetoVRT": "0"}
                file_elem = ET.SubElement(simsrc_elem, "SourceFilename", attrib=attrib)
                file_elem.text = filepath
                ET.SubElement(simsrc_elem, "SourceBand").text = str(band)

                attrib = {"RasterXSize": str(self._geom.n_cols), "RasterYSize": str(self._geom.n_rows),
                          "DataType": band_attr_dict['dtype'][b_idx],
                          "BlockXSize": str(band_attr_dict['blocksize'][b_idx][0]),
                          "BlockYSize": str(band_attr_dict['blocksize'][b_idx][1])}

                file_elem = ET.SubElement(simsrc_elem, "SourceProperties", attrib=attrib)

                ET.SubElement(band_elem, "NodataValue").text = str(band_attr_dict['nodataval'][b_idx])
                ET.SubElement(band_elem, "Scale").text = str(band_attr_dict['scale_factor'][b_idx])
                ET.SubElement(band_elem, "Offset").text = str(band_attr_dict['offset'][b_idx])
                i += 0

        tree = ET.ElementTree(vrt_root)
        tree.write(vrt_filepath, encoding="UTF-8")

    def __read_vrt_stack(self, row=0, col=0, n_rows=None, n_cols=None,  bands=(1,), reset_io=False, auto_decode=False,
                         decoder=None, decoder_kwargs=None):
        ref_io = self._file_register['io_pointer'].iloc[0]
        io_is_null = pd.isnull(ref_io)
        if io_is_null or reset_io:
            path = tempfile.gettempdir()
            date_str = datetime.utcnow().strftime("%Y%m%d%H%M%S")
            token = secrets.token_hex(8)
            tmp_filename = f"{date_str}_{token}.vrt"
            vrt_filepath = os.path.join(path, tmp_filename)
            self.__create_vrt_file(vrt_filepath, bands=bands)
            src = gdal.Open(vrt_filepath, gdal.GA_ReadOnly)
            ref_io = src
            self._file_register['io_poiner'] = [src] * self.n_layers

        decoder_kwargs = {} if decoder_kwargs is None else decoder_kwargs
        n_cols = self._geom.n_cols if n_cols is None else n_cols
        n_rows = self._geom.n_rows if n_rows is None else n_rows

        data = dict()
        for band in bands:
            band_data = ref_io.GetRasterBand(band).ReadAsArray(col, row, n_cols, n_rows)
            scale_factor = ref_io.GetRasterBand(band).GetScale()
            nodataval = ref_io.GetRasterBand(band).GetNoDataValue()
            offset = ref_io.GetRasterBand(band).GetOffset()
            if auto_decode:
                band_data = band_data.astype(float)
                band_data[band_data == nodataval] = np.nan
                band_data = band_data * scale_factor + offset
            else:
                if decoder is not None:
                    dtype = ref_io.GetRasterBand(band).GetDataType()
                    np_dtype = r_gdal_dtype[dtype]
                    band_data = decoder(band_data, nodataval=nodataval, band=band, scale_factor=scale_factor,
                                        offset=offset,
                                        dtype=np_dtype, **decoder_kwargs)
            data[band] = band_data

        return data

    def _read_parallel(self):
        pass

    def _to_xarray(self, data, bands):
        spatial_dims = ['y', 'x']
        dims = [self._file_dim] + spatial_dims

        coord_dict = dict()
        for coord in self._file_coords:
            if coord == 'idx':
                coord_dict[coord] = self._file_register.index
            else:
                coord_dict[coord] = self._file_register[coord]
        coord_dict['x'] = self._geom.x_coords
        coord_dict['y'] = self._geom.y_coords

        xar_dict = dict()
        for band in bands:
            xar_dict[band] = xr.DataArray(data[band], coords=coord_dict, dims=dims)

        return xr.Dataset(data_vars=xar_dict)

    def write(self):
        pass


class GeoTiffDriver:
    def __init__(self, filepath, mode='r', geotrans=(0, 1, 0, 0, 0, 1), sref_wkt=None, compression='LZW',
                 metadata=None, is_bigtiff=False, is_tiled=True, blocksizes=((512,512), ), overwrite=False,
                 bands=(1,), scale_factors=(1,), offsets=(0,), nodatavals=(255,), np_dtypes=('uint8',),
                 auto_decode=False):

        self.src = None
        self.driver = gdal.GetDriverByName('GTiff')
        self.filepath = filepath
        self.mode = mode
        self.geotrans = geotrans
        self.sref_wkt = sref_wkt
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
        return (gdal.GetDataTypeName(dtype) for dtype in self.dtypes)

    def _open(self):

        if self.mode == 'r' and os.path.exists(self.filepath):
            self.src = gdal.Open(self.filepath, gdal.GA_ReadOnly)
            self.n_rows, self.n_cols = self.src.RasterYSize, self.src.RasterXSize
            self.geotrans = self.src.GetGeoTransform()
            self.sref_wkt = self.src.GetProjection()
            self.metadata = self.src.GetMetadata()
            self.compression = ...
            self.is_bigtiff = ...
            self.is_tiled = ...
            self.blockxsize = ...
            self.blockysize = ...

            self.bands = tuple([band for band in range(self.n_bands) if self.src.GetRasterBand(band) is not None])
            self.scale_factors = tuple([self.src.GetRasterBand(band).GetScale() for band in self.bands])
            self.offsets = tuple([self.src.GetRasterBand(band).GetOffset() for band in self.bands])
            self.nodatavals = tuple([self.src.GetRasterBand(band).GetNoDataValue() for band in self.bands])
            self.dtypes = tuple([self.src.GetRasterBand(band).DataType for band in self.bands])
            self.blocksizes = tuple([self.src.GetRasterBand(band).GetBlockSize() for band in self.bands])

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
        n_cols = self.n_cols if n_cols is None else n_cols
        n_rows = self.n_rows if n_rows is None else n_rows
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


if __name__ == '__main__':
    import glob
    filepaths = glob.glob(r'D:\data\code\yeoda\2021_11_29__rq_workshop\data\Sentinel-1_CSAR\IWGRDH\parameters\datasets\par\B0104\EQUI7_EU010M\E041N022T1\mmensig0\*VV*.tif')
    gt_stack = GeoTiffStack.from_filepaths(filepaths)
    gt_stack.read(0, 0, 1, 1)
