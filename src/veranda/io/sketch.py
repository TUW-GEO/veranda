import os
import abc
import warnings
import secrets
import tempfile
import xarray as xr
import numpy as np
import pandas as pd
from osgeo import ogr
from osgeo import gdal
from datetime import datetime
import xml.etree.ElementTree as ET

from geospade.tools import any_geom2ogr_geom
from geospade.tools import rel_extent
from geospade.crs import SpatialRef
from geospade.raster import RasterGeometry
from geospade.raster import Tile
from geospade.raster import MosaicGeometry

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
    def __init__(self, file_register, mosaic, data=None, mosaic_class=MosaicGeometry, mosaic_kwargs=None):
        self._file_register = file_register
        self._mosaic = mosaic
        self._data = data
        self._mosaic_class = mosaic_class
        self._mosaic_kwargs = dict() if mosaic_kwargs is None else mosaic_kwargs

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

    def select_tile(self, tile_name):
        if 'geom_id' in self._file_register.columns:
            self._file_register = self._file_register.loc[self._file_register['geom_id'] == tile_name]

            # TODO: implement this in geospade.raster.MosaicGeometry
            self._mosaic._tiles['active'] = False
            self._mosaic._tiles.loc[tile_name, 'active'] = True

        else:
            wrn_msg = "The data is not available as a mosaic anymore."
            warnings.warn(wrn_msg)
        return self

    def select_xy(self, x, y, sref=None):
        tile_oi = self._mosaic.xy2tile(x, y, sref=sref)
        row, col = tile_oi.xy2rc(x, y, sref=sref)

        tile_oi.slice_by_rc(row, col, inplace=True)
        tile_oi.active = True
        self._mosaic = self._mosaic_class([tile_oi], check_consistency=False, **self._mosaic_kwargs)

        if 'geom_id' in self._file_register.columns:
            self._file_register = self._file_register.drop('geom_id')

        if self._data is not None:
            self._data = self._data[..., row, col]
        return self

    def select_bbox(self, bbox, sref=None):
        ogr_geom = any_geom2ogr_geom(bbox, sref=sref)
        return self.select_polygon(ogr_geom, apply_mask=False)

    def select_polygon(self, polygon, sref=None, apply_mask=True):
        tiles_oi = list(self._mosaic.slice_tiles_by_geom(polygon, active_only=False).values())

        # TODO: implement this in geospade.raster.MosaicGeometry
        for tile_oi in tiles_oi:
            tile_oi.active = True

        self._mosaic = self._mosaic_class(tiles_oi, check_consistency=False, **self._mosaic_kwargs)

        if 'geom_id' in self._file_register.columns:
            geom_ids_mask = self._file_register['geom_id'] == tiles_oi[0].parent_root.name
            for tile_oi in tiles_oi[1:]:
                geom_ids_mask_curr = self._file_register['geom_id'] == tile_oi.parent_root.name
                geom_ids_mask = geom_ids_mask | geom_ids_mask_curr
            self._file_register = self._file_register.loc[geom_ids_mask]

        if self._data is not None:
            tile_oi = self._mosaic.tiles[0]
            tile_oi.slice_by_geom(polygon, sref=sref, inplace=True)

            # TODO: implement this in geospade.raster.RasterGeometry
            origin = (tile_oi.parent.ul_x, tile_oi.parent.ul_y)

            min_col, min_row, max_col, max_row = rel_extent(origin, tile_oi.coord_extent,
                                                            x_pixel_size=tile_oi.x_pixel_size,
                                                            y_pixel_size=tile_oi.y_pixel_size)
            self._data = self._data[..., min_row: max_row + 1, min_col:max_col + 1]

        return self


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

        tile = Tile(n_rows, n_cols, sref=SpatialRef(sref_wkt), geotrans=geotrans, name=1)
        mosaic_geom = MosaicGeometry([tile], check_consistency=False)

        return cls(file_register, mosaic_geom)

    @classmethod
    def from_mosaic_filepaths(cls, filepaths):
        file_register_dict = dict()
        file_register_dict['filepath'] = filepaths
        file_register_dict['io_pointer'] = [None] * len(filepaths)

        geom_ids = []
        layer_ids = []
        tiles = []
        tile_idx = 1
        for filepath in filepaths[1:]:
            with GeoTiffDriver(filepath, 'r') as gt_driver:
                sref_wkt = gt_driver.sref_wkt
                geotrans = gt_driver.geotrans
                n_rows, n_cols = gt_driver.n_rows, gt_driver.n_cols
            curr_tile = Tile(n_rows, n_cols, sref=SpatialRef(sref_wkt), geotrans=geotrans, name=tile_idx)
            curr_tile_idx = None
            for tile in tiles:
                if curr_tile == tile:
                    curr_tile_idx = tile.name
                    break
            if curr_tile_idx is None:
                tiles.append(curr_tile)
                curr_tile_idx = tile_idx
                tile_idx += 1

            geom_ids.append(curr_tile_idx)
            layer_id = sum(np.array(geom_ids) == curr_tile_idx)
            layer_ids.append(layer_id)

        file_register_dict['geom_id'] = geom_ids
        file_register_dict['layer_id'] = layer_ids
        file_register = pd.DataFrame(file_register_dict)

        mosaic_geom = MosaicGeometry(tiles, check_consistency=False)

        return cls(file_register, mosaic_geom)

    def read(self, bands=(1,), engine='vrt', auto_decode=False, decoder=None, decoder_kwargs=None, tile_class=Tile,
             tile_kwargs=None):
        n_cols = self._mosaic.n_cols if n_cols is None else n_cols
        n_rows = self._mosaic.n_rows if n_rows is None else n_rows

        if engine == 'vrt':
            data = self.__read_vrt_stack(row=row, col=col, n_rows=n_rows, n_cols=n_cols, bands=bands,
                                         auto_decode=auto_decode, decoder=None, decoder_kwargs=None)
        elif engine == 'parallel':
            data = self.__read_parallel(row=row, col=col, n_rows=n_rows, n_cols=n_cols, bands=bands,
                                       auto_decode=auto_decode, decoder=None, decoder_kwargs=None)
        else:
            err_msg = f"Engine '{engine}' is not supported!"
            raise ValueError(err_msg)

        if n_cols ==
        tile = tile_class(n_rows, n_cols)
        self._mosaic_class
        self._data = self._to_xarray(data, bands)
        return self._data

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

        attrib = {"rasterXSize": str(self._mosaic.n_cols), "rasterYSize": str(self._mosaic.n_rows)}
        vrt_root = ET.Element("VRTDataset", attrib=attrib)

        geot_elem = ET.SubElement(vrt_root, "GeoTransform")
        geot_elem.text = ",".join(map(str, self._mosaic.geotrans))

        geot_elem = ET.SubElement(vrt_root, "SRS")
        geot_elem.text = self._mosaic.sref.wkt

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

                attrib = {"RasterXSize": str(self._mosaic.n_cols), "RasterYSize": str(self._mosaic.n_rows),
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

    def __read_vrt_stack(self, row=0, col=0, n_rows=None, n_cols=None,  bands=(1,), reset_io=False, auto_decode=False,
                         decoder=None, decoder_kwargs=None):
        n_bands = len(bands)
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

        data = dict()
        vrt_data = ref_io.ReadAsArray(col, row, n_cols, n_rows)
        for band in bands:
            band_data = vrt_data[(band - 1)::n_bands, ...]
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

    def __read_parallel(self):
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
        coord_dict['x'] = self._mosaic.x_coords
        coord_dict['y'] = self._mosaic.y_coords

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
        return [gdal.GetDataTypeName(dtype) for dtype in self.dtypes]

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
    data = gt_stack.read(0, 0, 1, 1)
    pass
