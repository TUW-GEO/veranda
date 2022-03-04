import os
import abc
import warnings
import secrets
import tempfile
import rioxarray
import xarray as xr
import numpy as np
import pandas as pd
import shapely.wkt
from osgeo import ogr
from osgeo import gdal
from affine import Affine
from datetime import datetime
import xml.etree.ElementTree as ET

from geospade.tools import any_geom2ogr_geom
from geospade.tools import rel_extent
from geospade.tools import rasterise_polygon
from geospade.crs import SpatialRef
from geospade.transform import transform_geom
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
    def __init__(self, file_register, mosaic, data=None):
        self._file_register = file_register
        self._drivers = dict()
        self._mosaic = mosaic
        self._data = data

    @abc.abstractmethod
    def read(self):
        pass

    @abc.abstractmethod
    def write(self):
        pass

    @abc.abstractmethod
    def export(self):
        pass

    @abc.abstractmethod
    def _to_xarray(self, *args, **kwargs):
        pass

    @classmethod
    def _from_xarray(cls, data, file_register):
        sref_wkt = data.spatial_ref.attrs.get('spatial_ref')
        x_pixel_size = data.x.data[1] - data.x.data[0]
        y_pixel_size = data.y.data[0] - data.y.data[1]
        extent = [data.x.data[0], data.y.data[-1] - y_pixel_size, data.x.data[-1] + x_pixel_size, data.y.data[0]]
        tile = Tile.from_extent(extent, SpatialRef(sref_wkt),
                                x_pixel_size=x_pixel_size, y_pixel_size=y_pixel_size,
                                name=1)
        mosaic = MosaicGeometry([tile], check_consistency=False)
        return cls(file_register, mosaic, data=data)

    @abc.abstractmethod
    def apply_nan(self):
        for dvar in self._data.data_vars:
            dar = self._data[dvar]
            self._data[dvar] = dar.where(dar != dar.attrs['fill_value'])

    @property
    def n_layers(self):
        return max(self._file_register['layer_id'])

    # TODO data also needs to be cropped!
    def select_layer(self, layer_ids):
        self.close()
        self._file_register = self._file_register[self._file_register['layer_id'].isin(layer_ids)]
        return self

    def select_tile(self, tile_name):
        if 'geom_id' in self._file_register.columns:
            self._file_register = self._file_register.loc[self._file_register['geom_id'] == tile_name]
            self._mosaic.slice_by_tile_names([tile_name])
        else:
            wrn_msg = "The data is not available as a mosaic anymore."
            warnings.warn(wrn_msg)
        return self

    def select_xy(self, x, y, sref=None):
        tile_oi = self._mosaic.xy2tile(x, y, sref=sref)
        raster_data = None
        if tile_oi is not None:
            row, col = tile_oi.xy2rc(x, y, sref=sref)
            tile_oi.slice_by_rc(row, col, inplace=True, name=1)
            tile_oi.active = True
            self._mosaic = self._mosaic._from_sliced_tiles([tile_oi])

            if 'geom_id' in self._file_register.columns:
                self._file_register = self._file_register.drop(columns='geom_id')

            if self._data is not None:
                xrars = dict()
                for dvar in self._data.data_vars:
                    xrars[dvar] = self._data[dvar][..., row:row+1, col:col+1]
                self._data = xr.Dataset(xrars)
            raster_data = self

        return raster_data

    def select_bbox(self, bbox, sref=None):
        ogr_geom = any_geom2ogr_geom(bbox, sref=sref)
        return self.select_polygon(ogr_geom, apply_mask=False)

    def select_polygon(self, polygon, sref=None, apply_mask=True):
        polygon = any_geom2ogr_geom(polygon, sref=sref)
        sliced_mosaic = self._mosaic.slice_mosaic_by_geom(polygon, active_only=False)

        if apply_mask:
            if not sliced_mosaic.sref.osr_sref.IsSame(sref.osr_sref):
                polygon = transform_geom(polygon, sliced_mosaic.sref)
            for tile in sliced_mosaic.tiles:
                intrsctn = tile.boundary_ogr.Intersection(polygon)
                intrsctn.FlattenTo2D()

                polygon_mask = rasterise_polygon(shapely.wkt.loads(intrsctn.ExportToWkt()),
                                                 tile.x_pixel_size,
                                                 tile.y_pixel_size,
                                                 extent=tile.coord_extent)

                tile.mask = tile.mask * polygon_mask

        self._mosaic = self._mosaic._from_sliced_tiles(sliced_mosaic.tiles)

        if 'geom_id' in self._file_register.columns:
            geom_ids_mask = self._file_register['geom_id'] == self._mosaic.tiles[0].parent_root.name
            for tile in self._mosaic.tiles[1:]:
                geom_ids_mask_curr = self._file_register['geom_id'] == tile.parent_root.name
                geom_ids_mask = geom_ids_mask | geom_ids_mask_curr
            self._file_register = self._file_register.loc[geom_ids_mask]

        if self._data is not None:
            ref_tile = self._mosaic.tiles[0]
            ref_tile.slice_by_geom(polygon, sref=sref, inplace=True)
            origin = (ref_tile.parent.ul_x, ref_tile.parent.ul_y)

            min_col, min_row, max_col, max_row = rel_extent(origin, ref_tile.coord_extent,
                                                            x_pixel_size=ref_tile.x_pixel_size,
                                                            y_pixel_size=ref_tile.y_pixel_size)

            xrars = dict()
            for dvar in self._data.data_vars:
                xrars[dvar] = self._data[dvar][..., min_row: max_row + 1, min_col:max_col + 1]
            self._data = xr.Dataset(xrars)

        return self

    def close(self):
        self._file_register['driver_id'] = None
        for driver in self._drivers.values():
            driver.close()
        self._drivers = dict()

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()


class GeoTiffStack(RasterData):
    def __init__(self, file_register, mosaic, data=None, file_dimension=None, file_coords=None):
        super().__init__(file_register, mosaic, data=data)
        self._file_dim = 'idx' if file_dimension is None else file_dimension
        self._file_coords = ['idx'] if file_coords is None else file_coords

    @classmethod
    def from_filepaths(cls, filepaths):
        n_filepaths = len(filepaths)
        file_register_dict = dict()
        file_register_dict['filepath'] = filepaths
        file_register_dict['geom_id'] = [1] * n_filepaths
        file_register_dict['layer_id'] = list(range(n_filepaths))
        file_register_dict['driver_id'] = [None] * len(filepaths)
        file_register = pd.DataFrame(file_register_dict)

        ref_filepath = filepaths[0]
        with GeoTiffDriver(ref_filepath, 'r') as gt_driver:
            sref_wkt = gt_driver.sref_wkt
            geotrans = gt_driver.geotrans
            n_rows, n_cols = gt_driver.shape

        tile = Tile(n_rows, n_cols, sref=SpatialRef(sref_wkt), geotrans=geotrans, name=1)
        mosaic_geom = MosaicGeometry([tile], check_consistency=False)

        return cls(file_register, mosaic_geom)

    @classmethod
    def from_mosaic_filepaths(cls, filepaths):
        file_register_dict = dict()
        file_register_dict['filepath'] = filepaths
        file_register_dict['driver_id'] = [None] * len(filepaths)

        geom_ids = []
        layer_ids = []
        tiles = []
        tile_idx = 1
        for filepath in filepaths:
            with GeoTiffDriver(filepath, 'r') as gt_driver:
                sref_wkt = gt_driver.sref_wkt
                geotrans = gt_driver.geotrans
                n_rows, n_cols = gt_driver.shape
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

    # TODO: maybe move this to RasterData
    @classmethod
    def from_other(cls, gt_stack, file_register=None, mosaic=None, data=None, file_dimension=None, file_coords=None):
        file_register = gt_stack._file_register if file_register is None else file_register
        mosaic = gt_stack._mosaic if mosaic is None else mosaic
        data = gt_stack._data if data is None else data
        file_dimension = gt_stack._file_dim if file_dimension is None else file_dimension
        file_coords = gt_stack._file_coords if file_coords is None else file_coords

        return cls(file_register, mosaic, data=data, file_dimension=file_dimension, file_coords=file_coords)

    def read(self, bands=(1,), engine='vrt', auto_decode=False, decoder=None, decoder_kwargs=None):

        ref_filepath = self._file_register['filepath'].iloc[0]
        with GeoTiffDriver(ref_filepath, 'r') as gt_driver:
            dtypes = [dtype.lower() for dtype in gt_driver.dtype_names]
            nodatavals = gt_driver.nodatavals

        coord_extents = []
        for tile in self._mosaic.tiles:
            coord_extents.extend(list(tile.coord_extent))
        x_coords, y_coords = coord_extents[::2], coord_extents[1::2]
        min_x, min_y, max_x, max_y = min(x_coords), min(y_coords), max(x_coords), max(y_coords)
        ref_tile = self._mosaic.tiles[0]
        x_pixel_size = ref_tile.x_pixel_size
        y_pixel_size = ref_tile.y_pixel_size
        new_extent = (min_x, min_y - y_pixel_size,
                      max_x + x_pixel_size,
                      max_y)
        new_tile = ref_tile.from_extent(new_extent, sref=self._mosaic.sref,
                                        x_pixel_size=x_pixel_size,
                                        y_pixel_size=y_pixel_size,
                                        name=1)
        data = dict()
        for band in bands:
            np_dtype = np.dtype(dtypes[band - 1])
            nodatavals[band - 1] = np.array((nodatavals[band - 1])).astype(np_dtype)
            data[band] = np.ones((self.n_layers, new_tile.n_rows, new_tile.n_cols), dtype=np_dtype) * nodatavals[band - 1]

        for tile in self._mosaic.tiles:
            origin = tile.parent.ul_x, tile.parent.ul_y
            min_col, min_row, _, _ = rel_extent(origin, tile.coord_extent, x_pixel_size, y_pixel_size)
            if engine == 'vrt':
                tile_data = self.__read_vrt_stack(row=int(min_row), col=int(min_col),
                                                  n_rows=tile.n_rows, n_cols=tile.n_cols,
                                                  bands=bands, geom_id=tile.parent_root.name,
                                                  auto_decode=auto_decode, decoder=decoder, decoder_kwargs=decoder_kwargs)
            elif engine == 'parallel':
                tile_data = self.__read_parallel(row=int(min_row), col=int(min_col),
                                                 n_rows=tile.n_rows, n_cols=tile.n_cols,
                                                 bands=bands,
                                                 auto_decode=auto_decode, decoder=decoder, decoder_kwargs=decoder_kwargs)
            else:
                err_msg = f"Engine '{engine}' is not supported!"
                raise ValueError(err_msg)

            file_register = self._file_register.loc[self._file_register['geom_id'] == tile.parent.name]
            origin = new_tile.ul_x, new_tile.ul_y
            min_col, min_row, max_col, max_row = rel_extent(origin, tile.coord_extent, x_pixel_size, y_pixel_size)
            layer_ids = file_register['layer_id']
            for band in bands:
                tile_data[band][:, ~tile.mask.astype(bool)] = nodatavals[band-1]
                data[band][np.array(layer_ids) - 1, min_row:max_row + 1, min_col:max_col + 1] = tile_data[band]

        self._mosaic = self._mosaic._from_sliced_tiles([new_tile])
        self._data = self._to_xarray(data, nodatavals)
        return self._data

    def write(self, data=None):
        data = self._data if data is None else data
        raster_data = RasterData._from_xarray(data, self._file_register)
        rd_tile = raster_data._mosaic.tiles[0]

        bands = list(data.data_vars)
        nodatavals = []
        scale_factors = []
        offsets = []
        dtypes = []
        for band in bands:
            nodatavals.append(data[band].attrs.get('fill_value', 0))
            scale_factors.append(data[band].attrs.get('scale_factor', 1))
            offsets.append(data[band].attrs.get('offset', 0))
            dtypes.append(data[band].data.dtype.name)

        for i, entry in self._file_register.iterrows():
            filepath = entry['filepath']
            layer_id = entry.get('layer_id')
            geom_id = entry.get('geom_id', 1)
            tile = self._mosaic[geom_id]
            if not tile.intersects(rd_tile):
                continue
            driver_id = entry.get('driver_id', None)
            if driver_id is None:
                gt_driver = GeoTiffDriver(filepath, mode='w', geotrans=tile.geotrans, sref_wkt=tile.sref.wkt,
                                          shape=tile.shape, bands=bands, scale_factors=scale_factors, offsets=offsets,
                                          nodatavals=nodatavals, np_dtypes=dtypes)
                driver_id = len(list(self._drivers.keys())) + 1
                self._drivers[driver_id] = gt_driver
                self._file_register.loc[i, 'driver_id'] = driver_id

            gt_driver = self._drivers[driver_id]
            origin = tile.ul_x, tile.ul_y
            out_tile = rd_tile.slice_by_geom(tile, inplace=False)
            col, row, _, _ = rel_extent(origin, out_tile.coord_extent,
                                        out_tile.x_pixel_size, out_tile.y_pixel_size)
            origin = rd_tile.ul_x, rd_tile.ul_y
            min_col, min_row, max_col, max_row = rel_extent(origin, out_tile.coord_extent,
                                                            out_tile.x_pixel_size, out_tile.y_pixel_size)
            xrds = data[{self._file_dim: layer_id}]
            data_write = xrds[bands].to_array().data
            gt_driver.write(data_write[..., min_row: max_row + 1, min_col: max_col + 1], bands, row=row, col=col)

    def export(self, file_register=None, overwrite=False):
        file_register = self._file_register if file_register is None else file_register
        bands = list(self._data.data_vars)
        nodatavals = []
        scale_factors = []
        offsets = []
        dtypes = []
        for band in bands:
            nodatavals.append(self._data[band].attrs.get('fill_value', 0))
            scale_factors.append(self._data[band].attrs.get('scale_factor', 1))
            offsets.append(self._data[band].attrs.get('offset', 0))
            dtypes.append(self._data[band].data.dtype.name)

        for i, entry in file_register.iterrows():
            filepath = entry['filepath']
            geom_id = entry.get('geom_id', 1)
            layer_id = entry.get('layer_id')
            tile = self._mosaic[geom_id]

            with GeoTiffDriver(filepath, mode='w', geotrans=tile.geotrans, sref_wkt=tile.sref.wkt,
                               shape=tile.shape, bands=bands, scale_factors=scale_factors, offsets=offsets,
                               nodatavals=nodatavals, np_dtypes=dtypes) as gt_driver:
                xrds = self._data[{self._file_dim: layer_id}]
                gt_driver.write(xrds[bands].to_array().data, bands)

    def __create_vrt_file(self, filepaths, vrt_filepath, bands=(1,)):
        n_filepaths = len(filepaths)
        n_bands = len(bands)
        n_rows = self._mosaic.tiles[0].parent_root.n_rows
        n_cols = self._mosaic.tiles[0].parent_root.n_cols

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
        geot_elem.text = ",".join(map(str, self._mosaic.tiles[0].geotrans))

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

    def __read_vrt_stack(self, row=0, col=0, n_rows=None, n_cols=None, bands=(1,), geom_id=1, reset_driver=False,
                         auto_decode=False, decoder=None, decoder_kwargs=None):
        n_bands = len(bands)
        reg_mask = self._file_register['geom_id'] == geom_id
        file_register = self._file_register[reg_mask]
        driver_id = file_register['driver_id'].iloc[0]
        driver_is_null = pd.isnull(driver_id)
        if driver_is_null or reset_driver:
            path = tempfile.gettempdir()
            date_str = datetime.utcnow().strftime("%Y%m%d%H%M%S")
            token = secrets.token_hex(8)
            tmp_filename = f"{date_str}_{token}.vrt"
            vrt_filepath = os.path.join(path, tmp_filename)
            self.__create_vrt_file(list(file_register['filepath']), vrt_filepath, bands=bands)
            src = gdal.Open(vrt_filepath, gdal.GA_ReadOnly)
            ref_driver = src
            driver_id = len(list(self._drivers.keys())) + 1
            self._drivers[driver_id] = ref_driver
            self._file_register.loc[reg_mask, 'driver_id'] = [driver_id] * len(file_register)
        else:
            ref_driver = self._drivers[driver_id]

        decoder_kwargs = {} if decoder_kwargs is None else decoder_kwargs

        data = dict()
        vrt_data = ref_driver.ReadAsArray(col, row, n_cols, n_rows)
        for band in bands:
            band_data = vrt_data[(band - 1)::n_bands, ...]
            scale_factor = ref_driver.GetRasterBand(band).GetScale()
            nodataval = ref_driver.GetRasterBand(band).GetNoDataValue()
            offset = ref_driver.GetRasterBand(band).GetOffset()
            if auto_decode:
                band_data = band_data.astype(float)
                band_data[band_data == nodataval] = np.nan
                band_data = band_data * scale_factor + offset
            else:
                if decoder is not None:
                    dtype = ref_driver.GetRasterBand(band).GetDataType()
                    np_dtype = r_gdal_dtype[dtype]
                    band_data = decoder(band_data, nodataval=nodataval, band=band, scale_factor=scale_factor,
                                        offset=offset,
                                        dtype=np_dtype, **decoder_kwargs)
            data[band] = band_data

        return data

    def __read_parallel(self):
        pass

    def _to_xarray(self, data, nodatavals):
        spatial_dims = ['y', 'x']
        dims = [self._file_dim] + spatial_dims

        coord_dict = dict()
        for coord in self._file_coords:
            if coord == 'idx':
                coord_dict[coord] = range(self.n_layers)
            else:
                coord_dict[coord] = self._file_register[coord]
        coord_dict['x'] = self._mosaic.tiles[0].x_coords
        coord_dict['y'] = self._mosaic.tiles[0].y_coords

        xar_dict = dict()
        for band in data.keys():
            xar_dict[band] = xr.DataArray(data[band], coords=coord_dict, dims=dims,
                                          attrs={'fill_value': nodatavals[band-1]})

        xrds = xr.Dataset(data_vars=xar_dict)

        ref_tile = self._mosaic.tiles[0]
        xrds.rio.write_crs(ref_tile.sref.wkt, inplace=True)
        xrds.rio.write_transform(Affine(*ref_tile.geotrans), inplace=True)

        return xrds




class GeoTiffDriver:
    def __init__(self, filepath, mode='r', geotrans=(0, 1, 0, 0, 0, 1), sref_wkt=None, shape=(1, 1), compression='LZW',
                 metadata=None, is_bigtiff=False, is_tiled=True, blocksizes=((512,512), ), overwrite=False,
                 bands=(1,), scale_factors=(1,), offsets=(0,), nodatavals=(255,), np_dtypes=('uint8',),
                 auto_decode=False):

        self.src = None
        self.driver = gdal.GetDriverByName('GTiff')
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
            self.src = self.driver.Create(self.filepath, self.shape[1], self.shape[0],
                                          self.n_bands, self.dtypes[0],
                                          options=gdal_opt)

            self.src.SetGeoTransform(self.geotrans)
            self.src.SetProjection(self.sref_wkt)

            if self.metadata is not None:
                self.src.SetMetadata(self.metadata)
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


if __name__ == '__main__':
    import glob
    from shapely.geometry import Polygon
    #import matplotlib.pyplot as plt
    polygon = Polygon(((50.860565, -0.791828), (51.404334, -0.570728), (50.852763, -0.162861)))
    polygon = ogr.CreateGeometryFromWkt(polygon.wkt)
    filepaths_E041N022T1 = glob.glob(r'D:\data\code\yeoda\2021_11_29__rq_workshop\data\Sentinel-1_CSAR\IWGRDH\parameters\datasets\par\B0104\EQUI7_EU010M\E041N022T1\mmensig0\*VV*.tif')
    filepaths_E041N021T1 = glob.glob(
        r'D:\data\code\yeoda\2021_11_29__rq_workshop\data\Sentinel-1_CSAR\IWGRDH\parameters\datasets\par\B0104\EQUI7_EU010M\E041N021T1\mmensig0\*VV*.tif')
    filepaths = filepaths_E041N021T1 + filepaths_E041N022T1
    gt_stack = GeoTiffStack.from_mosaic_filepaths(filepaths)
    gt_stack.select_polygon(polygon, sref=SpatialRef(4326))
    data = gt_stack.read()

    n_rows = gt_stack._mosaic.tiles[0].n_rows
    data_1 = data.sel(y=slice(None, data.y.data[int(n_rows/3)]))
    data_2 = data.sel(y=slice(data.y.data[int(2*n_rows / 3)], None))

    out_dirpath = r'D:\data\tmp\veranda\write_test'
    file_register_dict = dict()
    layer_ids = list(range(1, gt_stack.n_layers))
    out_filepaths = [os.path.join(out_dirpath, f"{layer_id}.tif") for layer_id in layer_ids]
    file_register_dict['filepath'] = out_filepaths
    file_register_dict['geom_id'] = [gt_stack._mosaic.tile_names[0]] * len(out_filepaths)
    file_register_dict['layer_id'] = layer_ids
    out_file_register = pd.DataFrame(file_register_dict)
    with GeoTiffStack.from_other(gt_stack, file_register=out_file_register) as gt_stack_out:
        gt_stack_out.write(data_1)
        gt_stack_out.write(data_2)
    #gt_stack.export(file_register=out_file_register)
    #gt_stack_out = GeoTiffStack.from_other(gt_stack, file_register=out_file_register)
    #gt_stack_ts = gt_stack_out.select_xy(4149234, 2189107)
    #gt_stack_ts = gt_stack_out.select_xy(4149234, 2189107)
    #gt_stack_ts = gt_stack_out.select_xy(4149234, 2189107)
    #gt_stack_out.export()
    #gt_stack.apply_nan()
    #plt.imshow(gt_stack._data[1].data[3, ...])
    #plt.show()
    #data = gt_stack.read(0, 0, 1, 1)
    pass
