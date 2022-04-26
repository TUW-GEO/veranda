import warnings
import secrets
import tempfile
import rioxarray
import xarray as xr
import numpy as np
import pandas as pd
import shapely.wkt
from osgeo import gdal
from affine import Affine
from datetime import datetime
from multiprocessing import shared_memory, Pool, RawArray
#from threading

from geospade.tools import any_geom2ogr_geom
from geospade.tools import rel_extent
from geospade.tools import rasterise_polygon
from geospade.crs import SpatialRef
from geospade.transform import transform_geom
from geospade.raster import Tile
from geospade.raster import MosaicGeometry

from veranda.driver.geotiff import GeoTiffDriver
from veranda.driver.geotiff import r_gdal_dtype
from veranda.driver.geotiff import create_vrt_file
from veranda.raster.base import RasterData

PROC_OBJS = {}


def read_init(fr, pm, sm, ad, dc, dk):
    PROC_OBJS['global_file_register'] = fr
    PROC_OBJS['px_map'] = pm
    PROC_OBJS['shm_map'] = sm
    PROC_OBJS['auto_decode'] = ad
    PROC_OBJS['decoder'] = dc
    PROC_OBJS['decoder_kwargs'] = dk


class GeoTiffData(RasterData):
    def __init__(self, file_register, mosaic, data=None, file_dimension='idx', file_coords=None):
        super().__init__(file_register, mosaic, data=data, file_dimension=file_dimension, file_coords=file_coords)

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

    def select_xy(self, x, y, sref=None, inplace=True):
        return super().select_xy(x, y, sref=sref, inplace=inplace, child_class=GeoTiffData)

    def select_polygon(self, polygon, sref=None, apply_mask=True, inplace=True):
        return super().select_polygon(polygon, sref=sref, apply_mask=apply_mask, inplace=inplace)

    def read(self, bands=(1,), engine='vrt', n_cores=1,
             auto_decode=False, decoder=None, decoder_kwargs=None):

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
        shm_map = dict()
        for band in bands:
            np_dtype = np.dtype(dtypes[band - 1])
            nodatavals[band - 1] = np.array((nodatavals[band - 1])).astype(np_dtype)
            data_nshm = np.ones((self.n_layers, new_tile.n_rows, new_tile.n_cols), dtype=np_dtype) * nodatavals[band - 1]
            shm_ar_shape = data_nshm.shape
            c_dtype = np.ctypeslib.as_ctypes_type(data_nshm.dtype)
            shm_rar = RawArray(c_dtype, data_nshm.size)
            shm_data = np.frombuffer(shm_rar, dtype=np_dtype).reshape(shm_ar_shape)
            shm_data[:] = data_nshm[:]
            shm_map[band] = (shm_rar, shm_ar_shape)

        px_map = dict()
        data_mask = np.ones(new_tile.shape)
        for tile in self._mosaic.tiles:
            origin = tile.parent.ul_x, tile.parent.ul_y
            col, row, _, _ = rel_extent(origin, tile.coord_extent, tile.x_pixel_size, tile.y_pixel_size)
            n_rows, n_cols = tile.shape

            origin = new_tile.ul_x, new_tile.ul_y
            min_col, min_row, max_col, max_row = rel_extent(origin, tile.coord_extent, tile.x_pixel_size,
                                                            tile.y_pixel_size)
            # TODO: harmonize px slices
            px_map[tile.parent_root.name] = ((col, row, n_cols, n_rows), (min_col, min_row, max_col, max_row),
                                             tile.parent_root.sref.wkt, tile.parent_root.geotrans,
                                             tile.parent_root.shape)
            data_mask[min_row:max_row + 1, min_col: max_col + 1] = tile.mask

        if engine == 'vrt':
            self.__read_vrt_stack(px_map, shm_map, n_cores, auto_decode, decoder, decoder_kwargs)
        elif engine == 'parallel':
            self.__read_parallel()
        else:
            err_msg = f"Engine '{engine}' is not supported!"
            raise ValueError(err_msg)

        data = dict()
        for band in shm_map.keys():
            shm_rar, shm_ar_shape = shm_map[band]
            shm_data = np.frombuffer(shm_rar, dtype=dtypes[band - 1]).reshape(shm_ar_shape)
            shm_data[:, ~data_mask.astype(bool)] = nodatavals[band - 1]
            data[band] = shm_data

        self._mosaic = self._mosaic._from_sliced_tiles([new_tile])
        self._data = self._to_xarray(data, nodatavals)
        return self._data

    def write(self, data, overwrite=False):
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
            xrds = data.sel(**{self._file_dim: layer_id})
            data_write = xrds[bands].to_array().data
            gt_driver.write(data_write[..., min_row: max_row + 1, min_col: max_col + 1], bands, row=row, col=col)

    def export(self, overwrite=False):
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

        for i, entry in self._file_register.iterrows():
            filepath = entry['filepath']
            geom_id = entry.get('geom_id', 1)
            layer_id = entry.get('layer_id')
            tile = self._mosaic[geom_id]

            with GeoTiffDriver(filepath, mode='w', geotrans=tile.geotrans, sref_wkt=tile.sref.wkt,
                               shape=tile.shape, bands=bands, scale_factors=scale_factors, offsets=offsets,
                               nodatavals=nodatavals, np_dtypes=dtypes) as gt_driver:
                xrds = self._data[{self._file_dim: layer_id}]
                gt_driver.write(xrds[bands].to_array().data, bands)

    def __read_vrt_stack(self, pm, sm, n_cores=1,
                         ad=False, dc=None, dk=None):

        px_map = pm
        shm_map = sm
        auto_decode = ad
        decoder = dc
        decoder_kwargs = {} if dk is None else dk

        global_file_register = self._file_register

        with Pool(n_cores, initializer=read_init, initargs=(global_file_register, px_map, shm_map,
                                                             auto_decode, decoder, decoder_kwargs)) as p:
            p.map(read_vrt_stack, px_map.keys())

    def __read_parallel(self):
        pass

    def _to_xarray(self, data, nodatavals):
        spatial_dims = ['y', 'x']
        dims = [self._file_dim] + spatial_dims

        coord_dict = dict()
        for coord in self._file_coords:
            if coord == 'layer_id':
                coord_dict[coord] = range(1, self.n_layers + 1)
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


class GeoTiffData(RasterData):
    def __init__(self, file_register, mosaic, data=None, file_dimension='idx', file_coords=None):
        super().__init__(file_register, mosaic, data=data, file_dimension=file_dimension, file_coords=file_coords)

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

    def select_xy(self, x, y, sref=None, inplace=True):
        return super().select_xy(x, y, sref=sref, inplace=inplace, child_class=GeoTiffData)

    def select_bbox(self, bbox, sref=None, inplace=True):
        return super().select_xy(x, y, sref=sref, inplace=inplace, child_class=GeoTiffData)

    def select_polygon(self, polygon, sref=None, apply_mask=True, inplace=True):
        return super().select_polygon(polygon, sref=sref, apply_mask=apply_mask, inplace=inplace, child_class=GeoTiffData)


def read_vrt_stack(geom_id):
    global_file_register = PROC_OBJS['global_file_register']
    px_map = PROC_OBJS['px_map']
    shm_map = PROC_OBJS['shm_map']
    auto_decode = PROC_OBJS['auto_decode']
    decoder = PROC_OBJS['decoder']
    decoder_kwargs = PROC_OBJS['decoder_kwargs']

    col, row, n_cols, n_rows = px_map[geom_id][0]
    min_col, min_row, max_col, max_row = px_map[geom_id][1]
    sref_wkt = px_map[geom_id][2]
    geotrans = px_map[geom_id][3]
    shape = px_map[geom_id][4]
    bands = list(shm_map.keys())
    n_bands = len(bands)
    file_register = global_file_register.loc[global_file_register['geom_id'] == geom_id]
    layer_ids = file_register['layer_id']

    path = tempfile.gettempdir()
    date_str = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    token = secrets.token_hex(8)
    tmp_filename = f"{date_str}_{token}.vrt"
    vrt_filepath = os.path.join(path, tmp_filename)
    filepaths = list(file_register['filepath'])
    create_vrt_file(filepaths, vrt_filepath, shape, sref_wkt, geotrans, bands=bands)

    src = gdal.Open(vrt_filepath, gdal.GA_ReadOnly)
    vrt_data = src.ReadAsArray(col, row, n_cols, n_rows)
    stack_idxs = np.array(layer_ids) - 1
    for band in bands:
        band_data = vrt_data[(band - 1)::n_bands, ...]
        scale_factor = src.GetRasterBand(band).GetScale()
        nodataval = src.GetRasterBand(band).GetNoDataValue()
        offset = src.GetRasterBand(band).GetOffset()
        dtype = src.GetRasterBand(band).DataType
        np_dtype = r_gdal_dtype[dtype]
        if auto_decode:
            band_data = band_data.astype(float)
            band_data[band_data == nodataval] = np.nan
            band_data = band_data * scale_factor + offset
        else:
            if decoder is not None:
                band_data = decoder(band_data, nodataval=nodataval, band=band, scale_factor=scale_factor,
                                    offset=offset,
                                    dtype=np_dtype, **decoder_kwargs)

        shm_rar, shm_ar_shape = shm_map[band]
        shm_data = np.frombuffer(shm_rar, dtype=np_dtype).reshape(shm_ar_shape)
        shm_data[stack_idxs, min_row:max_row + 1, min_col:max_col + 1] = band_data