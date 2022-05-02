import os
import secrets
import tempfile
import xarray as xr
import numpy as np
import pandas as pd
from osgeo import gdal

from datetime import datetime
from multiprocessing import Pool, RawArray

from geospade.crs import SpatialRef
from geospade.raster import Tile
from geospade.raster import MosaicGeometry

from veranda.raster.driver import GeoTiffDriver
from veranda.raster.driver import create_vrt_file
from veranda.raster.driver import GDAL_TO_NUMPY_DTYPE
from veranda.raster.data.base import RasterDataReader, RasterDataWriter, RasterAccess

PROC_OBJS = {}


def read_init(fr, am, sm, ad, dc, dk):
    """ Helper method for setting the entries of global variable `PROC_OBJS` to be available during multiprocessing. """
    PROC_OBJS['global_file_register'] = fr
    PROC_OBJS['access_map'] = am
    PROC_OBJS['shm_map'] = sm
    PROC_OBJS['auto_decode'] = ad
    PROC_OBJS['decoder'] = dc
    PROC_OBJS['decoder_kwargs'] = dk


class GeoTiffAccess(RasterAccess):
    """
    Helper class to build the link between indexes of the source array/tile (access) and the target array/tile
    (assignment). The base class `RasterAccess` is extended with some properties needed for accessing GeoTIFF files.

    """
    def __init__(self, src_raster_geom, dst_raster_geom, src_root_raster_geom=None):
        """
        Constructor of `GeoTiffAccess`.

        Parameters
        ----------
        src_raster_geom : geospade.raster.RasterGeometry
            Raster geometry representing the extent and indices of the data to access.
        dst_raster_geom : geospade.raster.RasterGeometry
            Raster geometry representing the extent and indices of the data to assign.
        src_root_raster_geom : geospade.raster.RasterGeometry, optional
            Raster geometry representing the origin to which `src_raster_geom` should be referred to. Defaults to None,
            i.e. the root parent of `src_raster_geom` is used.

        """
        super().__init__(src_raster_geom, dst_raster_geom, src_root_raster_geom=src_root_raster_geom)
        self.src_wkt = src_raster_geom.parent_root.sref.wkt
        self.src_geotrans = src_raster_geom.parent_root.geotrans
        self.src_shape = src_raster_geom.parent_root.shape

    @property
    def gdal_args(self):
        """ 4-tuple : Prepares the needed positional arguments for GDAL's `ReadAsArray()` function. """
        min_row, min_col, max_row, max_col = self.src_window
        n_cols, n_rows = max_col - min_col + 1, max_row - min_row + 1
        return min_col, min_row, n_cols, n_rows

    @property
    def read_args(self):
        """
        4-tuple : Prepares the needed positional arguments for the `read()` function of the internal GeoTIFF driver.

        """
        min_col, min_row, n_cols, n_rows = self.gdal_args
        return min_row, min_col, n_rows, n_cols


class GeoTiffDataReader(RasterDataReader):
    """ Allows to read and modify a stack of GeoTIFF data. """
    def __init__(self, file_register, mosaic, file_dimension='layer_id', file_coords=None):
        """
        Constructor of `GeoTiffDataReader`.

        Parameters
        ----------
        file_register : pd.Dataframe
            Data frame managing a stack/list of files containing the following columns:
                - 'filepath' : str
                    Full file path to a geospatial file.
                - 'layer_id' : int
                    Specifies an ID to which layer a file belongs to.
                - 'tile_id' : str or int
                    Tile name or ID to which tile a file belongs to.
        mosaic : geospade.raster.MosaicGeometry
            Mosaic representing the spatial allocation of the given files. The tiles of the mosaic have to match the
            ID's/names of the 'tile_id' column.
        file_dimension : str, optional
            Dimension/column name of the dimension, where to stack the files along (first axis), e.g. time, bands etc.
            Defaults to 'layer_id', i.e. the layer ID's are used as the main coordinates to stack the files.
        file_coords : list, optional
            Additional columns of `file_register` to use as coordinates. Defaults to None, i.e. only coordinates along
            `file_dimension` are used.

        """
        super().__init__(file_register, mosaic, file_dimension=file_dimension, file_coords=file_coords)

        ref_filepath = self._file_register['filepath'].iloc[0]
        with GeoTiffDriver(ref_filepath, 'r') as gt_driver:
            self.dtypes = gt_driver.dtypes
            self.nodatavals = gt_driver.nodatavals

    @classmethod
    def from_filepaths(cls, filepaths, mosaic_class=MosaicGeometry, mosaic_kwargs=None, tile_kwargs=None):
        """
        Creates a `GeoTiffDataReader` instance as one stack of GeoTIFF files.

        Parameters
        ----------
        filepaths : list of str
            List of full system paths to a GeoTIFF file.
        mosaic_class : geospade.raster.MosaicGeometry, optional
            Mosaic class used to manage the spatial properties of the file stack. If None, the most generic mosaic will be
            used by default. The initialised mosaic will only contain one tile.
        mosaic_kwargs : dict, optional
            Additional arguments for initialising `mosaic_class`.
        tile_kwargs : dict, optional
            Additional arguments for initialising a tile class associated with `mosaic_class`.

        Returns
        -------
        GeoTiffDataReader

        """
        mosaic_kwargs = mosaic_kwargs or dict()
        tile_kwargs = tile_kwargs or dict()

        n_filepaths = len(filepaths)
        file_register_dict = dict()
        file_register_dict['filepath'] = filepaths
        file_register_dict['geom_id'] = [1] * n_filepaths
        file_register_dict['layer_id'] = list(range(n_filepaths))
        file_register = pd.DataFrame(file_register_dict)

        ref_filepath = filepaths[0]
        with GeoTiffDriver(ref_filepath, 'r') as gt_driver:
            sref_wkt = gt_driver.sref_wkt
            geotrans = gt_driver.geotrans
            n_rows, n_cols = gt_driver.shape

        tile_class = mosaic_class.get_tile_class()
        tile = tile_class(n_rows, n_cols, sref=SpatialRef(sref_wkt), geotrans=geotrans, name=1, **tile_kwargs)
        mosaic_geom = mosaic_class.from_tile_list([tile], check_consistency=False, **mosaic_kwargs)

        return cls(file_register, mosaic_geom)

    @classmethod
    def from_mosaic_filepaths(cls, filepaths, mosaic_class=MosaicGeometry, mosaic_kwargs=None):
        """
        Creates a `GeoTiffDataReader` instance as multiple stacks of GeoTIFF files.

        Parameters
        ----------
        filepaths : list of str
            List of full system paths to a GeoTIFF file.
        mosaic_class : geospade.raster.MosaicGeometry, optional
            Mosaic class used to manage the spatial properties of the file stacks. If None, the most generic mosaic will be
            used by default.
        mosaic_kwargs : dict, optional
            Additional arguments for initialising `mosaic_class`.

        Returns
        -------
        GeoTiffDataReader

        """
        mosaic_kwargs = mosaic_kwargs or dict()
        file_register_dict = dict()
        file_register_dict['filepath'] = filepaths

        geom_ids = []
        layer_ids = []
        tiles = []
        tile_idx = 1
        tile_class = mosaic_class.get_tile_class()
        for filepath in filepaths:
            with GeoTiffDriver(filepath, 'r') as gt_driver:
                sref_wkt = gt_driver.sref_wkt
                geotrans = gt_driver.geotrans
                n_rows, n_cols = gt_driver.shape
            curr_tile = tile_class(n_rows, n_cols, sref=SpatialRef(sref_wkt), geotrans=geotrans, name=tile_idx)
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

        mosaic_geom = mosaic_class.from_tile_list(tiles, check_consistency=False, **mosaic_kwargs)

        return cls(file_register, mosaic_geom)

    def read(self, bands=(1,), band_names=None, engine='vrt', n_cores=1,
             auto_decode=False, decoder=None, decoder_kwargs=None):
        """
        Reads data from disk.

        Parameters
        ----------
        bands : tuple, optional
            The GeoTIFF bands of interest. Defaults to the first band, i.e. (1,).
        band_names : tuple, optional
            Names associated with the respective bands of the GeoTIFF files. Defaults to None, i.e. the band numbers
            will be used as a name.
        engine : str, optional
            Engine used in the background to read the data. The following options are available:
                - 'vrt' : Uses GDAL's VRT format to stack the GeoTIFF files per tile and load them as once. This option
                          yields good performance if the data is stored locally on one drive. Parallelisation is applied
                          across tiles.
                - 'parallel' : Reads file by file, but in a parallelised manner. This option yields good performance if
                               the data is stored on a distributed file system.
        n_cores : int, optional
            Number of cores used to read the data in a parallelised manner (defaults to 1).
        auto_decode : bool, optional
            True if data should be decoded according to the information available in its metadata (default).
        decoder : callable, optional
            Function allowing to decode data read from disk.
        decoder_kwargs : dict, optional
            Keyword arguments for the decoder.

        """
        new_tile = Tile.from_extent(self._mosaic.outer_extent, sref=self._mosaic.sref,
                                    x_pixel_size=self._mosaic.x_pixel_size,
                                    y_pixel_size=self._mosaic.y_pixel_size,
                                    name=1)
        shm_map = dict()
        for band in bands:
            np_dtype = np.dtype(self.dtypes[band - 1])
            self.nodatavals[band - 1] = np.array((self.nodatavals[band - 1])).astype(np_dtype)
            data_nshm = np.ones((self.n_layers, new_tile.n_rows, new_tile.n_cols), dtype=np_dtype) * self.nodatavals[band - 1]
            shm_ar_shape = data_nshm.shape
            c_dtype = np.ctypeslib.as_ctypes_type(data_nshm.dtype)
            shm_rar = RawArray(c_dtype, data_nshm.size)
            shm_data = np.frombuffer(shm_rar, dtype=np_dtype).reshape(shm_ar_shape)
            shm_data[:] = data_nshm[:]
            shm_map[band] = (shm_rar, shm_ar_shape)

        access_map = dict()
        data_mask = np.ones(new_tile.shape)
        for tile in self._mosaic.tiles:
            gt_access = GeoTiffAccess(tile, new_tile)
            data_mask[gt_access.dst_row_slice, gt_access.dst_col_slice] = tile.mask
            access_map[tile.parent_root.name] = gt_access

        if engine == 'vrt':
            self.__read_vrt_stack(access_map, shm_map, n_cores, auto_decode, decoder, decoder_kwargs)
        elif engine == 'parallel':
            self.__read_parallel(access_map, shm_map, n_cores, auto_decode, decoder, decoder_kwargs)
        else:
            err_msg = f"Engine '{engine}' is not supported!"
            raise ValueError(err_msg)

        data = dict()
        for band in shm_map.keys():
            shm_rar, shm_ar_shape = shm_map[band]
            shm_data = np.frombuffer(shm_rar, dtype=self.dtypes[band - 1]).reshape(shm_ar_shape)
            shm_data[:, ~data_mask.astype(bool)] = self.nodatavals[band - 1]
            data[band] = shm_data

        self._data_geom = new_tile
        self._data = self._to_xarray(data, band_names)
        self._add_grid_mapping()
        return self

    def __read_vrt_stack(self, access_map, shm_map, n_cores=1,
                         auto_decode=False, decoder=None, decoder_kwargs=None):
        """
        Reads GeoTIFF data from a stack of GeoTIFF files by using GDAL's VRT format.

        Parameters
        ----------
        access_map : dict
            Dictionary mapping tile/geometry ID's with `GeoTiffAccess` instances to define the access patterns between
            the data to load and to assign.
        shm_map : dict
            Dictionary mapping band numbers with the respective name of the memory buffer of the shared numpy raw array.
        n_cores : int, optional
            Number of cores used to read the data in a parallelised manner (defaults to 1).
        auto_decode : bool, optional
            True if data should be decoded according to the information available in its metadata (default).
        decoder : callable, optional
            Function allowing to decode data read from disk.
        decoder_kwargs : dict, optional
            Keyword arguments for the decoder.

        """
        decoder_kwargs = decoder_kwargs or None
        global_file_register = self._file_register

        with Pool(n_cores, initializer=read_init, initargs=(global_file_register, access_map, shm_map,
                                                            auto_decode, decoder, decoder_kwargs)) as p:
            p.map(read_vrt_stack, access_map.keys())

    def __read_parallel(self, access_map, shm_map, n_cores=1,
                        auto_decode=False, decoder=None, decoder_kwargs=None):
        """
        Reads GeoTIFF data on a file-by-file basis, in a parallel manner.

        Parameters
        ----------
        access_map : dict
            Dictionary mapping tile/geometry ID's with `GeoTiffAccess` instances to define the access patterns between
            the data to load and to assign.
        shm_map : dict
            Dictionary mapping band numbers with the respective name of the memory buffer of the shared numpy raw array.
        n_cores : int, optional
            Number of cores used to read the data in a parallelised manner (defaults to 1).
        auto_decode : bool, optional
            True if data should be decoded according to the information available in its metadata (default).
        decoder : callable, optional
            Function allowing to decode data read from disk.
        decoder_kwargs : dict, optional
            Keyword arguments for the decoder.

        """
        decoder_kwargs = decoder_kwargs or None
        global_file_register = self._file_register

        with Pool(n_cores, initializer=read_init, initargs=(global_file_register, access_map, shm_map,
                                                            auto_decode, decoder, decoder_kwargs)) as p:
            p.map(read_single_files, global_file_register.index)

    def _to_xarray(self, data, band_names=None):
        """
        Converts data being available as a NumPy array to an xarray dataset.

        Parameters
        ----------
        data : dict
            Dictionary mapping band numbers to GeoTIFF raster data being available as a NumPy array.
        band_names : list of str, optional
            Band names associated with the respective band number.

        Returns
        -------
        xrds : xr.Dataset

        """
        spatial_dims = ['y', 'x']
        dims = [self._file_dim] + spatial_dims

        coord_dict = dict()
        for coord in self._file_coords:
            if coord == 'layer_id':
                coord_dict[coord] = range(1, self.n_layers + 1)
            else:
                coord_dict[coord] = self._file_register[coord]
        coord_dict['x'] = self._data_geom.x_coords
        coord_dict['y'] = self._data_geom.y_coords

        xar_dict = dict()
        bands = list(data.keys())
        band_names = band_names or bands
        for i, band in enumerate(bands):
            xar_dict[band_names[i]] = xr.DataArray(data[band], coords=coord_dict, dims=dims,
                                                   attrs={'fill_value': self.nodatavals[band-1]})

        xrds = xr.Dataset(data_vars=xar_dict)

        return xrds


class GeoTiffDataWriter(RasterDataWriter):
    """ Allows to write and modify a stack of GeoTIFF data. """
    def __init__(self, mosaic, file_register=None, data=None, file_dimension='layer_id', file_coords=None, dirpath=None,
                 file_naming_pattern='{layer_id}.tif'):
        """
        Constructor of `GeoTiffDataWriter`.

        Parameters
        ----------
        mosaic : geospade.raster.MosaicGeometry
            Mosaic representing the spatial allocation of the given files. The tiles of the mosaic have to match the
            ID's/names of the 'tile_id' column.
        file_register : pd.Dataframe, optional
            Data frame managing a stack/list of files containing the following columns:
                - 'filepath' : str
                    Full file path to a geospatial file.
                - 'layer_id' : int
                    Specifies an ID to which layer a file belongs to.
                - 'tile_id' : str or int
                    Tile name or ID to which tile a file belongs to.
            If it is none, then it will be created from the information stored in `data`, `dirpath`, and
            `file_name_pattern`.
        data : xr.Dataset, optional
            Raster data stored in memory. It must match the spatial sampling and CRS of the mosaic, but not its spatial
            extent or tiling. Moreover, the dimension of the data along the first dimension (stack/file dimension), must
            match the entries/filepaths in `file_register`.
        file_dimension : str, optional
            Dimension/column name of the dimension, where to stack the files along (first axis), e.g. time, bands etc.
            Defaults to 'layer_id', i.e. the layer ID's are used as the main coordinates to stack the files.
        file_coords : list, optional
            Additional columns of `file_register` to use as coordinates. Defaults to None, i.e. only coordinates along
            `file_dimension` are used.
        dirpath : str, optional
            Directory path to the folder where the GeoTIFF files should be written to. Defaults to none, i.e. the
            current working directory is used.
        file_naming_pattern : str, optional
            Pattern for the filename of the new GeoTIFF files. To fill in specific parts of the new file name with
            information from the file register, you can specify the respective file register column names in curly
            brackets and add them to the pattern string as desired. Defaults to '{layer_id}.tif'.

        """

        if file_register is None and data is None:
            err_msg = "Either a file register ('file_register') or an xarray dataset ('data') has to be provided."
            raise ValueError(err_msg)

        if file_register is None:
            n_layers = len(data[file_dimension])
            layer_ids = range(n_layers)
            file_register_dict = dict()
            file_register_dict['layer_id'] = layer_ids
            file_register = pd.DataFrame(file_register_dict)

        n_entries = len(file_register)
        if 'layer_id' not in file_register.columns:
            layer_ids = range(n_entries)
            file_register['layer_id'] = layer_ids

        if 'geom_id' not in file_register.columns:
            file_register['geom_id'] = [0] * n_entries

        if 'filepath' not in file_register.columns:
            dirpath = dirpath or os.getcwd()
            filenames = [file_naming_pattern.format(**row) for _, row in file_register.iterrows()]
            filepaths = [os.path.join(dirpath, filename) for filename in filenames]
            file_register['filepath'] = filepaths

        super().__init__(file_register, mosaic, data=data, file_dimension=file_dimension, file_coords=file_coords)

    def write(self, data, encoder=None, encoder_kwargs=None, overwrite=False, **kwargs):
        """
        Writes a certain chunk of data to disk.

        Parameters
        ----------
        data : xr.Dataset
            Data chunk to be written to disk or being appended to existing data.
        encoder : callable, optional
            Function allowing to encode data before writing it to disk.
        encoder_kwargs : dict, optional
            Keyword arguments for the encoder.
        overwrite : bool, optional
            True if data should be overwritten, false if not (default).

        """
        data_geom = self.raster_geom_from_data(data)

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
            if not tile.intersects(data_geom):
                continue
            driver_id = entry.get('driver_id', None)
            if driver_id is None:
                gt_driver = GeoTiffDriver(filepath, mode='w', geotrans=tile.geotrans, sref_wkt=tile.sref.wkt,
                                          shape=tile.shape, bands=bands, scale_factors=scale_factors, offsets=offsets,
                                          nodatavals=nodatavals, dtypes=dtypes)
                driver_id = len(list(self._drivers.keys())) + 1
                self._drivers[driver_id] = gt_driver
                self._file_register.loc[i, 'driver_id'] = driver_id

            gt_driver = self._drivers[driver_id]
            out_tile = data_geom.slice_by_geom(tile, inplace=False)
            gt_access = GeoTiffAccess(out_tile, tile, src_root_raster_geom=data_geom)
            xrds = data.sel(**{self._file_dim: layer_id})
            data_write = xrds[bands].to_array().data
            gt_driver.write(data_write[..., gt_access.src_row_slice, gt_access.src_col_slice], bands,
                            row=gt_access.dst_window[0], col=gt_access.dst_window[1],
                            encoder=encoder, encoder_kwargs=encoder_kwargs)

    def export(self, apply_tiling=False, encoder=None, encoder_kwargs=None, overwrite=False, **kwargs):
        """
        Writes all the internally stored data to disk.

        Parameters
        ----------
        apply_tiling : bool, optional
            True if the internal data should be tiled according to the mosaic.
            False if the internal data composes a new tile and should not be tiled (default).
        encoder : callable, optional
            Function allowing to encode data before writing it to disk.
        encoder_kwargs : dict, optional
            Keyword arguments for the encoder.
        overwrite : bool, optional
            True if data should be overwritten, false if not (default).

        """
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
            layer_id = entry.get('layer_id')
            geom_id = entry.get('layer_id', 1)
            tile = self._mosaic[geom_id] if apply_tiling else self._data_geom

            with GeoTiffDriver(filepath, mode='w', geotrans=tile.geotrans, sref_wkt=tile.sref.wkt,
                               shape=tile.shape, bands=bands, scale_factors=scale_factors, offsets=offsets,
                               nodatavals=nodatavals, dtypes=dtypes) as gt_driver:
                xrds = self._data[{self._file_dim: layer_id}]
                gt_driver.write(xrds[bands].to_array().data, bands, encoder=encoder, encoder_kwargs=encoder_kwargs)


def read_vrt_stack(geom_id):
    """
    Function being responsible to create a new VRT file from a stack of GeoTIFF files, read data from this files, and
    assign it to a shared memory array. This function is meant to be executed in parallel on different cores.

    Parameters
    ----------
    geom_id : str or int
        Tile/geometry ID coming from the Pool's mapping function.

    """
    global_file_register = PROC_OBJS['global_file_register']
    access_map = PROC_OBJS['access_map']
    shm_map = PROC_OBJS['shm_map']
    auto_decode = PROC_OBJS['auto_decode']
    decoder = PROC_OBJS['decoder']
    decoder_kwargs = PROC_OBJS['decoder_kwargs']

    gt_access = access_map[geom_id]
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
    create_vrt_file(filepaths, vrt_filepath, gt_access.src_shape, gt_access.src_wkt, gt_access.src_geotrans,
                    bands=bands)

    src = gdal.Open(vrt_filepath, gdal.GA_ReadOnly)
    vrt_data = src.ReadAsArray(*gt_access.gdal_args)
    stack_idxs = np.array(layer_ids) - 1
    for band in bands:
        band_data = vrt_data[(band - 1)::n_bands, ...]
        scale_factor = src.GetRasterBand(band).GetScale()
        nodataval = src.GetRasterBand(band).GetNoDataValue()
        offset = src.GetRasterBand(band).GetOffset()
        dtype = GDAL_TO_NUMPY_DTYPE[src.GetRasterBand(band).DataType]
        if auto_decode:
            band_data = band_data.astype(float)
            band_data[band_data == nodataval] = np.nan
            band_data = band_data * scale_factor + offset
        else:
            if decoder is not None:
                band_data = decoder(band_data, nodataval=nodataval, band=band, scale_factor=scale_factor,
                                    offset=offset,
                                    dtype=dtype, **decoder_kwargs)

        shm_rar, shm_ar_shape = shm_map[band]
        shm_data = np.frombuffer(shm_rar, dtype=dtype).reshape(shm_ar_shape)
        shm_data[stack_idxs, gt_access.dst_row_slice, gt_access.dst_col_slice] = band_data


def read_single_files(file_idx):
    """
    Function being responsible to read data from a single GeoTIFF file and assign it to a shared memory array.
    This function is meant to be executed in parallel on different cores.

    Parameters
    ----------
    file_idx : any
        Index value to access a specific row of the file register. The actual value should come from the Pool's
        mapping function.

    """
    global_file_register = PROC_OBJS['global_file_register']
    access_map = PROC_OBJS['access_map']
    shm_map = PROC_OBJS['shm_map']
    auto_decode = PROC_OBJS['auto_decode']
    decoder = PROC_OBJS['decoder']
    decoder_kwargs = PROC_OBJS['decoder_kwargs']

    file_entry = global_file_register.loc[file_idx]
    layer_id = file_entry['layer_id']
    geom_id = file_entry['geom_id']
    filepath = file_entry['filepath']
    gt_access = access_map[geom_id]
    bands = list(shm_map.keys())

    with GeoTiffDriver(filepath, mode='r', auto_decode=auto_decode) as gt_file:
        gt_data = gt_file.read(*gt_access.read_args, bands=bands, decoder=decoder, decoder_kwargs=decoder_kwargs)
        for band in bands:
            dtype = gt_file.dtypes[band]
            shm_rar, shm_ar_shape = shm_map[band]
            shm_data = np.frombuffer(shm_rar, dtype=dtype).reshape(shm_ar_shape)
            shm_data[layer_id, gt_access.dst_row_slice, gt_access.dst_col_slice] = gt_data[(band - 1), ...]