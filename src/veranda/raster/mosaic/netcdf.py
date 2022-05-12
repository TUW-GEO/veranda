import os
import xarray as xr
import numpy as np
import pandas as pd
import netCDF4
from netCDF4 import MFDataset

from geospade.crs import SpatialRef
from geospade.raster import Tile
from geospade.raster import MosaicGeometry

from veranda.utils import to_list
from veranda.raster.native.netcdf import NetCdf4File
from veranda.raster.mosaic.base import RasterDataReader, RasterDataWriter, RasterAccess


class NetCdfReader(RasterDataReader):
    """ Allows to read and modify a stack of NetCDF files. """
    def __init__(self, file_register, mosaic, file_dimension='layer_id', file_coords=None):
        """
        Constructor of `NetCdfReader`.

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
        with NetCdf4File(ref_filepath, 'r') as nc_file:
            self.data_variables = nc_file.data_variables
            self.nodatavals = nc_file.nodatavals
            self.scale_factors = nc_file.nodatavals
            self.offsets = nc_file.offsets
            self.dtypes = nc_file.dtypes
            self.metadata = nc_file.metadata

    @classmethod
    def from_filepaths(cls, filepaths, mosaic_class=MosaicGeometry, mosaic_kwargs=None, tile_kwargs=None):
        """
        Creates a `NetCdf4Reader` instance as one stack of NetCDF files.

        Parameters
        ----------
        filepaths : list of str
            List of full system paths to a NetCDF file.
        mosaic_class : geospade.raster.MosaicGeometry, optional
            Mosaic class used to manage the spatial properties of the file stack. If None, the most generic mosaic will be
            used by default. The initialised mosaic will only contain one tile.
        mosaic_kwargs : dict, optional
            Additional arguments for initialising `mosaic_class`.
        tile_kwargs : dict, optional
            Additional arguments for initialising a tile class associated with `mosaic_class`.

        Returns
        -------
        NetCdfReader

        """
        mosaic_kwargs = mosaic_kwargs or dict()
        tile_kwargs = tile_kwargs or dict()

        n_filepaths = len(filepaths)
        file_register_dict = dict()
        file_register_dict['filepath'] = filepaths
        file_register_dict['geom_id'] = ['0'] * n_filepaths
        file_register_dict['layer_id'] = list(range(n_filepaths))
        file_register = pd.DataFrame(file_register_dict)

        ref_filepath = filepaths[0]
        with NetCdf4File(ref_filepath, 'r') as nc_file:
            sref_wkt = nc_file.sref_wkt
            geotrans = nc_file.geotrans
            n_rows, n_cols = nc_file.raster_shape

        tile_class = mosaic_class.get_tile_class()
        tile = tile_class(n_rows, n_cols, sref=SpatialRef(sref_wkt), geotrans=geotrans, name='0', **tile_kwargs)
        mosaic_geom = mosaic_class.from_tile_list([tile], check_consistency=False, **mosaic_kwargs)

        return cls(file_register, mosaic_geom)

    @classmethod
    def from_mosaic_filepaths(cls, filepaths, mosaic_class=MosaicGeometry, mosaic_kwargs=None):
        """
        Creates a `NetCdf4Reader` instance as multiple stacks of NetCDF files.

        Parameters
        ----------
        filepaths : list of str
            List of full system paths to a NetCDF file.
        mosaic_class : geospade.raster.MosaicGeometry, optional
            Mosaic class used to manage the spatial properties of the file stacks. If None, the most generic mosaic
            will be used by default.
        mosaic_kwargs : dict, optional
            Additional arguments for initialising `mosaic_class`.

        Returns
        -------
        NetCdfReader

        """
        mosaic_kwargs = mosaic_kwargs or dict()
        file_register_dict = dict()
        file_register_dict['filepath'] = filepaths

        geom_ids = []
        layer_ids = []
        tiles = []
        tile_idx = 0
        tile_class = mosaic_class.get_tile_class()
        for filepath in filepaths:
            with NetCdf4File(filepath, 'r') as nc_file:
                sref_wkt = nc_file.sref_wkt
                geotrans = nc_file.geotrans
                n_rows, n_cols = nc_file.raster_shape
            curr_tile = tile_class(n_rows, n_cols, sref=SpatialRef(sref_wkt), geotrans=geotrans, name=str(tile_idx))
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

    def read(self, data_variables=None, engine='netcdf4', parallel=True, auto_decode=False,
             decoder=None, decoder_kwargs=None):
        """
        Reads mosaic from disk.

        Parameters
        ----------
        bands : tuple, optional
            The GeoTIFF bands of interest. Defaults to the first band, i.e. (1,).
        band_names : tuple, optional
            Names associated with the respective bands of the GeoTIFF files. Defaults to None, i.e. the band numbers
            will be used as a name.
        engine : str, optional
            Engine used in the background to read the mosaic. The following options are available:
                - 'vrt' : Uses GDAL's VRT format to stack the GeoTIFF files per tile and load them as once. This option
                          yields good performance if the mosaic is stored locally on one drive. Parallelisation is applied
                          across tiles.
                - 'parallel' : Reads file by file, but in a parallelised manner. This option yields good performance if
                               the mosaic is stored on a distributed file system.
        n_cores : int, optional
            Number of cores used to read the mosaic in a parallelised manner (defaults to 1).
        auto_decode : bool, optional
            True if mosaic should be decoded according to the information available in its metadata (default).
        decoder : callable, optional
            Function allowing to decode mosaic read from disk.
        decoder_kwargs : dict, optional
            Keyword arguments for the decoder.

        """
        data_variables = to_list(data_variables)
        new_tile = Tile.from_extent(self._mosaic.outer_extent, sref=self._mosaic.sref,
                                    x_pixel_size=self._mosaic.x_pixel_size,
                                    y_pixel_size=self._mosaic.y_pixel_size,
                                    name='0')
        if engine == 'netcdf4':
            data = self.__read_netcdf4(new_tile, data_variables=data_variables, auto_decode=auto_decode,
                                       decoder=decoder, decoder_kwargs=decoder_kwargs)
        elif engine == 'xarray':
            data = self.__read_xarray(new_tile, data_variables=data_variables, parallel=parallel, auto_decode=auto_decode,
                                      decoder=decoder, decoder_kwargs=decoder_kwargs)
        else:
            err_msg = f"Engine '{engine}' is not supported!"
            raise ValueError(err_msg)

        self._data_geom = new_tile
        self._data = data
        self._add_grid_mapping()
        return self

    def __read_netcdf4(self, new_tile, data_variables=None, auto_decode=False, decoder=None, decoder_kwargs=None):
        data_variables = data_variables or self.data_variables
        decoder_kwargs = decoder_kwargs or dict()
        data = []
        for tile in self._mosaic.tiles:
            geom_id = tile.name
            raster_access = RasterAccess(tile, new_tile)
            file_register = self.file_register.loc[self.file_register['geom_id'] == geom_id]
            filepaths = list(file_register['filepath'])
            nc_ds = MFDataset(filepaths, aggdim='time')
            nc_ds.set_auto_maskandscale(auto_decode)
            data_tile = dict()
            metadata = dict()
            for data_variable in data_variables:
                data_tile[data_variable] = nc_ds[data_variable][...,
                                                                raster_access.src_row_slice,
                                                                raster_access.src_col_slice]
                if tile.mask:
                    data_tile[data_variable][:, ~tile.mask.astype(bool)] = self.nodatavals[data_variable]
                if decoder:
                    data_tile[data_variable] = decoder(data_tile[data_variable],
                                          nodataval=self.nodatavals[data_variable],
                                          data_variable=data_variable, scale_factor=self.scale_factors[data_variable],
                                          offset=self.offsets[data_variable], dtype=self.dtypes[data_variable],
                                          **decoder_kwargs)
                metadata[data_variable] = NetCdf4File.get_metadata(nc_ds[data_variable])
            times = netCDF4.num2date(nc_ds['time'][:],
                                     units=getattr(nc_ds['time'], 'units', None),
                                     calendar=getattr(nc_ds['time'], 'calendar', 'standard'),
                                     only_use_cftime_datetimes=False,
                                     only_use_python_datetimes=True)
            data_xr = self._to_xarray(data_tile, tile, times, metadata)
            data.append(data_xr)

        return xr.combine_by_coords(data)

    def __read_xarray(self, new_tile, data_variables=None, parallel=True, auto_decode=False, decoder=None,
                      decoder_kwargs=None):
        data_variables = data_variables or self.data_variables
        decoder_kwargs = decoder_kwargs or dict()
        data = []
        for tile in self._mosaic.tiles:
            geom_id = tile.name
            raster_access = RasterAccess(tile, new_tile)
            file_register = self.file_register.loc[self.file_register['geom_id'] == geom_id]
            filepaths = file_register['filepath']
            xr_ds = xr.open_mfdataset(filepaths, concat_dim="time", combine="nested", data_vars='minimal',
                                      coords='minimal', compat='override', parallel=parallel,
                                      mask_and_scale=auto_decode)
            data_tile = dict()
            for data_variable in data_variables:
                xr_sliced = xr_ds[data_variable][..., raster_access.src_row_slice, raster_access.dst_col_slice]
                xr_sliced = xr_sliced.where(~tile.mask.astype(bool), self.nodatavals[data_variable])
                if decoder:
                    xr_sliced = decoder(xr_sliced,
                                        nodataval=self.nodatavals[data_variable],
                                        data_variable=data_variable, scale_factor=self.scale_factors[data_variable],
                                        offset=self.offsets[data_variable], dtype=self.dtypes[data_variable],
                                        **decoder_kwargs)
                data_tile[data_variable] = xr_sliced
            xr_ds = xr.Dataset(data_tile, coords=xr_ds.coords, attrs=xr_ds.attrs)
            data.append(xr_ds)

        return xr.combine_by_coords(data)

    def _to_xarray(self, data, tile, times, metadata):
        """
        Converts mosaic being available as a NumPy array to an xarray dataset.

        Parameters
        ----------
        data : dict
            Dictionary mapping band numbers to GeoTIFF raster mosaic being available as a NumPy array.
        band_names : list of str, optional
            Band names associated with the respective band number.

        Returns
        -------
        xrds : xr.Dataset

        """
        dims = ['time', 'y', 'x']
        coord_dict = dict()
        coord_dict['x'] = tile.x_coords
        coord_dict['y'] = tile.y_coords
        coord_dict['time'] = times

        xar_dict = dict()
        data_variables = list(data.keys())
        for i, data_variable in enumerate(data_variables):
            xar_dict[data_variable] = xr.DataArray(data[data_variable], coords=coord_dict, dims=dims,
                                                   attrs=metadata[data_variable])

        xrds = xr.Dataset(data_vars=xar_dict)

        return xrds


class NetCdfWriter(RasterDataWriter):
    """ Allows to write and modify a stack of NetCDF files. """
    def __init__(self, mosaic, file_register=None, data=None, file_dimension='layer_id', file_coords=None, dirpath=None,
                 fn_pattern='{layer_id}.tif'):
        """
        Constructor of `GeoTiffWriter`.

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
            If it is none, then it will be created from the information stored in `mosaic`, `dirpath`, and
            `file_name_pattern`.
        data : xr.Dataset, optional
            Raster mosaic stored in memory. It must match the spatial sampling and CRS of the mosaic, but not its spatial
            extent or tiling. Moreover, the dimension of the mosaic along the first dimension (stack/file dimension), must
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
        fn_pattern : str, optional
            Pattern for the filename of the new GeoTIFF files. To fill in specific parts of the new file name with
            information from the file register, you can specify the respective file register column names in curly
            brackets and add them to the pattern string as desired. Defaults to '{layer_id}.tif'.

        """

        super().__init__(mosaic, file_register=file_register, data=data, file_dimension=file_dimension,
                         file_coords=file_coords, dirpath=dirpath, fn_pattern=fn_pattern)

    @classmethod
    def from_data(self, data, filepath, mosaic=None, **kwargs):
        file_register_dict = dict()
        file_register_dict['layer_id'] = [0]
        file_register_dict['geom_id'] = [0]
        file_register_dict['filepath'] = [filepath]
        file_register = pd.DataFrame(file_register_dict)
        return super().from_xarray(data, file_register, mosaic=mosaic, **kwargs)

    def write(self, data, encoder=None, encoder_kwargs=None, overwrite=False, **kwargs):
        """
        Writes a certain chunk of mosaic to disk.

        Parameters
        ----------
        data : xr.Dataset
            Data chunk to be written to disk or being appended to existing mosaic.
        encoder : callable, optional
            Function allowing to encode mosaic before writing it to disk.
        encoder_kwargs : dict, optional
            Keyword arguments for the encoder.
        overwrite : bool, optional
            True if mosaic should be overwritten, false if not (default).

        """
        data_geom = self.raster_geom_from_data(data, sref=self.mosaic.sref)
        data_variables = list(data.data_vars)
        nodatavals = dict()
        scale_factors = dict()
        offsets = dict()
        dtypes = dict()
        for data_variable in data_variables:
            dtypes[data_variable] = data[data_variable].data.dtype.name
            nodatavals[data_variable] = data[data_variable].attrs.get('_FillValue', 0)
            scale_factors[data_variable] = data[data_variable].attrs.get('scale_factor', 1)
            offsets[data_variable] = data[data_variable].attrs.get('add_offset', 0)

        for filepath, file_group in self._file_register.groupby('filepath'):
            geom_id = file_group.iloc[0].get('geom_id', '0')
            file_coords = list(file_group[self._file_dim])
            tile = self._mosaic[geom_id]
            if not tile.intersects(data_geom):
                continue
            file_id = file_group.iloc[0].get('file_id', None)
            if file_id is None:
                gt_driver = NetCdf4File(filepath, mode='w', geotrans=tile.geotrans, sref_wkt=tile.sref.wkt,
                                        raster_shape=tile.shape, data_variables=data_variables, dtypes=dtypes,
                                        scale_factors=scale_factors, offsets=offsets, nodatavals=nodatavals,
                                        metadata=data.attrs, **kwargs)
                file_id = len(list(self._files.keys())) + 1
                self._files[file_id] = gt_driver
                self._file_register.loc[file_group.index, 'file_id'] = file_id

            nc_file = self._files[file_id]
            tile_write = data_geom.slice_by_geom(tile, inplace=False)
            raster_access = RasterAccess(tile_write, tile, src_root_raster_geom=data_geom)
            xrds = data.sel(**{self._file_dim: file_coords, 'y': tile_write.y_coords, 'x': tile_write.x_coords})
            data_write = xrds[data_variables]
            nc_file.write(data_write, row=raster_access.dst_window[0], col=raster_access.dst_window[1],
                          encoder=encoder, encoder_kwargs=encoder_kwargs)

    def export(self, apply_tiling=False, encoder=None, encoder_kwargs=None, overwrite=False, **kwargs):
        """
        Writes all the internally stored mosaic to disk.

        Parameters
        ----------
        apply_tiling : bool, optional
            True if the internal mosaic should be tiled according to the mosaic.
            False if the internal mosaic composes a new tile and should not be tiled (default).
        encoder : callable, optional
            Function allowing to encode mosaic before writing it to disk.
        encoder_kwargs : dict, optional
            Keyword arguments for the encoder.
        overwrite : bool, optional
            True if mosaic should be overwritten, false if not (default).

        """
        data_variables = list(self._data.data_vars)
        nodatavals = dict()
        scale_factors = dict()
        offsets = dict()
        dtypes = dict()
        for data_variable in data_variables:
            dtypes[data_variable] = self._data[data_variable].data.dtype.name
            nodatavals[data_variable] = self._data[data_variable].attrs.get('_FillValue', 0)
            scale_factors[data_variable] = self._data[data_variable].attrs.get('scale_factor', 1)
            offsets[data_variable] = self._data[data_variable].attrs.get('add_offset', 0)

        data_write = self._data[data_variables]
        for filepath, file_group in self._file_register.groupby('filepath'):
            geom_id = file_group.iloc[0].get('geom_id', '0')
            if apply_tiling:
                tile = self._mosaic[geom_id]
                if not tile.intersects(self._data_geom):
                    continue
                tile_write = self._data_geom.slice_by_geom(tile, inplace=False)
                data_write = data_write.sel(**{'y': tile_write.y_coords, 'x': tile_write.x_coords})
            else:
                tile_write = self._data_geom

            with NetCdf4File(filepath, mode='w', data_variables=data_variables, scale_factors=scale_factors,
                             offsets=offsets, nodatavals=nodatavals, dtypes=dtypes, geotrans=tile_write.geotrans,
                             sref_wkt=tile_write.sref.wkt, metadata=data_write.attrs, **kwargs) as nc_file:
                nc_file.write(data_write, encoder=encoder, encoder_kwargs=encoder_kwargs)


if __name__ == '__main__':
    pass