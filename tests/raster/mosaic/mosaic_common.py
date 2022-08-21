import pytest
import numpy as np
import pandas as pd
import xarray as xr

from geospade.crs import SpatialRef
from geospade.raster import Tile
from geospade.raster import MosaicGeometry


def write_pixel_window(writer, row, col, height, width):
   writer_sel = writer.select_px_window(row, col, height, width)
   writer.write(writer_sel.data_view, use_mosaic=True)

   return writer


def assert_reader_data(ref_data, reader, row, col, height, width, data_var_names, **read_kwargs):
   reader_sel = reader.select_px_window(row, col, height=height, width=width, inplace=False)
   reader_sel.read(**read_kwargs)
   for data_var_name in data_var_names:
      np.testing.assert_equal(reader_sel.data_view[data_var_name],
                              ref_data[data_var_name].data[:, row:row + height, col:col + width])


@pytest.fixture
def xsize():
    return 60


@pytest.fixture
def ysize():
    return 50


@pytest.fixture
def tile(xsize, ysize):
    return Tile(ysize, xsize, SpatialRef(4326), name='0')


@pytest.fixture
def mosaic(tile):
    return MosaicGeometry.from_tile_list([tile])


@pytest.fixture
def simple_ds(tile):
    num_files = 50
    dims = ['time', 'y', 'x']
    coords = {'time': pd.date_range('2000-01-01', periods=num_files),
              'y': tile.y_coords,
              'x': tile.x_coords}
    data = np.random.randn(num_files, tile.n_rows, tile.n_cols)
    return xr.Dataset({'data': (dims, data)}, coords=coords)


@pytest.fixture
def complex_ds(tile):
    num_files = 25
    band_names = ['data1', 'data2']

    dims = ['time', 'y', 'x']
    coords = {'time': pd.date_range('2000-01-01', periods=num_files),
              'y': tile.y_coords,
              'x': tile.x_coords}
    data = np.ones((num_files, tile.n_rows, tile.n_cols))
    attr1 = {'unit': 'dB', 'scale_factor': 2, 'add_offset': 3, 'fill_value': -9999}
    attr2 = {'unit': 'dB', 'fill_value': -9999}
    return xr.Dataset({band_names[0]: (dims, data, attr1), band_names[1]: (dims, data, attr2)}, coords=coords)
