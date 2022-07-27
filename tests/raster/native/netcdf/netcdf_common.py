import os
import pytest
import numpy as np
import pandas as pd
import xarray as xr
from tempfile import mkdtemp


@pytest.fixture
def filepath():
    return os.path.join(mkdtemp(), 'test.nc')


@pytest.fixture
def simple_ds():
    data = np.ones((100, 100, 100), dtype=np.float32)
    dims = ['time', 'y', 'x']
    coords = {'time': pd.date_range('2000-01-01', periods=data.shape[0])}

    ds = xr.Dataset({'sig': (dims, data),
                     'inc': (dims, data)}, coords=coords)

    return ds


@pytest.fixture
def three_var_ds():
    data = np.ones((100, 100, 100), dtype=np.float32)
    dims = ['time', 'y', 'x']
    coords = {'time': pd.date_range('2000-01-01', periods=data.shape[0])}
    attr1 = {'unit': 'dB'}
    attr2 = {'unit': 'degree'}

    ds = xr.Dataset({'sig': (dims, data, attr1),
                     'inc': (dims, data, attr2),
                     'azi': (dims, data, attr2)}, coords=coords)
    return ds


@pytest.fixture
def complex_three_var_ds():
    data = np.ones((100, 100, 100), dtype=np.float32)
    dims = ['time', 'y', 'x']
    coords = {'time': pd.date_range('2000-01-01', periods=data.shape[0])}
    attr1 = {'unit': 'dB', 'scale_factor': 2, 'add_offset': 3, 'fill_value': -9999}
    attr2 = {'unit': 'degree', '_FillValue': -9999, 'scale_factor': 2, 'add_offset': 0}
    attr3 = {'unit': 'degree', '_FillValue': -9999}

    ds = xr.Dataset({'sig': (dims, data, attr1),
                     'inc': (dims, data, attr2),
                     'azi': (dims, data, attr3)}, coords=coords)

    return ds


@pytest.fixture
def nt_ds():
    data = np.ones((100, 100, 100), dtype=np.float32)
    dims = ['layer', 'y', 'x']
    coords = {'layer': range(data.shape[0])}
    attr1 = {'unit': 'dB'}
    attr2 = {'unit': 'degree'}

    ds = xr.Dataset({'sig': (dims, data, attr1),
                     'inc': (dims, data, attr2),
                     'azi': (dims, data, attr2)}, coords=coords)

    return ds
