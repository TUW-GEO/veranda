from netcdf_common import *
from veranda.raster.native.netcdf import NetCdfXrFile


def test_read_write(filepath, three_var_ds):

    with NetCdfXrFile(filepath, mode='w', data_variables=['sig', 'inc', 'azi'],
                      nodatavals={'inc': -9999, 'azi': -9999}, dtypes={'inc': 'int32', 'azi': 'int32'}) as nc:
        nc.write(three_var_ds)

    with NetCdfXrFile(filepath) as nc:
        ds = nc.read()
        np.testing.assert_array_equal(ds['sig'][:], three_var_ds['sig'][:])
        np.testing.assert_array_equal(ds['inc'][:], three_var_ds['inc'][:])
        np.testing.assert_array_equal(ds['azi'][:], three_var_ds['azi'][:])


def test_use_auto_decoding(filepath, complex_three_var_ds):
    with NetCdfXrFile(filepath, mode='w') as nc:
        nc.write(complex_three_var_ds)

    with NetCdfXrFile(filepath, mode='r', auto_decode=True) as nc:
        ds = nc.read()
        np.testing.assert_array_equal(ds['sig'][:], complex_three_var_ds['sig'][:] * 2 + 3)
        np.testing.assert_array_equal(ds['inc'][:], complex_three_var_ds['inc'][:] * 2)
        np.testing.assert_array_equal(ds['azi'][:], complex_three_var_ds['azi'][:])


def test_ignore_auto_decoding(filepath, complex_three_var_ds):
    with NetCdfXrFile(filepath, mode='w') as nc:
        nc.write(complex_three_var_ds)

    with NetCdfXrFile(filepath, mode='r', auto_decode=False) as nc:
        ds = nc.read()
        np.testing.assert_array_equal(ds['sig'][:], complex_three_var_ds['sig'][:])
        np.testing.assert_array_equal(ds['inc'][:], complex_three_var_ds['inc'][:])
        np.testing.assert_array_equal(ds['azi'][:], complex_three_var_ds['azi'][:])


def test_chunksizes(filepath, simple_ds):
    chunksizes = (100, 10, 10)
    with NetCdfXrFile(filepath, mode='w', data_variables=list(simple_ds.data_vars), chunksizes=chunksizes) as nc:
        nc.write(simple_ds)

    with NetCdfXrFile(filepath, mode='r') as nc:
        ds = nc.read()
        assert ds['sig'].data.chunksize == chunksizes
        assert ds['inc'].data.chunksize == chunksizes


def test_time_units(filepath, simple_ds):
    time_units = 'days since 2000-01-01 00:00:00'
    with NetCdfXrFile(filepath, mode='w', attrs={'time': {'units': time_units}}) as nc:
        nc.write(simple_ds)

    with NetCdfXrFile(filepath) as nc:
        ds = nc.read()
        np.testing.assert_array_equal(pd.DatetimeIndex(ds['time'].data),
                                      simple_ds.coords['time'])


def test_geotransform(filepath, simple_ds):
    xdim = 100
    ydim = 100
    geotrans = (3000000.0, 500.0, 0.0, 1800000.0, 0.0, -500.0)
    with NetCdfXrFile(filepath, mode='w', geotrans=geotrans) as nc:
        nc.write(simple_ds)

    with NetCdfXrFile(filepath) as nc:
        ds = nc.read()

    x = geotrans[0] + (0.5 + np.arange(xdim)) * geotrans[1] + \
        (0.5 + np.arange(xdim)) * geotrans[2]
    y = geotrans[3] + (0.5 + np.arange(ydim)) * geotrans[4] + \
        (0.5 + np.arange(ydim)) * geotrans[5]

    np.testing.assert_array_equal(ds['x'].values, x)
    np.testing.assert_array_equal(ds['y'].values, y)


def test_non_temporal_read_and_write(filepath, nt_ds):
    with NetCdfXrFile(filepath, mode='w', data_variables=['sig', 'inc', 'azi'], stack_dims={'layer': None},
                      nodatavals={'inc': -9999, 'azi': -9999}, dtypes={'inc': 'int32', 'azi': 'int32'}) as nc:
        nc.write(nt_ds)

    with NetCdfXrFile(filepath) as nc:
        ds = nc.read()
        np.testing.assert_array_equal(ds['sig'][:], nt_ds['sig'][:])
        np.testing.assert_array_equal(ds['inc'][:], nt_ds['inc'][:])
        np.testing.assert_array_equal(ds['azi'][:], nt_ds['azi'][:])
