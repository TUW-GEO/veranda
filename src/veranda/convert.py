# import numpy as np
# import xarray as xr
#
#
# # TODO: can we represent a rotated array with xarray?
# # ToDO: change band to list of bands
# def convert_data_coding(data, coder, coder_kwargs=None, band=None):
#     """
#     Converts data values via a given coding function.
#     A band/data variable (`band`) needs to be given if one works with xarray data sets.
#
#     Parameters
#     ----------
#     data : numpy.ndarray or xarray.Dataset
#         Array-like object containing image pixel values.
#     coder : function
#         Coding function, which expects NumPy/Dask arrays.
#     coder_kwargs : dict, optional
#         Keyword arguments for the coding function.
#     band : str or int, optional
#         Band/data variable of xarray data set.
#
#     Returns
#     -------
#     numpy.ndarray or xarray.Dataset
#         Array-like object containing coded image pixel values.
#     """
#
#     code_kwargs = {} if coder_kwargs is None else coder_kwargs
#
#     if isinstance(data, xr.Dataset):
#         if band is None:
#             err_msg = "A band name has to be specified for coding the data."
#             raise KeyError(err_msg)
#         data[band].data = coder(data[band].data, **code_kwargs)
#         return data
#     elif isinstance(data, np.ndarray):
#         return coder(data, **code_kwargs)
#     else:
#         err_msg = "Data type is not supported for coding the data."
#         raise Exception(err_msg)
#
#
# # TODO: where should we put this?
# # TODO: create grid mapping name with it?
# def convert_data_type(data, *coord_args, data_type="numpy", band=None, dim_names=None):
#     """
#     Converts `data` into an array-like object defined by `data_type`. It accepts NumPy arrays or Xarray data sets and
#     can convert to Numpy arrays, Xarray data sets or Pandas data frames.
#
#     Parameters
#     ----------
#     data : numpy.ndarray or xarray.Dataset
#         Array-like object containing image pixel values.
#     data_type : str
#         Data type of the returned array-like structure. It can be:
#             - 'xarray': converts data to an xarray.Dataset
#             - 'numpy': convert data to a numpy.ndarray (default)
#             - 'dataframe': converts data to a grouped pandas.DataFrame
#     *coord_args : unzipped tuple of lists
#         Coordinate arguments defined as a list, e.g.:
#         - *(xs, ys, timestamps): contains a list of world system coordinates in X direction, a list of world
#         system coordinates in Y direction and a list of timestamps.
#     band : int or str, optional
#         Band number or data variable name to select from an xarray data set (relevant for an xarray -> numpy conversion).
#     dim_names : list of str, optional
#         List of dimension names having the same length as `*coord_args`. The default behaviour is ['y', 'x', 'time']
#         ATTENTION: The order needs to follow the same order as `*coord_args`!
#
#     Returns
#     -------
#     numpy.ndarray or xarray.Dataset
#         Array-like object containing image pixel values.
#     """
#
#     n_coord_args = len(coord_args)
#     if dim_names is None:
#         if n_coord_args == 2:
#             dim_names = ['y', 'x']
#         elif n_coord_args == 3:
#             dim_names = ['time', 'y', 'x']
#     else:
#         n_dim_names = len(dim_names)
#         if n_coord_args != n_dim_names:
#             err_msg = "Number of coordinate arguments ({}) " \
#                       "does not match number of dimension names ({}).".format(n_coord_args, n_dim_names)
#             raise Exception(err_msg)
#
#     if data_type == "xarray":
#         if isinstance(data, np.ndarray):
#             coords = OrderedDict()
#             for i, dim_name in enumerate(dim_names):
#                 coords[dim_name] = coord_args[i]
#             xr_ar = xr.DataArray(data, coords=coords, dims=dim_names)
#             conv_data = xr.Dataset(data_vars={band: xr_ar})
#         elif isinstance(data, xr.Dataset):
#             conv_data = data
#         else:
#             raise DataTypeMismatch(type(data), data_type)
#     elif data_type == "numpy":
#         if isinstance(data, xr.Dataset):
#             if band is None:
#                 err_msg = "Band/label/data variable argument is not specified."
#                 raise Exception(err_msg)
#             conv_data = np.array(data[band].data)
#         elif isinstance(data, np.ndarray):
#             conv_data = data
#         else:
#             raise DataTypeMismatch(type(data), data_type)
#     elif data_type == "dataframe":
#         xr_ds = convert_data_type(data, 'xarray', *coord_args, band=band, dim_names=dim_names)
#         conv_data = xr_ds.to_dataframe()
#     else:
#         raise DataTypeMismatch(type(data), data_type)
#
#     return conv_data