=======
veranda
=======

*veranda* stands for *"vector and raster data access"* and is a place for IO related classes and operations dealing
with raster and vector data. Besides bridging the gap between rigid and complex packages like GDAL to increase
user-friendliness and flexibility (similar to *rasterio*) it defines common ground to unite the world of raster and
vector data and harmonise the entry point to access different data formats or multiple files.

*veranda* consist of two modules *raster* and *vector* each containing the submodules *native* and *mosaic*. *native*
contains several classes for interacting with one file/data format, e.g. GeoTIFF or NetCDF. On the other hand, the
*mosaic* module offers a datacube-like interface to work with multiple, structured files, which can be distributed based on a
mosaic/grid in space or along a stack dimension, e.g. time, atmospheric layers, etc.



Contents
========

.. toctree::
   :maxdepth: 2

   Installation <install>
   Mosaic <notebooks/mosaic>
   Module Reference <api/modules>
   License <license>
   Authors <authors>
   Changelog <changelog>



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _toctree: http://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html
.. _reStructuredText: http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
.. _references: http://www.sphinx-doc.org/en/stable/markup/inline.html
.. _Python domain syntax: http://sphinx-doc.org/domains.html#the-python-domain
.. _Sphinx: http://www.sphinx-doc.org/
.. _Python: http://docs.python.org/
.. _Numpy: http://docs.scipy.org/doc/numpy
.. _SciPy: http://docs.scipy.org/doc/scipy/reference/
.. _matplotlib: https://matplotlib.org/contents.html#
.. _Pandas: http://pandas.pydata.org/pandas-docs/stable
.. _Scikit-Learn: http://scikit-learn.org/stable
.. _autodoc: http://www.sphinx-doc.org/en/stable/ext/autodoc.html
.. _Google style: https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings
.. _NumPy style: https://numpydoc.readthedocs.io/en/latest/format.html
.. _classical style: http://www.sphinx-doc.org/en/stable/domains.html#info-field-lists