# veranda
[![Build Status](https://travis-ci.com/TUW-GEO/veranda.svg?branch=master)](https://travis-ci.org/TUW-GEO/veranda)
[![Coverage Status](https://coveralls.io/repos/github/TUW-GEO/veranda/badge.svg?branch=master)](https://coveralls.io/github/TUW-GEO/veranda?branch=master)
[![PyPi Package](https://badge.fury.io/py/veranda.svg)](https://badge.fury.io/py/veranda)
[![RTD](https://readthedocs.org/projects/veranda/badge/?version=latest)](https://veranda.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Description
*veranda* stands for *"vector and raster data access"* and is a place for IO related classes and operations dealing 
with raster and vector data. Besides bridging the gap between rigid and complex packages like GDAL to increase 
user-friendliness and flexibility (similar to *rasterio*) it defines common ground to unite the world of raster and 
vector data and harmonise the entry point to access different data formats or multiple files.

*veranda* consist of two modules *raster* and *vector* each containing the submodules *native* and *mosaic*. *native* 
contains several classes for interacting with one file/data format, e.g. GeoTIFF or NetCDF. On the other hand, the 
*mosaic* module offers a datacube-like interface to work with multiple, structured files, which can be distributed based on a 
mosaic/grid in space or along a stack dimension, e.g. time, atmospheric layers, etc.

For further details we recommend to look at *veranda*'s documentation or tests. 


## Installation
The package can be either installed via pip or if you want to contribute, we recommend to 
install it as a conda environment.

### pip
To install *veranda* via pip in your own environment, use:
```
pip install veranda
```
**ATTENTION**: GDAL needs more OS support and has more dependencies then other packages and can therefore not be installed solely via pip.
Please have a look at https://pypi.org/project/GDAL/ what requirements are needed. Thus, for a fresh setup, an existing environment 
with working a GDAL installation is expected.

### conda
The packages also comes along with one conda environment ``conda_environment.yml``. 
This is especially recommended if you want to contribute to the project.
The following script will install miniconda and setup the environment on a UNIX
like system. Miniconda will be installed into ``$HOME/miniconda``.
```
wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
conda env create -f conda_environment.yml
source activate veranda
```
This script adds ``$HOME/miniconda/bin`` temporarily to the ``PATH`` to do this
permanently add ``export PATH="$HOME/miniconda/bin:$PATH"`` to your ``.bashrc``
or ``.zshrc``.

For Windows, use the following setup:
  * Download the latest [miniconda 3 installer](https://docs.conda.io/en/latest/miniconda.html) for Windows
  * Click on ``.exe`` file and complete the installation.
  * Add the folder ``condabin`` folder to your environment variable ``PATH``. 
    You can find the ``condabin`` folder usually under: ``C:\Users\username\AppData\Local\Continuum\miniconda3\condabin``
  * Finally, you can set up the conda environment via:
    ```
    conda env create -f conda_environment.yml
    source activate veranda
    ```
    
After that you should be able to run 
```
python setup.py test
```
to run the test suite.


## Contribution

We are happy if you want to contribute. Please raise an issue explaining what
is missing or if you find a bug. We will also gladly accept pull requests
against our master branch for new features or bug fixes.
If you want to contribute please follow these steps:

  * Fork the *veranda* repository to your account
  * Clone the *veranda* repository
  * Make a new feature branch from the *veranda* master branch
  * Add your feature
  * Please include tests for your contributions in one of the test directories.
    We use *py.test* so a simple function called ``test_my_feature`` is enough
  * Submit a pull request to our master branch
  
## Outlook
The next major release will contain significant support for vector data including IO for SHP and LASZ files.
In addition the *raster* module will be extended to allow for accessing ZARR or HDF data for performant time series queries. 

## Citation
If you use this software in a publication then please cite it using the Zenodo DOI.

## Note
This project has been set up using PyScaffold 3.2.2. For details and usage
information on PyScaffold see https://pyscaffold.org/.