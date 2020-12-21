# veranda
[![Build Status](https://travis-ci.com/TUW-GEO/veranda.svg?branch=master)](https://travis-ci.org/TUW-GEO/veranda)
[![Coverage Status](https://coveralls.io/repos/github/TUW-GEO/veranda/badge.svg?branch=master)](https://coveralls.io/github/TUW-GEO/veranda?branch=master)
[![PyPi Package](https://badge.fury.io/py/veranda.svg)](https://badge.fury.io/py/veranda)
[![RTD](https://readthedocs.org/projects/veranda/badge/?version=latest)](https://veranda.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Description
*veranda* stands for *"vector and raster data access"* and is a place for IO related classes and operations dealing 
with vector and raster data. Currently, there is only one module `io`, which adds support for GeoTIFF (`geotiff`) and 
NetCDF (`netcdf`) files and their image stack representations (`timestack`).

## Limitations and Outlook
Support for vector data is still missing, which could for instance include reading and writing Shape-Files or well-known 
data formats like CSV for storing point-based *in-situ* data.

Performant data access is a key-feature for data cubes storing Earth Observation (EO) data. 
The core-interface between higher-level data cubes (cf. *yeoda*) and the data stored on disk will be also
implemented in *veranda*, allowing efficient and unambiguous writing and reading of EO data.

## Installation
The package can be either installed via pip or if you solely want to work with *veranda* or contribute, we recommend to 
install it as a conda environment. If you work already with your own environment, please have look at ``requirements.txt``.

### pip
To install *veranda* via pip in your own environment, use:
```
pip install veranda
```

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
  
## Citation

If you use this software in a publication then please cite it using the Zenodo DOI.
Be aware that this badge links to the latest package version.

Please select your specific version at  (link to first release) to get the DOI of that version.
You should normally always use the DOI for the specific version of your record in citations.
This is to ensure that other researchers can access the exact research artefact you used for reproducibility.

You can find additional information regarding DOI versioning at <http://help.zenodo.org/#versioning>.

## Note

This project has been set up using PyScaffold 3.2.2. For details and usage
information on PyScaffold see https://pyscaffold.org/.