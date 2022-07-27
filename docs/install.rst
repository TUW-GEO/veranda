============
Installation
============

The package can be either installed via pip or if you solely want to work with *veranda* or contribute, we recommend to
install it as a conda environment.

pip
===

To install *veranda* via pip in your own environment, use:

.. code-block:: console

   pip install veranda

conda
=====
The packages also comes along a conda environments: ``conda_environment.yml``.
This is especially recommended if you want to contribute to the project.
The following script will install miniconda and setup the environment on a UNIX
like system. Miniconda will be installed into ``$HOME/miniconda``.

.. code-block:: console

   wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
   bash miniconda.sh -b -p $HOME/miniconda
   export PATH="$HOME/miniconda/bin:$PATH"
   conda env create -f conda_environment.yml
   source activate veranda

This script adds ``$HOME/miniconda/bin`` temporarily to the ``PATH`` to do this
permanently add ``export PATH="$HOME/miniconda/bin:$PATH"`` to your ``.bashrc``
or ``.zshrc``.

For Windows, use the following setup:

    - Download the latest `miniconda 3 installer <https://docs.conda.io/en/latest/miniconda.html>`_ for Windows

    - Click on ``.exe`` file and complete the installation.

    - Add the folder ``condabin`` folder to your environment variable ``PATH``. You can find the ``condabin`` folder usually under: ``C:\Users\username\AppData\Local\Continuum\miniconda3\condabin``

    - Finally, you can set up the conda environment via:

        .. code-block:: console

            conda env create -f conda_environment.yml
            source activate veranda


After that you should be able to run

.. code-block:: console

   python setup.py test

to run the test suite.