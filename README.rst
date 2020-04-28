PySLM Python Library for Selective Laser Melting
==================================================

.. image:: https://github.com/drlukeparry/pyslm/workflows/Python%20application/badge.svg
    :target: https://github.com/drlukeparry/pyslm/actions
.. image:: https://readthedocs.org/projects/pyslm/badge/?version=latest
    :target: https://pyslm.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

PySLM is a python library for processing the input files used on Selective Laser Melting (SLM), Direct Metal Laser Sintering (DMLS)
platform typically used in both academia and industry for Additive Manufacturing. The core capabilities aim to include
slicing, hatching and support generation and providing  an interface to the binary build file formats available for platforms.
The library is built of core classes which may provide the basic functionality to generate the scan vectors used on systems
and also be used as building blocks to prototype and develop new alogirthms.

PySLM is built-upon python libraries `Trimesh <https://github.com/mikedh/trimesh>`_ and based on some custom modifications
to the `PyClipper <https://pypi.org/project/pyclipper/>`_ libraries which are leveraged to provide the  slicing and
manipulation of polygons, such as offsetting and clipping of lines. Additional functionality will be added to provide basic capabilities.

The aims is this library provides especially for an academic environment, a useful set of tools for prototyping and used
in-conjunction with simulation and analytic studies.


Current Features
******************

**Support Structure Generation**
* [TODO] A prototype for support structure generation

**Slicing:**

* Slicing of triangular meshes supported via the `Trimesh <https://github.com/mikedh/trimesh>`_ library.

**Hatching:**

* Standard 'alternating' hatching is available

**Visualisation:**

* The laser scan vectors can be visualised and

**Export to Machine Files:**

* Currently WIP to port previous c++ code and provide generic bindings to platforms (Renishaw MTT, Realizer, EOS CLI formats)

Installation
*************
Installation is currently supported on Windows and Linux environments. The prerequisties for using PySLM can be installed
via PyPi and/or Anaconda distribution.

.. code:: bash

    conda install -c conda-forge shapely, Rtree, networkx, scikit-image
    pip install trimesh

Installation of pyslm can then be performed

.. code:: bash

    pip install pyslm

or alternatively downloading the package directly. On Windows platforms the path of the executable needs to be initialised before use.

.. code:: python

    from pyslm.core import Simulation


Usage
******

Documented examples are provided in `examples <https://github.com/drlukeparry/pyslm/tree/master/examples>`_ .