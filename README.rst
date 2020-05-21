PySLM Python Library for Selective Laser Melting
==================================================

.. image:: https://github.com/drlukeparry/pyslm/workflows/Python%20application/badge.svg
    :target: https://github.com/drlukeparry/pyslm/actions
.. image:: https://readthedocs.org/projects/pyslm/badge/?version=latest
    :target: https://pyslm.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. image:: https://badge.fury.io/py/PythonSLM.svg
    :target: https://badge.fury.io/py/PythonSLM
.. image:: https://badges.gitter.im/pyslm/community.svg
    :target: https://gitter.im/pyslm/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge
    :alt: Chat on Gitter


PySLM is a python library for processing the input files used on Selective Laser Melting (SLM), Direct Metal Laser Sintering (DMLS)
platform typically used in both academia and industry for Additive Manufacturing. The core capabilities aim to include
slicing, hatching and support generation and providing  an interface to the binary build file formats available for platforms.
The library is built of core classes which may provide the basic functionality to generate the scan vectors used on systems
and also be used as building blocks to prototype and develop new algorithms.

PySLM is built-upon python libraries `Trimesh <https://github.com/mikedh/trimesh>`_ and based on some custom modifications
to the `PyClipper <https://pypi.org/project/pyclipper/>`_ libraries which are leveraged to provide the  slicing and
manipulation of polygons, such as offsetting and clipping of lines. Additional functionality will be added to provide basic capabilities.

The aims is this library provides especially for an academic environment, a useful set of tools for prototyping and used
in-conjunction with simulation and analytic studies.


Current Features
******************

PySLM is building up a core feature set aiming to provide the basic blocks for primarily generating the scan paths and
additional design features used for AM systems typically (SLM/SLS/SLA) systems which consolidate material using
a single/multi point exposure by generating a series of scan vectors in a region.

**Support Structure Generation**
* [TODO] A prototype for support structure generation

**Slicing:**

* Slicing of triangular meshes supported via the `Trimesh <https://github.com/mikedh/trimesh>`_ library.

**Hatching:**
The following operations are provided as a convenience to aid developing the scan strategies:

* Offsetting of contours and boundaries
* Trimming of lines and hatch vectors (sequentially ordered)

The following scan strategies have been implemented as reference on platforms:

* Standard 'Alternating' hatching
* Stripe Scan Strategy
* Island or Checkerboard Scan Strategy

**Visualisation:**

The laser scan vectors can be visualised using ``Matplotlib``. The order of the scan vectors can be shown to aid development
of the scan strategies.

**Export to Machine Files:**

Currently WIP to provide `libSLM <https://github.com/drlukeparry/libSLM>`_  to enable import and export from
* Renishaw MTT,
* DMG Mori Realizer,
* EOS CLI formats

Installation
*************
Installation is currently supported on Windows and Linux environments. The pre-requisites for using PySLM can be installed
via PyPi and/or Anaconda distribution.

.. code:: bash

    conda install -c conda-forge shapely, Rtree, networkx, scikit-image
    conda install trimesh

Installation of pyslm can then be performed using pre-built python packages using the PyPi repository.

.. code:: bash

    pip install PythonSLM

Alternatively, PySLM may be compiled from source. Currently the prerequisites are the cython package and a compliant c++ build environment.

.. code:: bash

    git clone https://github.com/drlukeparry/pyslm.git && cd ./pyslm
    python setup.py install

Usage
******
A basic example below, shows how relatively straightforward it is to generate a single layer from a STL mesh which generates
a the hatch infill using a Stripe Scan Strategy typically employed on some commercial systems to limit the maximum scan vector
length generated in a region.

.. code:: python

    import pyslm
    from pyslm import hatching as hatching

    # Imports the part and sets the geometry to  an STL file (frameGuide.stl)
    solidPart = pyslm.Part('myFrameGuide')
    solidPart.setGeometry('../models/frameGuide.stl')

    # Set te slice layer position
    z = 23.

    # Create a StripeHatcher object for performing any hatching operations
    myHatcher = hatching.StripeHatcher()
    myHatcher.stripeWidth = 5.0

    # Set the base hatching parameters which are generated within Hatcher
    myHatcher.hatchAngle = 10
    myHatcher.volumeOffsetHatch = 0.08
    myHatcher.spotCompensation = 0.06
    myHatcher.numInnerContours = 2
    myHatcher.numOuterContours = 1

    # Slice the object
    geomSlice = solidPart.getVectorSlice(z, returnCoordPaths = True)

    #Perform the hatching operations
    layer = myHatcher.hatch(geomSlice)

    # Plot the layer geometries
    hatching.Hatcher.plot(layer, plot3D=False, plotOrderLine=True) # plotArrows=True)


For further guidance please look at documented examples are provided in `examples <https://github.com/drlukeparry/pyslm/tree/master/examples>`_ .
