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

PySLM is building up a core feature set aiming to provide the basic blocks for primarily generating the scan paths and
additional design features used for AM systems typically (SLM/SLS/SLA) systems which consolidate material using
a single/multi point exposure by generating a series of scan vectors in a region.

**Support Structure Generation**
* [TODO] A prototype for support structure generation

**Slicing:**

* Slicing of triangular meshes supported via the `Trimesh <https://github.com/mikedh/trimesh>`_ library.

**Hatching:**

* Standard 'alternating' hatching is available
* Stripe Scan Strategy Available

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

Installation of pyslm can then be performed using pre-built python packages using the pypi project.

.. code:: bash

    pip install pyslm

Alternatively, pyslm may be compiled from source. Currently the prerequisites are the cython packagee and a c++ build environment.

.. code:: bash

    git clone https://github.com/drlukeparry/pyslm.git && cd ./pyslm
    python setup.py install

Usage
******
A basic example below, shows how relativly straightforward it is to generate a single layer from a STL mesh which generates
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


Documented examples are provided in `examples <https://github.com/drlukeparry/pyslm/tree/master/examples>`_ .
