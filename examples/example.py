"""
A simple example showing how t use PySLM for generating a Stripe Scan Strategy across a single layer.
"""
import numpy as np
import pyslm
from pyslm import hatching as hatching


# Imports the part and sets the geometry to  an STL file (frameGuide.stl)
solidPart = pyslm.Part('myFrameGuide')
solidPart.setGeometry('../models/frameGuide.stl')

"""
Transform the part:
Rotate the part 30 degrees about the Z-Axis - given in degrees
Translate by an offset of (5,10) and drop to the platform the z=0 Plate boundary
"""
solidPart.origin = [5.0, 10.0, 0.0]
solidPart.rotation = np.array([0, 0, 30])
solidPart.dropToPlatform()

print(solidPart.boundingBox)

# Set te slice layer position
z = 23.

# Create a BasicIslandHatcher object for performing any hatching operations (
myHatcher = hatching.BasicIslandHatcher()
myHatcher.stripeWidth = 5.0

# Set the base hatching parameters which are generated within Hatcher
myHatcher.hatchAngle = 10 # [°] The angle used for the islands
myHatcher.volumeOffsetHatch = 0.08 # [mm] Offset between internal and external boundary
myHatcher.spotCompensation = 0.06 # [mm] Additional offset to account for laser spot size
myHatcher.numInnerContours = 2
myHatcher.numOuterContours = 1
myHatcher.hatchSortMethod = hatching.AlternateSort()

"""
Perform the slicing. Return coords paths should be set so they are formatted internally.
This is internally performed using Trimesh to obtain a closed set of polygons.
Further polygon simplification may be required to reduce excessive number of edges in the boundaries.
"""
geomSlice = solidPart.getVectorSlice(z)

#Perform the hatching operations
print('Hatching Started')
layer = myHatcher.hatch(geomSlice)
print('Completed Hatching')

"""
Note the hatches are ordered sequentially across the stripe. Additional sorting may be required to ensure that the
the scan vectors are processed generally in one-direction from left to right.
The stripes scan strategy will tend to provide the correct order per isolated region.
"""

"""
Plot the layer geometries using matplotlib
The order of scanning for the hatch region can be displayed by setting the parameter (plotOrderLine=True)
Arrows can be enables by setting the parameter plotArrows to True
"""
pyslm.visualise.plot(layer, plot3D=False, plotOrderLine=True, plotArrows=False)


