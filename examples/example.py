"""
A simple example showing how t use PySLM for generating a Stripe Scan Strategy across a single layer.
"""
import pyslm
from pyslm import hatching as hatching

# Imports the part and sets the geometry to  an STL file (frameGuide.stl)
solidPart = pyslm.Part('myFrameGuide')
solidPart.setGeometry('../models/frameGuide.stl')

print(solidPart.boundingBox)

# Set te slice layer position
z = 23.

# Create a StripeHatcher object for performing any hatching operations
myHatcher = hatching.IslandHatcher()
myHatcher.stripeWidth = 5.0

# Set the base hatching parameters which are generated within Hatcher
myHatcher.hatchAngle = 10
myHatcher.volumeOffsetHatch = 0.08
myHatcher.spotCompensation = 0.06
myHatcher.numInnerContours = 2
myHatcher.numOuterContours = 1

# Perform the slicing. Return coords paths should be set so they are formatted internally.
# This is internally performed using Trimesh to obtain a closed set of polygons.
# Further polygon simplification may be required to reduce excessive number of edges in the boundaries.

geomSlice = solidPart.getVectorSlice(z, returnCoordPaths = True)

#Perform the hatching operations
print('Hatching Started')
layer = myHatcher.hatch(geomSlice)
print('Completed Hatching')

# Note the hatches are ordered sequentially across the stripe. Additional sorting may be required to ensure that the
# the scan vectors are processed generally in one-direction from left to right.
# The stripes scan strategy will tend to provide the correct order per isolated region.

# Plot the layer geometries using matplotlib
# The order of scanning for the hatch region can be displayed by setting the parameter (plotOrderLine=True)
# Arrows can be enables by setting the parameter plotArrows to True
pyslm.visualise.plot(layer, plot3D=False, plotOrderLine=True) # plotArrows=True)


