"""
A simple example showing the basic structure and layout required for generating a machine build file for a SLM System. This is automatically built into the hatching classes, but for simple experiments or simulations it can be useful to create these manually.
"""
import pyslm

# The  structures necessary for creating a machine build file should be imported from the geometry submodule. Fallback python classes that are equivalent to those in libSLM are provided to ensure prototyping can take place.

from pyslm import geometry as slm

# A header is needed to include an internal filename. This is used as a descriptor internally for the Machine Build File. The translator Writer in libSLM will specify the actual filename.
header = slm.Header()
header.filename = "MachineBuildFile"

# Depending on the file format the version should be provided as a tuple
header.version = (1,0)

# The zUnit is the uniform layer thickness as an integer unit in microns
header.zUnit = 40 # μm 

# A BuildStyle represents a set of laser parameters used for scanning a particular layer geometry. The set of parameters typically inclueds laser power, laser speed, exposure times and distances for Q-Switched laser sources. Each BuildStyle should have a unique id (.bid) for each mid used.
contourBuildStyle = slm.BuildStyle()
contourBuildStyle.bid = 1

# Set the laser parameters for the Contour Build Style
contourBuildStyle.laserPower = 200 # W
contourBuildStyle.laserSpeed = 500 # mm/s - Note this is used on some systems but should be set
contourBuildStyle.laserFocus = 0.0 # mm - (Optional) Some new systems can modify the focus position real-time. 

# The point exposure parameters are specified for some systems (typically Q-Switch Pulse Lasers)
# Note: the laser speed v = pointDistance / pointExposureTime
contourBuildStyle.pointDistance = 50 # μm - Distance between exposure points
contourBuildStyle.pontExposureTime = 80 # ms - Exposure time

# A model represents a container with a unique ID (.mid) which has a set of BuildStyles which are assigned to various LayerGeometry features within a build file. 
model = slm.Model()
model.mid = 1
model.name = "Model A"

# Add the BuildStyle to 
model.buildStyles.append(contourBuildStyle)


# Create a contour geometry feature
# A contour geometry feature uses connected points to represent the scan vectors
contourGeom = slm.ContourGeometry()

# Assign the specific BuildStyle from a chosen Model
contourGeom.mid = model.mid
contourGeom.bid = contourBuildStyle.bid

# Assign a set of coordinates
coords = np.array([(0.0, 0.0),
                   (0.0, 100.0)
                   (100.0, 100.0)
                   (100.0, 0.0)
                   (0.0, 0.0)])

contourGeom.coords = coords

# Create a Layer to store the LayerGeometry objects. Each layer contains a set of
# LayerGeometry. A unique id (.id) must be provided and its z position (.z)

# Note: the z is an integer increment of the layer thickness e.g. z=3 with a layer
# thickness of 40 μm infers a current layer position  of 120 μm.
layer = slm.Layer(id = 0, z = 0)

# Added the Layer Geometry in sequential order
layer.geometry.append(contourGeom)




