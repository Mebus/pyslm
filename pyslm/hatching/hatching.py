import pyclipper
import numpy as np

import abc
from typing import Any, Tuple, List
from skimage.measure import approximate_polygon, subdivide_polygon

from ..geometry import LayerGeometry, ContourGeometry, HatchGeometry, Layer


class BaseHatcher(abc.ABC):

    PYCLIPPER_SCALEFACTOR = 1e4
    """ 
    The scaling factor used for polygon clipping and offsetting in PyClipper for the decimal component of each polygon
    coordinate. This should be set to inverse of the required decimal tolerance e.g. 0.01 requires a minimum 
    scalefactor of 1e2. 
    """

    def __init__(self):
        pass

    def __str__(self):
        return 'BaseHatcher <{:s}>'.format(self.name)

    def scaleToClipper(self, feature):
        return pyclipper.scale_to_clipper(feature, BaseHatcher.PYCLIPPER_SCALEFACTOR)

    def scaleFromClipper(self, feature):
        return pyclipper.scale_from_clipper(feature, BaseHatcher.PYCLIPPER_SCALEFACTOR)

    @classmethod
    def error(cls):
        """
        Returns the accuracy of the polygon clipping depending on the chosen scale factor :attribute:`~hatching.BaseHatcher.PYCLIPPER_SCALEFACTOR`"
        """
        return 1./cls.PYCLIPPER_SCALEFACTOR

    @staticmethod
    def plot(layer: Layer, zPos=0, plotContours=True, plotHatches=True, plotPoints=True, plot3D=True, plotArrows = False, plotOrderLine = False, handle=None) -> None:
        """
        Plots the all the scan vectors and point exposures in the Layer Geometry which includes the
        :param layer: The Layer containing the Layer Geometry
        :param zPos: The position of the layer when using the 3D plot (optional)
        :param plotContours: Plots the inner hatch scan vectors
        :param plotHatches: Plots the hatch scan vectors
        :param plotPoints: Plots point exposures
        :param plot3D: Plots the layer in 3D
        :param plotArrows: Plot the direction of each scan vector. This reduces the plotting performance due to use of matplotlib annotations, should be disabled for large datasets
        :param plotOrderLine: Plots an additional line showing the order of vecctor scanning
        :param handle: Matplotlib handle to re-use
        """

        import matplotlib.pyplot as plt
        import matplotlib.colors
        import matplotlib.collections as mc

        if handle:
            fig = handle[0]
            ax = handle[1]

        else:
            if plot3D:
                from mpl_toolkits.mplot3d import Axes3D
                fig = plt.figure()
                ax = plt.axes(projection='3d')
            else:
                fig, ax = plt.subplots()

        ax.axis('equal')
        plotNormalize = matplotlib.colors.Normalize()

        if plotHatches:
            hatches = [hatchGeom.coords for hatchGeom in layer.hatches]

            if len(hatches) > 0:

                hatches = np.vstack([hatchGeom.coords for hatchGeom in layer.hatches])

                lc = mc.LineCollection(hatches,
                                       colors=plt.cm.rainbow(plotNormalize(np.arange(len(hatches)))),
                                       linewidths=0.5)

                if plotArrows and not plot3D:
                    for hatch in hatches:
                        midPoint = np.mean(hatch, axis=0)
                        delta = hatch[1,:] - hatch[0,:]

                        plt.annotate('',
                                     xytext=midPoint-delta*1e-4,
                                     xy=midPoint,
                                     arrowprops={'arrowstyle' : "->",'facecolor': 'black'})

                if plot3D:
                    ax.add_collection3d(lc, zs=zPos)

                if not plot3D and plotOrderLine:
                    ax.add_collection(lc)
                    midPoints = np.mean(hatches, axis=1)
                    idx6 = np.arange(len(hatches))
                    ax.plot(midPoints[idx6][:, 0], midPoints[idx6][:, 1])

                ax.add_collection(lc)



        if plotContours:
            for contourGeom in layer.contours:

                if contourGeom.type == 'inner':
                    lineColor = '#f57900';
                    lineWidth = 1
                elif contourGeom.type == 'outer':
                    lineColor = '#204a87';
                    lineWidth = 1.4
                else:
                    lineColor = 'k';
                    lineWidth = 0.7

                if plotArrows and not plot3D:
                    for i in range(contourGeom.coords.shape[0]-1):
                        midPoint = np.mean(contourGeom.coords[i:i+2], axis=0)
                        delta = contourGeom.coords[i+1, :] - contourGeom.coords[i, :]

                        plt.annotate('',
                                     xytext=midPoint - delta * 1e-4,
                                     xy=midPoint,
                                     arrowprops={'arrowstyle': "->", 'facecolor': 'black'})

                if plot3D:
                    ax.plot(contourGeom.coords[:, 0], contourGeom.coords[:, 1], zs=zPos, color=lineColor,
                            linewidth=lineWidth)
                else:
                    ax.plot(contourGeom.coords[:, 0], contourGeom.coords[:, 1], color=lineColor,
                            linewidth=lineWidth)

        if plotPoints:
            for pointsGeom in layer.points:
                ax.scatter(pointsGeom.coords[:, 0], pointsGeom.coords[:, 1], 'x')

        return fig, ax


    def offsetPolygons(self, polygons, offset: float):
        """
        Offsets the boundaries across a collection of polygons

        :param polygons:
        :param offset: The offset applied to the poylgon
        :return:
        """
        return [self.offsetBoundary(poly, offset) for poly in polygons]


    def offsetBoundary(self, paths, offset: float):
        """
        Offsets a single path for a single polygon

        :param paths:
        :param offset: The offset applied to the poylgon
        :return:
        """
        pc = pyclipper.PyclipperOffset()

        clipperOffset = self.scaleToClipper(offset)

        # Append the paths to libClipper offsetting algorithm
        for path in paths:
            pc.AddPath(self.scaleToClipper(path),
                       pyclipper.JT_ROUND,
                       pyclipper.ET_CLOSEDPOLYGON)

        # Perform the offseting operation
        boundaryOffsetPolys = pc.Execute2(clipperOffset)

        offsetContours = []
        # Convert these nodes back to paths
        for polyChild in boundaryOffsetPolys.Childs:
            offsetContours += self._getChildPaths(polyChild)

        return offsetContours


    def _getChildPaths(self, poly):

        offsetPolys = []

        # Create single closed polygons for each polygon
        paths = [path.Contour for path in poly.Childs]  # Polygon holes
        paths.append(poly.Contour)  # Path holes

        # Append the first point to the end of each path to close loop
        for path in paths:
            path.append(path[0])

        paths = self.scaleFromClipper(paths)

        offsetPolys.append(paths)

        for polyChild in poly.Childs:
            if len(polyChild.Childs) > 0:
                for polyChild2 in polyChild.Childs:
                    offsetPolys += self._getChildPaths(polyChild2)

        return offsetPolys

    def polygonBoundingBox(self, obj) -> np.ndarray:
        """
        Returns the bounding box of the polygon

        :param obj:
        :return: The bounding box of the polygon
        """
        # Path (n,2) coords that

        if not isinstance(obj, list):
            obj = [obj]

        bboxList = []

        for subObj in obj:
            path = np.array(subObj)[:,:2] # Use only coordinates in XY plane
            bboxList.append(np.hstack([np.min(path, axis=0), np.max(path, axis=0)]))

        bboxList = np.vstack(bboxList)
        bbox = np.hstack([np.min(bboxList[:, :2], axis=0), np.max(bboxList[:, -2:], axis=0)])

        return bbox

    def clipLines(self, paths, lines):
        """
        This function clips a series of lines (hatches) across a closed polygon using Pyclipper. Note, the order is NOT
        guaranteed from the list of lines used, so these must be sorted. If order requires preserving this must be
        sequentially performed at a significant computational expense.

        :param paths:
        :param lines: The un-trimmed lines to clip from the boundary

        :return: A list of trimmed lines (paths)
        """

        pc = pyclipper.Pyclipper()

        for path in paths:
            pc.AddPath(self.scaleToClipper(path), pyclipper.PT_CLIP, True)

        # Reshape line list to create n lines with 2 coords(x,y,z)
        lineList = lines.reshape(-1, 2, 3)
        lineList = tuple(map(tuple, lineList))
        lineList = self.scaleToClipper(lineList)

        pc.AddPaths(lineList, pyclipper.PT_SUBJECT, False)

        # Note open paths (lines) have to used PyClipper::Execute2 in order to perform trimming
        result = pc.Execute2(pyclipper.CT_INTERSECTION, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)

        # Cast from PolyNode Struct from the result into line paths since this is not a list
        lineOutput = pyclipper.PolyTreeToPaths(result)

        return self.scaleFromClipper(lineOutput)

    def generateHatching(self, paths, hatchSpacing: float, hatchAngle: float = 90.0):
        """
        Generates un-clipped hatches which is guaranteed to cover the entire polygon region base on the maximum extent
        of the polygon bounding box

        :param paths:
        :param hatchSpacing: Hatch Spacing to use
        :param hatchAngle: Hatch angle (degrees) to rotate the scan vectors

        :return: Returns the list of unclipped scan vectors

        """

        # Hatch angle
        theta_h = np.radians(hatchAngle)  # 'rad'

        # Get the bounding box of the paths
        bbox = self.polygonBoundingBox(paths)

        print('bounding box bbox', bbox)
        # Expand the bounding box
        bboxCentre = np.mean(bbox.reshape(2, 2), axis=0)

        # Calculates the diagonal length for which is the longest
        diagonal = bbox[2:] - bboxCentre
        bboxRadius = np.sqrt(diagonal.dot(diagonal))

        # Construct a square which wraps the radius
        x = np.tile(np.arange(-bboxRadius, bboxRadius, hatchSpacing, dtype=np.float32).reshape(-1, 1), (2)).flatten()
        y = np.array([-bboxRadius, bboxRadius]);
        y = np.resize(y, x.shape)
        z = np.arange(0, x.shape[0]/2, 0.5).astype(np.int64)

        coords = np.hstack([x.reshape(-1, 1),
                            y.reshape(-1, 1),
                            z.reshape(-1,1)]);

        print('coords.', coords.shape)
        # Create the rotation matrix
        c, s = np.cos(theta_h), np.sin(theta_h)
        R = np.array([(c, -s, 0),
                      (s, c, 0),
                      (0, 0, 1.0)])

        # Apply the rotation matrix and translate to bounding box centre
        coords = np.matmul(R, coords.T)
        coords = coords.T + np.hstack([bboxCentre, 0.0])

        return coords

    def clipperToHatchArray(self, coords: np.ndarray) -> np.ndarray:
        """
        A helper method which converts the raw line lists from pyclipper into a array

        :param coords: The list of hatches generated from pyclipper
        """
        return np.transpose(np.dstack(coords), axes=[2, 0, 1])

    @abc.abstractmethod
    def hatch(self,boundaryFeature):
        raise NotImplementedError()


class InnerHatchRegion:

    def __init__(self, parent):

        self._parent = parent
        self._region = []
        raise NotImplementedError()

    def __str__(self):
        return 'InnerHatchRegion <{:s}>'

    @property
    def boundary(self):
        return self._boundary

    def isClipped(self):
        pass


class Hatcher(BaseHatcher):
    """
    Provides a generic SLM Hatcher 'recipe' with standard parameters for defining the hatch across regions. This
    includes generating multiple contour offsets and the generic infill (without) pattern. This class may be derived from
    to provide additional or customised behavior.
    """

    def __init__(self):

        # TODO check that the polygon boundary feature type
        # Contour information
        self._numInnerContours = 1
        self._numOuterContours = 1
        self._spotCompensation = 0.08  # mm
        self._contourOffset = 1 * self._spotCompensation
        self._volOffsetHatch = self._spotCompensation
        self._clusterDistance = 5  # mm

        # Hatch Information
        self._layerAngleIncrement = 0 # 66 + 2 / 3
        self._hatchDistance = 0.08  # mm
        self._hatchAngle = 45
        self._hatchSortMethod = 'alternate'

    """
    Properties for the Hatch Feature
    """

    @property
    def hatchDistance(self) -> float:
        """
        The distance between hatch scan vectors.
        """
        return self._hatchDistance

    @hatchDistance.setter
    def hatchDistance(self, value: float):
        self._hatchDistance = value

    @property
    def hatchAngle(self) -> float:
        """ The base hatch angle used for hatching the region in degrees [-180,180]."""
        return self._hatchAngle

    @hatchAngle.setter
    def hatchAngle(self, value: float):
        self._hatchAngle = value

    @property
    def layerAngleIncrement(self):
        """
        An additional offset used to increment the hatch angle between layers in degrees. This is typically set to
        66.6 Degrees per layer to provide additional uniformity of the scan vectors across multiple layers. By default
        this is set to 0.0."""
        return self._layerAngleIncrement

    @layerAngleIncrement.setter
    def layerAngleIncrement(self, value):
        self._layerAngleIncrement = value

    @property
    def hatchSortMethod(self):
        return self._hatchSortMethod

    @hatchSortMethod.setter
    def hatchSortMethod(self, value):
        self._hatchSortMethod = value

    @property
    def numInnerContours(self) -> int:
        """
        The total number of inner contours to generate by offsets from the boundary region.
        """
        return self._numInnerContours

    @numInnerContours.setter
    def numInnerContours(self, value : int):
        self._numInnerContours = value

    @property
    def numOuterContours(self) -> int:
        """
        The total number of outer contours to generate by offsets from the boundary region.
        """
        return self._numOuterContours

    @numOuterContours.setter
    def numOuterContours(self, value: int):
        self._numOuterContours = value

    @property
    def clusterDistance(self):
        return self._clusterDistance

    @clusterDistance.setter
    def clusterDistance(self, value):
        self._clusterDistance = value

    @property
    def spotCompensation(self) -> float:
        """
        The spot (laser point) compensation factor is the distance to offset the outer-boundary and other internal hatch
        features in order to factor in the exposure radius of the laser.
        """
        return self._spotCompensation

    @spotCompensation.setter
    def spotCompensation(self, value: float):
        self._spotCompensation = value

    @property
    def volumeOffsetHatch(self) -> float:
        """
        An additional offset may be added (positive or negative) between the contour and the internal hatching.
        """
        return self._volOffsetHatch

    @volumeOffsetHatch.setter
    def volumeOffsetHatch(self, value: float):
        self._volOffsetHatch = value

    def hatch(self, boundaryFeature):

        layer = Layer(0.0)
        # First generate a boundary with the spot compensation applied

        offsetDelta = 0.0
        offsetDelta -= self._spotCompensation

        for i in range(self._numOuterContours):
            offsetDelta -= self._contourOffset
            offsetBoundary = self.offsetBoundary(boundaryFeature, offsetDelta)

            for poly in offsetBoundary:
                for path in poly:
                    contourGeometry = ContourGeometry()
                    contourGeometry.coords = np.array(path)[:,:2]
                    contourGeometry.type = "outer"
                    layer.contours.append(contourGeometry)  # Append to the layer

        # Repeat for inner contours
        for i in range(self._numInnerContours):

            offsetDelta -= self._contourOffset
            offsetBoundary = self.offsetBoundary(boundaryFeature, offsetDelta)

            for poly in offsetBoundary:
                for path in poly:
                    contourGeometry = ContourGeometry()
                    contourGeometry.coords = np.array(path)[:,:2]
                    contourGeometry.type = "inner"
                    layer.contours.append(contourGeometry)  # Append to the layer

        # The final offset is applied to the boundary

        offsetDelta -= self._volOffsetHatch

        curBoundary = self.offsetBoundary(boundaryFeature, offsetDelta)

        scanVectors = []

        # Iterate through each closed polygon region in the slice. The currently individually sliced.
        for contour in curBoundary:
            # print('{:=^60} \n'.format(' Generating hatches '))

            paths = contour

            # Hatch angle will change per layer
            # TODO change the layer angle increment
            layerHatchAngle = np.mod(self._hatchAngle + self._layerAngleIncrement, 180)

            # The layer hatch angle needs to be bound by +ve X vector (i.e. -90 < theta_h < 90 )
            if layerHatchAngle > 90:
                layerHatchAngle = layerHatchAngle - 180

            # Generate the un-clipped hatch regions based on the layer hatchAngle and hatch distance
            hatches = self.generateHatching(paths, self._hatchDistance, layerHatchAngle)

            # Clip the hatch fill to the boundary
            clippedPaths = self.clipLines(paths, hatches)

            # Merge the lines together
            if len(clippedPaths) == 0:
                continue

            clippedLines = self.clipperToHatchArray(clippedPaths)

            # Extract only x-y coordinates and sort based on the pseudo-order stored in the z component.
            clippedLines = clippedLines[:,:,:3]
            id = np.argsort(clippedLines[:,0,2])
            clippedLines = clippedLines[id,:,:]

            scanVectors.append(clippedLines)


        if len(clippedLines) > 0:
            # Scan vectors have been

            # Construct a HatchGeometry containg the list of points
            hatchGeom = HatchGeometry()

            # Only copy the (x,y) points from the coordinate array.
            hatchVectors = np.vstack(scanVectors)
            hatchGeom.coords = hatchVectors[:,:,:2]

            layer.hatches.append(hatchGeom)

        return layer


class StripeHatcher(Hatcher):
    """
    The Stripe Hatcher extends the standard Hatcher but generates a set of stripe hatches of a fixed width to cover a region.
    This a common scan strategy adopted by users of EOS systems. This has the effect of limiting the max length of the scan vectors
    across a region in order to mitigate the effects of residual stress. """

    def __init__(self):

        super().__init__()

        self._stripeWidth = 5.0
        self._stripeOverlap = 0.1
        self._stripeOffset = 0.5

    def __str__(self):
        return 'StripeHatcher'

    @property
    def stripeWidth(self) -> float:
        """ The stripe width """
        return self._stripeWidth

    @stripeWidth.setter
    def stripeWidth(self, width):
        self._stripeWidth = width

    @property
    def stripeOverlap(self) -> float:
        """ The length of overlap between adjacent stripes"""
        return self._stripeOverlap

    @stripeOverlap.setter
    def stripeOverlap(self, overlap:float):
        self._stripeOverlap = overlap

    @property
    def stripeOffset(self):
        """ The stripe offset is the relative distance (hatch spacing) to move the scan vectors between adjacent stripes"""
        return self._stripeOffset

    @stripeOffset.setter
    def stripeOffset(self, offset: float):
        self._stripeOffset = offset

    def generateHatching(self, paths, hatchSpacing: float, hatchAngle: float = 90.0):
        """
        Generates un-clipped hatches which is guaranteed to cover the entire polygon region base on the maximum extent
        of the polygon bounding box

        :param paths:
        :param hatchSpacing: Hatch Spacing to use
        :param hatchAngle: Hatch angle (degrees) to rotate the scan vectors

        :return: Returns the list of unclipped scan vectors

        """
        # Hatch angle
        theta_h = np.radians(hatchAngle)  # 'rad'

        # Get the bounding box of the paths
        bbox = self.polygonBoundingBox(paths)

        print('bounding box bbox', bbox)
        # Expand the bounding box
        bboxCentre = np.mean(bbox.reshape(2, 2), axis=0)

        # Calculates the diagonal length for which is the longest
        diagonal = bbox[2:] - bboxCentre
        bboxRadius = np.sqrt(diagonal.dot(diagonal))

        numStripes = int(2 * bboxRadius / self._stripeWidth)+1

        # Construct a square which wraps the radius
        hatchOrder = 0
        coords = []
        for i in np.arange(0,numStripes):
            startX = -bboxRadius + i * (self._stripeWidth) - self._stripeOverlap
            endX = startX + (self._stripeWidth) + self._stripeOverlap

            y = np.tile(np.arange(-bboxRadius + np.mod(i,2) * self._stripeOffset*hatchSpacing,
                                  bboxRadius + np.mod(i,2) * self._stripeOffset*hatchSpacing, hatchSpacing, dtype=np.float32).reshape(-1, 1), (2)).flatten()
            #x = np.tile(np.arange(startX, endX, hatchSpacing, dtype=np.float32).reshape(-1, 1), (2)).flatten()
            x = np.array([startX, endX])
            x = np.resize(x, y.shape)
            z = np.arange(hatchOrder, hatchOrder + y.shape[0] / 2, 0.5).astype(np.int64)

            hatchOrder += x.shape[0] / 2

            coords += [ np.hstack([x.reshape(-1, 1),
                                y.reshape(-1, 1),
                                z.reshape(-1, 1)]) ]

        coords = np.vstack(coords)



        # Create the rotation matrix
        c, s = np.cos(theta_h), np.sin(theta_h)
        R = np.array([(c, -s, 0),
                      (s, c, 0),
                      (0, 0, 1.0)])

        # Apply the rotation matrix and translate to bounding box centre
        coords = np.matmul(R, coords.T)
        coords = coords.T + np.hstack([bboxCentre, 0.0])

        return coords
