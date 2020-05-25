import numpy as np

from enum import Enum
import abc

from typing import Any, List, Tuple


class Header:
    def __init__(self):
        self.filename = ""
        self.version = (0,0)
        self.zUnit = 1000

class BuildStyle:

    def __init__(self):
        self.bid = 0
        self.laserPower = 0.0
        self.laserSpeed = 0.0
        self.laserFocus = 0.0
        self.pointDistance = 0
        self.pointExposureTime = 0.0

    def setStyle(self, bid, focus, power,
                 pointExposureTime, pointExposureDistance, speed = 0.0):

        self.bid = bid
        self.laserFocus = focus
        self.laserPower = power
        self.pointExposureTime = pointExposureTime
        self.pointDistance = pointExposureDistance
        self.laserSpeed = 0.0

class Model:

    def __init__(self):
        self.mid = 0
        self.topLayerId = 0
        self.name = ""
        self.buildStyles = []

    def __len__(self):
        return len(self.buildStyles)


class LayerGeometryType(Enum):
    Invalid = 0
    Polygon = 1
    Hatch = 2
    Pnts = 3


class LayerGeometry(abc.ABC):
    def __init__(self, modelId: int = 0, buildStyleId: int = 0, coords: np.ndarray = None):
        self._bid = buildStyleId
        self._mid = modelId

        self._coords = np.array([])

        if coords:
            self.coords = coords

    def boundingBox(self) -> np.ndarray:
        return np.hstack(np.min(self.coords, axis=0), np.max(self.coords, axis=0))

    @property
    def coords(self) -> np.ndarray:
        """ Coordinate data stored by the Layer Geometry Group"""
        return self._coords

    @coords.setter
    def coords(self, coordValues: np.ndarray):
        print(coordValues.shape)
        if coordValues.shape[-1] != 2:
            raise ValueError('Coordinates provided to layer geometry must have (X,Y) values only')

        self._coords = coordValues

    @property
    def mid(self) -> int:
        """
        The Model Id for the layer geometry group. The Model Id refers to the collection of unique build-styles assigned
        to a part within a build.
        """
        return self._mid

    @mid.setter
    def mid(self, modelId: int):
        self._mid = modelId

    @property
    def bid(self) -> int:
        """
        The Build Style Id for the layer geometry group. The Build Style Id refers to the collection of laser parameters
        used during scanning of scan vector group.
        """
        return self._bid

    @bid.setter
    def bid(self, buildStyleId: int):
        self._bid = buildStyleId

    @abc.abstractmethod
    def type(self):
        return LayerGeometryType.Invalid


class HatchGeometry(LayerGeometry):
    def __init__(self, modelId: int = 0, buildStyleId: int = 0, coords: np.ndarray = None):
        super().__init__(modelId, buildStyleId, coords)
        # print('Constructed Hatch Geometry')

    def __str__(self):
        return 'Hatch Geometry <bid, {:d}, mid, {:d}>'.format(self._bid, self._mid)

    def __len__(self):
        return self.numHatches()

    def type(self):
        return LayerGeometryType.Hatch

    def numHatches(self) -> int:
        """
        Number of hatches within this layer geometry set
        """
        return self.pnts.shape[0]


class ContourGeometry(LayerGeometry):
    def __init__(self, modelId: int = 0, buildStyleId: int = 0, coords: np.ndarray = None):
        super().__init__(modelId, buildStyleId, coords)

    # print('Constructed Contour Geometry')

    # TODO add some type method
    def numContours(self) -> int:
        """
        Number of contour vectos in the geometry group.
        """
        return self.pnts.shape[0] - 1

    def __len__(self):
        return self.numContours()

    def __str__(self):
        return 'Contour Geometry'

    def type(self):
        return LayerGeometryType.Polygon


class PointsGeometry(LayerGeometry):
    def __init__(self, modelId: int = 0, buildStyleId: int = 0, coords: np.ndarray = None):

        super().__init__(modelId, buildStyleId, coords)

    def numPoints(self):
        """ Number of individual point exposures within the geometry group"""
        return self.pnts.shape[0]

    def __len__(self):
        return self.numPoints()

    def __str__(self):
        return 'Points Geometry'

    def type(self):
        return LayerGeometryType.Pnts

class ScanMode(Enum):
    Default = 0
    ContourFirst = 1
    HatchFirst = 2

class Layer:
    """
    Slice Layer is a simple class structure for containing a set of SLM Layer Geometries including
    Contour, Hatch, Point Geometry Types and also the current slice or layer position in z.
    """

    def __init__(self, z = 0, id = 0):
        self._z = z
        self._id = 0
        self._geometry = []
        self._name = ""

    @property
    def name(self):
        """ The Z Position of the Layer"""
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def layerId(self):
        """ The Z Position of the Layer"""
        return self._id

    @layerId.setter
    def layerId(self, id):
        self._id = id

    @property
    def z(self):
        """ The Z Position of the Layer"""
        return self._z

    @z.setter
    def z(self, z):
        """ The Z Position of the Layer"""
        self._z = z

    def __len__(self):
        return len(self._geometry)

    def __str__(self):
        return 'Layer <z = {:.3f}>'.format(self._z)

    def appendGeometry(self, geom: LayerGeometry):
        """
        Complimentary method to match libSLM API

        :param geom: The LayerGeometry to add to the layer
        """

        self._geometry.append(geom)

    def getGeometry(self, scanMode: ScanMode = ScanMode.Default) -> List[Any]:
        """
        Contains all the layer geometry groups in the layer.
        """
        geoms = []

        if scanMode is ScanMode.ContourFirst:
            geoms += self.getContourGeometry()
            geoms += self.getHatchGeometry()
            geoms += self.getPointsGeometry()
        elif scanMode is ScanMode.HatchFirst:
            geoms += self.getHatchGeometry()
            geoms += self.getContourGeometry()
            geoms += self.getPointsGeometry()
        else:
            geoms = self._geometry

        return geoms

    @property
    def geometry(self) -> List[Any]:
        """
        Contains all the layer geometry groups in the layer.
        """

        return self._geometry

    @geometry.setter
    def geometry(self, geoms: List[LayerGeometry]):
        self._geometry = geoms

    def getContourGeometry(self) -> List[HatchGeometry]:

        geoms = []
        for geom in self._geometry:
            if isinstance(geom, ContourGeometry):
                geoms.append(geom)

        return geoms

    def getHatchGeometry(self) -> List[HatchGeometry]:

        geoms = []
        for geom in self._geometry:
            if isinstance(geom, HatchGeometry):
                geoms.append(geom)

        return geoms

    def getPointsGeometry(self) -> List[PointsGeometry]:

        geoms = []
        for geom in self._geometry:
            if isinstance(geom, PointsGeometry):
                geoms.append(geom)

        return geoms

