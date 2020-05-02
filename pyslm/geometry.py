import numpy as np

from typing import Any, List, Tuple


class LayerGeometry:
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


class HatchGeometry(LayerGeometry):
    def __init__(self, modelId: int = 0, buildStyleId: int = 0, coords: np.ndarray = None):
        super().__init__(modelId, buildStyleId, coords)
        # print('Constructed Hatch Geometry')

    def __str__(self):
        return 'Hatch Geometry <bid, {:d}, mid, {:d}>'.format(self._bid, self._mid)

    def __len__(self):
        return self.numHatches()

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


class PntsGeometry(LayerGeometry):
    def __init__(self, modelId: int = 0, buildStyleId: int = 0, coords: np.ndarray = None):

        super().__init__(modelId, buildStyleId, coords)

    def numPoints(self):
        """ Number of individual point exposures within the geometry group"""
        return self.pnts.shape[0]

    def __len__(self):
        return self.numPoints()

    def __str__(self):
        return 'Points Geometry'


class Layer:
    """
    Slice Layer is a simple class structure for containing a set of SLM Layer Geometries including
    Contour, Hatch, Point Geometry Types and also the current slice or layer position in z.
    """

    def __init__(self, z):
        self._z = z
        self.id = 0

        self._contours = []
        self._hatches = []
        self._points = []

    # Const Methods Below
    @property
    def z(self):
        """ The Z Position of the Layer"""
        return self._z

    def __len__(self):
        return len(self.getLayerGeometry())

    def __str__(self):
        return 'Layer <z = {:.3f}>'.format(self._z)

    def getLayerGeometry(self) -> List[Any]:
        """
        Returns all the layer geometry groups in the layer.
        """
        return [self._contours, self._hatches, self._points]

    @property
    def contours(self) -> List[ContourGeometry]:
        return self._contours

    @contours.setter
    def contours(self, geoms: List[ContourGeometry]):
        self._contours = geoms

    @property
    def hatches(self) -> List[HatchGeometry]:
        return self._hatches

    @hatches.setter
    def hatches(self, geoms: List[HatchGeometry]):
        self._hatches = geoms

    @property
    def points(self) -> List[PntsGeometry]:
        return self._points

    @points.setter
    def points(self, geoms: List[PntsGeometry]):
        self._points = geoms
