import numpy as np

from typing import List, Tuple

class LayerGeometry:
    def __init__(self):
        self.bid = 0
        self.mid = 0

        self.coords = np.array([])

    def boundingBox(self):
        return np.hstack(np.min(self.coords, axis=0), np.max(self.coords, axis=0))


class HatchGeometry(LayerGeometry):
    def __init__(self):
        LayerGeometry.__init__(self)
        # print('Constructed Hatch Geometry')

    def numHatches(self):
        return self.pnts.shape[0]

    def __len__(self):
        return self.numHatches()

    def __str__(self):
        return 'Hatch Geometry'


class ContourGeometry(LayerGeometry):
    def __init__(self):
        LayerGeometry.__init__(self)

    # print('Constructed Contour Geometry')

    # TODO add some type method
    def numContours(self):
        return self.pnts.shape[0] - 1

    def __len__(self):
        return self.numContours()

    def __str__(self):
        return 'Contour Geometry'


class PntsGeometry(LayerGeometry):
    def __init__(self):
        LayerGeometry.__init__(self)
        # print('Constructed Pnts Geometry')

    def numPoints(self):
        return self.pnts.shape[0]

    def __len__(self):
        return self.numPoints()

    def __str__(self):
        return 'Points Geometry'


class Layer:
    """
    Slice Layer is a simple class structure for containing a set of SLM Layer Geometries
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
        return self._z

    def __str__(self):
        return 'Layer <z = {:.3f}>'.format(self._z)

    @property
    def contours(self) -> List[ContourGeometry]:
        return self._contours

    @contours.setter
    def contours(self, geoms):
         self._contours = geoms

    @property
    def hatches(self) -> List[HatchGeometry]:
        return self._hatches

    @hatches.setter
    def hatches(self, geoms):
         self._hatches = geoms

    @property
    def points(self) -> List[PntsGeometry]:
        return self._hatches

    @points.setter
    def points(self, geoms):
        self._points = geoms



