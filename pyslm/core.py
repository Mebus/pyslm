import numpy as np
import networkx as nx
import trimesh

from abc import ABC
from typing import Any, List, Tuple


class DocumentObject(ABC):

    def __init__(self, name):
        self._name = name
        self._label = 'Document Object'
        self._attributes = []

    # Attributes are those links to other document objects or properties
    @property
    def attributes(self):
        return self._attributes

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, label):
        self._label = label

    @property
    def name(self):
        return self._name

        # Protected method for assigning the set of Feature Attributes performed in constructor

    def _setAttributes(self, attributes):
        self._attributes = attributes

    def setName(self, name):
        self._name = name


    def boundingBox(self):  # const
        raise NotImplementedError('Abstract  method should be implemented in derived class')


class Document:

    def __init__(self):
        print('Initialising the Document Graph')

        # Create a direct acyclic graph using NetworkX
        self._graph = nx.DiGraph()

    def addObject(self, obj):

        if not issubclass(type(obj), DocumentObject):
            raise ValueError('Feature {:s} is not a Document Object'.format(obj))

        self._graph.add_node(obj)

        for attr in obj.attributes:
            # Add the subfeatures if they do not already exist in the document graph
            if attr is None:
                continue

            self.addObject(attr)

            # Add the depency link between parent and it's child attributes
            self._graph.add_edge(attr, obj)

        # Update the document accordingly
        self.recalculateDocument()

    def getObjectsByType(self, objType):
        objs = []

        for node in list(self._graph):

            # Determine if the document object requires boundary layers in calculation
            if type(node) is objType:
                objs.append(node)

        return objs

    def recalculateDocument(self):

        for node in list(nx.dag.topological_sort(self._graph)):

            # Determine if the document object requires boundary layers in calculation
            if type(node).usesBoundaryLayers():

                for childNode in list(nx.dag.ancestors(self._graph, node)):
                    childNode.setRequiresBoundaryLayers()

    @property
    def head(self):
        graphList = list(nx.dag.topological_sort(self._graph))
        return graphList[-1]

    @property
    def parts(self):

        objs = list(self._graph)
        parts = []

        for obj in objs:
            if issubclass(type(obj), Part):
                parts.append(obj)

        return parts

    @property
    def extents(self):
        # Method for calculating the total bounding box size of the document
        bbox = self.boundingBox
        return np.array([bbox[3] - bbox[0],
                         bbox[4] - bbox[1],
                         bbox[5] - bbox[2]])

    @property
    def partExtents(self):
        bbox = self.partBoundingBox
        return np.array([bbox[3] - bbox[0],
                         bbox[4] - bbox[1],
                         bbox[5] - bbox[2]])

    def getDependencyList(self):
        return list(nx.dag.topological_sort(self._graph))

    @property
    def partBoundingBox(self):
        """
        Returns the bounding box for all the parts. This is needed for calculating the grid
        """
        pbbox = np.vstack([part.boundingBox for part in self.parts])
        return np.hstack([np.min(pbbox[:, :3], axis=0), np.max(pbbox[:, 3:], axis=0)])

    @property
    def boundingBox(self):

        graphList = list(nx.dag.topological_sort(self._graph))
        graphList.reverse()
        return graphList[0].boundingBox

    def drawNetworkGraph(self):
        import networkx.drawing
        nodeLabels = [i.name for i in self._graph]
        networkLabels = dict(zip(self._graph, nodeLabels))
        networkx.drawing.draw(self._graph, labels=networkLabels)
    # networkx.drawing.draw_graphviz(self._graph, labels=networkLabels)


class Part(DocumentObject):
    """
    Part is a solid geometry within the document object tree
    """

    def __init__(self, name):

        super().__init__(name)

        self._geometry = None
        self._bbox = np.zeros((1, 6))

        self._partType = 'Undefined'
        self._origin = np.array((0.0, 0.0, 0.0))

    def __str__(self):
        return 'Part <{:s}>'.format(self.name)

    def setGeometry(self, filename):
        self._geometry = trimesh.load_mesh(filename, use_embree=False, process=True, Validate_faces=False)

        print('Geometry information <{:s}> - [{:s}]'.format(self.name, filename))
        print('\t bounds', self._geometry.bounds)
        print('\t extent', self._geometry.extents)

    @property
    def geometry(self):
        return self._geometry

    @property
    def boundingBox(self):  # const
        if not self._geometry:
            raise ValueError('Geometry was not set')
        else:
            return  self._geometry.bounds.flatten()

    @property
    def partType(self):
        return self._partType

    @property
    def origin(self):
        return self._origin

    @origin.setter
    def origin(self, origin):
        self._origin = origin;

    def getVectorSlice(self, z: float, returnCoordPaths: bool = False) -> Any:
        """
        The vector slice is created by using trimesh to slice the mesh into a polygon

        :param returnCoordPath:
        :param z: Slice z-position
        :return: Vector slice
        """
        if not self._geometry:
            raise ValueError('Geometry was not set')

        if z < self.boundingBox[2] or z > self.boundingBox[5]:
            return []

        transformMat = np.array(([1.0, 0.0, 0.0, 0.0],
                                 [0.0, 1.0, 0.0, 0.0],
                                 [0.0, 0.0, 1.0, 0.0],
                                 [0.0, 0.0, 0.0, 1.0]), dtype=np.float32)

        # Obtain the section through the STL polygon using Trimesh Algorithm (Shapely)
        sections = self._geometry.section(plane_origin=[0, 0, z],
                                          plane_normal=[0, 0, 1])

        if sections == None:
            return []

        # Obtain the 2D Planar Section at this Z-position
        planarSection, transform = sections.to_planar(transformMat)

        if not planarSection.is_closed:
            # Needed in case there are any holes in the stl mesh
            # Repairs the polygon boundary using a merge function built into Trimesh
            planarSection.fill_gaps(planarSection.scale / 100)

        if returnCoordPaths:
            return self.path2DToPathList(planarSection)
        else:
            return planarSection

    def path2DToPathList(self, shape: trimesh.path.Path2D) -> List[np.ndarray]:
        """
        Returns the list of paths and coordinates from a cross-section (i.e. Trimesh Path2D). This is required to be
        done for performing boolean operations and offsetting with the PyClipper package

        :param shape: A Trimesh Path2D representing a cross-section or container of closed polygons
        :return: A list of paths (Numpy Coordinate Arrays) describing fully closed and oriented paths.
        """
        paths = []

        for poly in shape.polygons_full:
            coords = np.array(poly.exterior.coords)
            paths.append(coords)

            for path in poly.interiors:
                coords = np.array(path.coords)
                paths.append(coords)

        return paths

    def getBitmapSlice(self, z, resolution):
        # Get slice returns the current bitmap slice for a mesh at z position
        # Construct a merged grid for this layer (fixed layer)
        gridSize = (self._geometry.extents[:2] / resolution) + 1  # Padded to prevent rounding issues

        sliceImg = np.zeros(gridSize.astype(dtype=np.int), dtype=np.bool)

        # ToDO for now assume an empty slice -> should be a None Type
        if z < self.boundingBox[2] and z > self.boundingBox[4]:
            return sliceImg

        polys = self.getVectorSlice(z)

        gridSize = (self._geometry.extents[:2] / resolution) + 1  # Padded to prevent rounding issues
        sliceImg = np.zeros(gridSize.astype(dtype=np.int), dtype=np.bool)

        for poly in polys:
            bounds = self._geometry.bounds
            localOffset, grid, gridPoints = trimesh.path.polygons.rasterize_polygon(poly, resolution)

            startPos = np.floor((localOffset - bounds[0, :2]) / resolution).astype(np.int)
            endPos = (startPos + grid.shape).astype(np.int)

            sliceImg[startPos[0]:endPos[0], startPos[1]:endPos[1]] += grid

        return sliceImg