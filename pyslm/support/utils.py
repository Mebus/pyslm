from typing import List
import numpy as np
import trimesh

from ..core import Part

def getSupportAngles(part: Part) -> np.ndarray:
    """

    :param part:
    """

    # Upward vector for support angles
    v0 = np.array([[0., 0., -1.0]])

    # Identify Support Angles
    v1 = part.geometry.face_normals
    theta = np.arccos(np.clip(np.dot(v0, v1.T), -1.0, 1.0))
    theta = np.degrees(theta).flatten()

    return theta


def getOverhangMesh(part: Part, overhangAngle: float) -> trimesh.Trimesh:
    """
    Gets the overhang mesh from a :class:`Part`.

    :param part:
    :param overhangAngle: The overhang angle in degrees
    :return:
    """
    # Upward vector for support angles
    v0 = np.array([[0., 0., -1.0]])

    # Identify Support Angles
    v1 = part.geometry.face_normals
    theta = np.arccos(np.clip(np.dot(v0, v1.T), -1.0, 1.0))
    theta = np.degrees(theta).flatten()

    supportFaceIds = np.argwhere(theta > 180 - overhangAngle).flatten()

    overhangMesh = trimesh.Trimesh(vertices=part.geometry.vertices,
                                   faces=part.geometry.faces[supportFaceIds])

    return overhangMesh

def approximateSupportMomentArea(part: Part, overhangAngle: float) -> float:
    """
    The support moment area is the project distance from the support surfaces multipled by the area. It gives a two
    parameter component cost function for the support area.

    :param part:
    :param overhangAngle: The overhang angle in degrees
    :return:
    """
    overhangMesh = getOverhangMesh(part, overhangAngle)

    zHeights = overhangMesh.triangles_center[:,2]

    # Use the projected area by flattening the support faces
    overhangMesh.vertices[:,2] = 0.0
    faceAreas = overhangMesh.area_faces
    
    return np.sum(faceAreas*zHeights)

def approximateSupportMapByCentroid(part: Part, overhangeAngle: float, includeTriangleVertices: bool=False) -> float:
    """
    This method to approximate the surface area, projects  a single ray (0,0,-1), form each triangle centroid in the
    overhangeMesh. A self intersection test is made and this is used to calculate the distance from te base-plate (z=0.0)
    which is used to generate a support height map.

    :param part:
    :param overhangeAngle: The overhang angle in degrees
    :return:
    """

    overhangMesh = getOverhangMesh(part, overhangeAngle)

    coords = overhangMesh.triangles_center

    if includeTriangleVertices:
        coords = np.vstack([coords, overhangMesh.vertices])

    ray_dir = np.tile(np.array([[0.,0.,-1.0]]), (coords.shape[0],1))

    # Find the first intersection hit of rays project from the triangle.
    hitLoc, index_ray, index_tri = part.geometry.ray.intersects_location(ray_origins=coords,
                                                                         ray_directions=ray_dir,
                                                                         multiple_hits = False)

    heightMap =  np.zeros((coords.shape[0],1), dtype=np.float)
    heightMap[index_ray] = hitLoc[:,2].reshape(-1,1)
    
    heightMap = np.abs(heightMap - coords[:,2])

    # Project the overhang area
    overhangMesh.vertices[:,2] = 0.0
    faceAreas = overhangMesh.area_faces
    
    return np.sum(faceAreas*heightMap)
