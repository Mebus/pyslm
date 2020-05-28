import numpy as np

def isValidHatchArray(hatchVectors: np.ndarray) -> bool:
    """ Utility method """
    return hatchVectors.ndim == 2 and (hatchVectors.shape[0] % 2) == 0


def to3DHatchArray(hatchVectors: np.ndarray) -> np.ndarray:
    """
    Utility to reshape a  flat 2D hatch vector array into a 3D array to allow manipulation of individual vectors

    :param hatchVectors: Numpy Array of Hatch Coordinates of shape (2n, 2) where n is the number of of individual hatch vectors
    :return: A view of the hatch vector formatted as 3D array of shape (n,2,2)
    """
    if hatchVectors.ndim != 2:
        raise ValueError('Hatch Vector Shape should be 2D array')

    return hatchVectors.reshape(-1, 2, 2)


def from3DHatchArray(hatchVectors: np.ndarray) -> np.ndarray:
    """
    Utility to reshape a 3D hatch vector array into a flat 2D array to allow manipulation of individual vectors

    :param hatchVectors: Numpy Array of Hatch Coordinates of shape (n, 2, 2) where n is the number of of individual hatch vectors
    :return: A view of the hatch vector formatted as 3D array of shape (2n,2)
    """
    if hatchVectors.ndim != 3:
        raise ValueError('Hatch Vector Shape should be 3D array')

    return hatchVectors.reshape(-1, 2)

