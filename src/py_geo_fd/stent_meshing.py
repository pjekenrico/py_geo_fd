from __future__ import annotations
import numpy as np


def mesh_strut(old_base: np.ndarray, new_base: np.ndarray) -> np.ndarray:
    """Generates elements between two sets of points that represent two wire cross-sections.

    Args:
        old_base (np.ndarray): Left point collection composed of indices.
        new_base (np.ndarray): Right point collection composed of indices.

    Returns:
        np.ndarray: Connectivity array of the generated elements.
    """
    n_base = len(old_base) - 1

    tetras = np.zeros((n_base, 3, 4))

    for k in range(1, n_base + 1, 1):
        if k != n_base:
            tetras[k - 1, 0] = [old_base[k], old_base[0], old_base[k + 1], new_base[0]]
            tetras[k - 1, 1] = [
                old_base[k],
                new_base[0],
                old_base[k + 1],
                new_base[k + 1],
            ]
            tetras[k - 1, 2] = [old_base[k], new_base[0], new_base[k + 1], new_base[k]]
        else:
            tetras[k - 1, 0] = [old_base[k], old_base[0], old_base[1], new_base[0]]
            tetras[k - 1, 1] = [old_base[k], new_base[0], old_base[1], new_base[1]]
            tetras[k - 1, 2] = [old_base[k], new_base[0], new_base[1], new_base[k]]

    return tetras.reshape(-1, 4)


def orient_tetras(tetras: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Reorients the tetrahedra in the given array based on the orientation of their vertices.

    Parameters:
        tetras (np.ndarray): Array of tetrahedra indices.
        points (np.ndarray): Array of points representing the vertices of the tetrahedra.

    Returns:
        np.ndarray: The reoriented tetrahedra array.
    """

    init_shape = tetras.shape

    nodes = points[tetras.astype(int)]
    edges = nodes - np.roll(nodes, 1, axis=1)
    cross = np.cross(edges[:, 1], edges[:, 2])
    mask = np.sum(cross * edges[:, 0], axis=1) > 0
    bad_tetra = tetras[mask]
    bad_tetra[:, [0, 1]] = bad_tetra[:, [1, 0]]
    tetras[mask] = bad_tetra.astype(int)

    assert np.all((np.array(tetras.shape) == np.array(init_shape)))
    return tetras
