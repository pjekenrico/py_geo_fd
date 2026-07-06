import numpy as np

from py_geo_fd.stent_meshing import mesh_strut, orient_tetras


def test_mesh_strut_shape_for_small_ring() -> None:
    old_base = np.arange(5)
    new_base = np.arange(5, 10)

    tetras = mesh_strut(old_base, new_base)

    assert tetras.shape == (12, 4)
    assert np.issubdtype(tetras.dtype, np.number)


def test_orient_tetras_preserves_shape_and_indices() -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    tetras = np.array([[0, 2, 1, 3]], dtype=int)

    oriented = orient_tetras(tetras.copy(), points)

    assert oriented.shape == tetras.shape
    assert set(oriented[0].tolist()) == {0, 1, 2, 3}
