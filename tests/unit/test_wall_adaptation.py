import numpy as np

from py_geo_fd.wall_adaptation import get_candidate_trias, np_ray_triangle_intersection


def test_get_candidate_trias_uses_precomputed_centers() -> None:
    trias = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [10.0, 1.0, 0.0]],
        ]
    )
    centers = np.array([[0.33, 0.33, 0.0], [10.33, 0.33, 0.0]])

    selected = get_candidate_trias(trias, pt=np.array([0.0, 0.0, 0.0]), radius=2.0, centers=centers)

    assert selected.shape == (1, 3, 3)
    assert np.allclose(selected[0], trias[0])


def test_np_ray_triangle_intersection_empty_candidates_returns_inf() -> None:
    ray_near = np.array([0.0, 0.0, 0.0])
    ray_dir = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    trias = np.empty((0, 3, 3))

    dists = np_ray_triangle_intersection(ray_near, ray_dir, trias)

    assert dists.shape == (2,)
    assert np.all(np.isinf(dists))
