import numpy as np
from scipy.spatial import cKDTree

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


def test_get_candidate_trias_with_tree_matches_center_filter() -> None:
    trias = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [10.0, 1.0, 0.0]],
            [[0.0, 10.0, 0.0], [1.0, 10.0, 0.0], [0.0, 11.0, 0.0]],
        ]
    )
    centers = np.mean(trias, axis=1)
    tree = cKDTree(centers)
    pt = np.array([0.0, 0.0, 0.0])
    radius = 2.0

    by_centers = get_candidate_trias(trias, pt=pt, radius=radius, centers=centers)
    by_tree = get_candidate_trias(trias, pt=pt, radius=radius, tree=tree)

    assert by_tree.shape == by_centers.shape
    assert np.allclose(by_tree, by_centers)


def test_get_candidate_trias_tree_with_center_radii_keeps_large_triangle() -> None:
    # Triangle centroid is farther than radius, but the triangle geometry extends into it.
    trias = np.array(
        [
            [[0.95, 0.0, 0.0], [3.0, 0.0, 0.0], [0.95, 0.2, 0.0]],
        ]
    )
    centers = np.mean(trias, axis=1)
    center_radii = np.max(np.linalg.norm(trias - centers[:, None, :], axis=2), axis=1)
    tree = cKDTree(centers)

    # radius smaller than center distance, but larger than center distance - triangle_radius
    pt = np.array([0.0, 0.0, 0.0])
    radius = 1.0

    by_tree = get_candidate_trias(
        trias,
        pt=pt,
        radius=radius,
        centers=centers,
        tree=tree,
        center_radii=center_radii,
    )

    assert by_tree.shape == (1, 3, 3)
    assert np.allclose(by_tree[0], trias[0])


def test_get_candidate_trias_aabb_filter_removes_far_candidates() -> None:
    trias = np.array(
        [
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]],
            [[5.0, 5.0, 5.0], [5.5, 5.0, 5.0], [5.0, 5.5, 5.0]],
        ]
    )
    centers = np.mean(trias, axis=1)
    tree = cKDTree(centers)
    center_radii = np.max(np.linalg.norm(trias - centers[:, None, :], axis=2), axis=1)
    tri_mins = np.min(trias, axis=1)
    tri_maxs = np.max(trias, axis=1)

    selected = get_candidate_trias(
        trias,
        pt=np.array([0.0, 0.0, 0.0]),
        radius=1.0,
        centers=centers,
        tree=tree,
        center_radii=center_radii,
        tri_mins=tri_mins,
        tri_maxs=tri_maxs,
    )

    assert selected.shape == (1, 3, 3)
    assert np.allclose(selected[0], trias[0])
