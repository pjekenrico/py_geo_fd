import numpy as np

from py_geo_fd.stent_generation import newton


def test_newton_returns_nan_for_zero_derivative() -> None:
    root = newton(lambda x: x + 1.0, lambda x: 0.0, x0=0.0)
    assert np.isnan(root)


def test_newton_solves_simple_quadratic() -> None:
    root = newton(lambda x: x * x - 4.0, lambda x: 2.0 * x, x0=3.0)
    np.testing.assert_allclose(root, 2.0, atol=1e-6)
