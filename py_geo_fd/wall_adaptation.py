from __future__ import annotations
from typing import Optional

import numpy as np
from functools import partial
import multiprocessing as mp
from scipy.interpolate import UnivariateSpline, RectBivariateSpline
import meshio, vtk, time
import vtkmodules.util.numpy_support as ns

from py_geo_fd.centerlines import CenterLine
from py_geo_fd.stent_config import Stent_Config


class Timer(object):
    def __init__(self, name: Optional[str] = None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        name = ""
        if self.name:
            name = "%s - " % self.name
        print(name + "Elapsed: %s" % (time.time() - self.tstart))


def cylinder_convolve(
    image: np.ndarray, kernel: np.ndarray = np.ones((3, 3)) / 9
) -> np.ndarray:
    # Get dimensions of the image and kernel
    kernel_height, kernel_width = kernel.shape

    # Pad the image to handle boundary conditions
    padded_image = np.pad(
        image,
        ((0, 0), (kernel_width // 2, kernel_width // 2)),
        mode="wrap",
    )

    padded_image = np.pad(
        padded_image,
        ((kernel_height // 2, kernel_height // 2), (0, 0)),
        mode="edge",
    )

    # Create a view of the image with overlapping windows
    strided_view = np.lib.stride_tricks.sliding_window_view(padded_image, kernel.shape)

    # Perform convolution by computing element-wise multiplication and summing along the last two axes
    convolved = np.tensordot(strided_view, kernel, axes=((2, 3), (0, 1)))

    return convolved


def read_surface_from_vtp(filename: str) -> list[np.ndarray, np.ndarray]:
    """Read a surface file in the vtp format using the vtk library.

    Args:
        filename (str): Path to vtp file.

    Returns:
        points, cells (np.ndarray, np.ndarray): Returns point and cell data.
    """
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    polydata = reader.GetOutput()

    points = ns.vtk_to_numpy(polydata.GetPoints().GetData())
    cells = polydata.GetPolys()
    nCells = cells.GetNumberOfCells()
    array = cells.GetData()
    # This holds true if all polys are of the same kind, e.g. triangles.
    assert array.GetNumberOfValues() % nCells == 0
    nCols = array.GetNumberOfValues() // nCells
    numpy_cells = ns.vtk_to_numpy(array)
    numpy_cells = numpy_cells.reshape((-1, nCols))[:, 1:]

    return points, numpy_cells


def write_envelope_to_vtu(filename: str, points: np.ndarray, data=None) -> None:
    """Mesh the points of the shape [n_l, n_rad] into a triangular surface that represents a cylindric envelope.

    Args:
        filename (str): Path to outfile (can be stl, vtk, vtu, ...)
        points (np.ndarray): points[n_l, n_rad, 3]
        data (np.ndarray, optional): Any point data to add to the file. Defaults to None.
    """

    n_s, n_th = points.shape[:2]
    cells = np.zeros((n_s - 1, 2 * n_th, 3), dtype=np.uint)
    idx = np.arange(n_s * n_th).reshape(n_s, -1)

    cells[:, : n_th - 1, 0] = idx[:-1, :-1]
    cells[:, : n_th - 1, 1] = idx[:-1, 1:]
    cells[:, : n_th - 1, 2] = idx[1:, :-1]

    cells[:, n_th - 1, 0] = idx[:-1, -1]
    cells[:, n_th - 1, 1] = idx[:-1, 0]
    cells[:, n_th - 1, 2] = idx[1:, -1]

    cells[:, n_th:-1, 0] = idx[1:, :-1]
    cells[:, n_th:-1, 1] = idx[1:, 1:]
    cells[:, n_th:-1, 2] = idx[:-1, 1:]

    cells[:, -1, 0] = idx[1:, -1]
    cells[:, -1, 1] = idx[1:, 0]
    cells[:, -1, 2] = idx[:-1, 0]

    meshio.Mesh(
        points.reshape(-1, 3),
        cells=[("triangle", cells.reshape(-1, 3))],
        point_data=data,
    ).write(filename)

    return


def np_ray_triangle_intersection(
    ray_near: np.ndarray, ray_dir: np.ndarray, trias: np.ndarray, eps: float = 1e-6
) -> np.ndarray:
    """
    Möller-Trumbore intersection algorithm in pure python
    trias [N_trias, N_edge, N_dim]
    """

    if ray_dir.ndim == 1:
        ray_dir[None, :, None]
    elif ray_dir.shape[1] == 3:
        ray_dir = ray_dir.T

    N_dir = ray_dir.shape[1]
    ray_dir = ray_dir[None, :, :]
    ray_near = ray_near[None, :]
    ray_near = np.repeat(ray_near[:, :, None], N_dir, axis=2)
    trias = trias[:, :, :, None]

    edge1 = trias[:, 1] - trias[:, 0]
    edge2 = trias[:, 2] - trias[:, 0]
    pvec = np.cross(ray_dir, edge2, axisa=1, axisb=1)
    pvec = np.swapaxes(pvec, 1, 2)

    det = np.sum(edge1 * pvec, axis=1)
    inters = np.abs(det) >= eps

    tvec = ray_near - trias[:, 0]
    u = np.sum(tvec * pvec, axis=1) / det
    inters[np.logical_or(u < 0.0, u > 1.0)] = False

    qvec = np.cross(tvec, edge1, axisa=1, axisb=1)
    qvec = np.swapaxes(qvec, 1, 2)

    v = np.sum(ray_dir * qvec, axis=1) / det
    inters[np.logical_or(v < 0.0, u + v > 1.0)] = False

    dists = np.nan * np.ones((pvec.shape[0], pvec.shape[-1]))
    dists[inters] = np.sum(edge2 * qvec, axis=1)[inters] / det[inters]
    inters[dists < eps] = False
    dists[np.logical_not(inters)] = np.nan
    dists[np.isnan(dists)] = np.inf

    try:
        min_dist = np.squeeze(np.min(dists, axis=0))
    except ValueError:
        min_dist = np.inf * np.ones(ray_near.shape[-1])

    return min_dist


def get_candidate_trias(
    trias: np.ndarray, pt: np.ndarray | float, radius: float
) -> np.ndarray:
    center_pts = np.mean(trias, axis=1)
    candidates = np.where(np.linalg.norm(center_pts - pt[None], axis=1) < radius)
    return trias[candidates]


def distance_wall(
    t: np.ndarray,
    C: CenterLine,
    th: np.ndarray,
    max_r: float,
    triangles: np.ndarray,
) -> np.ndarray:
    """Compute distances between the centerline and the walls.

    Args:
        t (np.ndarray): Positions of centerline.

        C (CenterLine): Centerline object.

        th (np.ndarray): Angular positions to search.

        max_r (float): Maximal distance to consider for scanning.

        triangles (np.ndarray): Array with a structure equivalent to an stl file [n_triangles, n_edges, n_points].

    Returns:
        np.ndarray: Centerline distances.
    """
    direction = (
        C.t1(t)[None, :] * np.cos(th)[:, None] + C.t2(t)[None, :] * np.sin(th)[:, None]
    )

    distances = np_ray_triangle_intersection(
        C(t), direction, get_candidate_trias(triangles, C(t), max_r)
    )

    return distances


class Adapt_Radius(object):
    """
    Fits the radius locally to the vessel surface.

    rad = Adapt_Radius(config: Stent_Config)

    then can be used by stent as:

    stent = Stent(..., radius_adapter=rad)
    """

    def __init__(self, config: Stent_Config, C: CenterLine) -> None:
        self.config = config.wall
        self.C = C
        self.n_cpu = config.n_cpu

        mesh = meshio.read(self.config.path)
        self.trias = mesh.points[mesh.cells_dict["triangle"]]

        max_rad = config.st.geom.d_nom * 0.5
        wire_radius = config.st.geom.wire_radius

        Nw = self.config.n_radial_segments
        N = self.config.n_axial_segments

        th = np.linspace(0, 2 * np.pi, Nw, endpoint=False)
        t = np.linspace(self.C.s_start, self.C.s_end, N)
        dists = np.zeros((N, Nw))

        with Timer("Generated equidistant interpolation points"):
            get_intersection = partial(
                distance_wall,
                C=self.C,
                th=th,
                max_r=2 * max_rad,
                triangles=self.trias,
            )

            with mp.Pool(processes=self.n_cpu) as pool:
                dists = pool.map(get_intersection, t)

        dists = np.array(dists)
        is_far = dists > (max_rad + wire_radius)
        dists[is_far] = max_rad
        dists -= wire_radius + self.config.margin

        if self.config.smoothing > 0:
            smoothings = np.copy(dists)

            for i in range(self.config.smoothing):
                smoothings = np.minimum(cylinder_convolve(smoothings), dists)

            dists = smoothings

        th = np.append(th, 2 * np.pi)
        dists = np.append(dists, dists[:, 0, None], axis=1)
        self.smooth_r = RectBivariateSpline(x=t, y=th, z=dists, kx=3, ky=2, s=0)
        self.C.r = UnivariateSpline(t, np.mean(dists, axis=1), k=1, s=0, ext=3)
        return

    def __call__(self, t, th) -> np.ndarray[np.float64]:
        if isinstance(self.smooth_r, RectBivariateSpline):
            if isinstance(th, np.ndarray) and not isinstance(t, np.ndarray):
                result = self.smooth_r(t, th, grid=False)
            elif (
                isinstance(th, np.ndarray)
                and isinstance(t, np.ndarray)
                and t.ndim == 2
                and th.ndim == 2
            ):
                result = self.smooth_r(t, th, grid=False)

            else:
                result = self.smooth_r(t, th, grid=True)
        else:
            print("Error: Unknow interpolator!")

        return result

    def output_wall(
        self, t: np.ndarray, th: np.ndarray, file_path: str, data: dict = None
    ) -> None:

        if not file_path.endswith(".vtu"):
            file_path += ".vtu"

        if th.ndim > 1:
            radia = self(t[:, None], th)
            directions = np.einsum("ik,ij->ijk", self.C.t1(t), np.cos(th))
            directions += np.einsum("ik,ij->ijk", self.C.t2(t), np.sin(th))
        else:
            radia = self(t, th)
            directions = np.einsum("ik,j->ijk", self.C.t1(t), np.cos(th))
            directions += np.einsum("ik,j->ijk", self.C.t2(t), np.sin(th))

        points = self.C(t)[:, None, :] + np.einsum("ij,ijk->ijk", radia, directions)

        if data is None:
            data = {"r": radia.reshape(-1)}
        else:
            data.update({"r": radia.reshape(-1)})

        write_envelope_to_vtu(file_path, points, data)
        return
