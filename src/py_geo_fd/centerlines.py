from __future__ import annotations
from typing import Optional, Iterable
from pathlib import Path

import warnings, json, vtk
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.integrate import quad
from vtkmodules.util import numpy_support as ns

from py_geo_fd.stent_config import Stent_Config


def normalized(n: np.ndarray, axis=None) -> np.ndarray:
    """
    Normalize vector or array of vectors.

    Args:
        n (numpy.ndarray): The vector or array of vectors to normalize.
        axis (int, optional): The axis along which to normalize. Defaults to None.

    Returns:
        numpy.ndarray: The normalized vector or array of vectors.
    """

    if axis is None:
        return n / np.linalg.norm(n)
    else:
        try:
            return n / np.linalg.norm(n, axis=axis)[:, np.newaxis]
        except:
            return n / np.linalg.norm(n, axis=axis)[:, :, np.newaxis]


def read_centerline_from_vtp(
    filename: str | Path, flip: bool = False
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Read centerline data from a VTP file.

    Parameters:
    - filename (str): The path to the VTP file.
    - flip (bool): Whether to flip the points and radius arrays.

    Returns:
    - points (ndarray): The coordinates of the centerline points.
    - radius (ndarray or None): The radius values of the centerline points, or None if not available.
    """

    filename = str(filename)

    if filename.endswith(".vtp"):
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(filename)
        reader.Update()
    elif filename.endswith(".vtk"):
        reader = vtk.vtkUnstructuredGridReader()
        reader.SetFileName(filename)
        reader.Update()
    elif filename.endswith(".vtu"):
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(filename)
        reader.Update()
    else:
        raise ValueError(f"Unsupported centerline format: {filename}")
    polydata = reader.GetOutput()
    points = ns.vtk_to_numpy(polydata.GetPoints().GetData())

    try:
        radius = ns.vtk_to_numpy(polydata.GetPointData().GetArray(0))
    except AttributeError:
        radius = None

    if flip:
        points = np.flip(points, axis=0)
        if radius is not None:
            radius = np.flip(radius)

    return points, radius


def write_centerline_to_vtp(
    filename: str | Path,
    points: np.ndarray,
    line_type: Optional[str] = "line",
    data: Optional[dict] = None,
) -> None:
    """
    Write centerline data to a VTK PolyData file (.vtp).

    Args:
        filename (str): The path and filename of the output VTP file.
        points (np.ndarray): The array of points representing the centerline.
        line_type (str, optional): The type of line to be written. Defaults to "line".
        data (dict, optional): Additional data to be associated with the centerline.
            The keys of the dictionary represent the data names, and the values are
            the corresponding data arrays. Defaults to None.

    Returns:
        None
    """

    # Create a vtkPoints object and store the points in it
    vtk_points = vtk.vtkPoints()
    # Create a cell array to store the lines in and add the lines to it
    lines = vtk.vtkCellArray()

    # Fill values
    if line_type == "line":
        n_cells = len(points) - 1
        n_points = len(points)
        for i, point in enumerate(points):
            vtk_points.InsertNextPoint(point)
            if i < len(points) - 1:
                line = vtk.vtkLine()
                line.GetPointIds().SetId(0, i)
                line.GetPointIds().SetId(1, i + 1)
                lines.InsertNextCell(line)

    elif line_type == "vector":
        n_cells = len(points[0])
        n_points = 2 * len(points[:, 0])
        for i, (p1, p2) in enumerate(zip(points[0], points[1])):
            vtk_points.InsertNextPoint(p1)
            vtk_points.InsertNextPoint(p2)
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, 2 * i)
            line.GetPointIds().SetId(1, 2 * i + 1)
            lines.InsertNextCell(line)
    else:
        print(f"Error: Unknown line type {line_type}.")

    # Create a polydata to store everything in
    linesPolyData = vtk.vtkPolyData()

    # Add the points to the dataset
    linesPolyData.SetPoints(vtk_points)

    # Add the lines to the dataset
    linesPolyData.SetLines(lines)

    vtk_data = list()

    if isinstance(data, dict):
        for key in data.keys():
            if len(data[key]) == n_points:
                vtk_data = ns.numpy_to_vtk(data[key].astype(float))
                vtk_data.SetName(key)
                linesPolyData.GetPointData().AddArray(vtk_data)
            elif len(data[key]) == n_cells:
                vtk_data = ns.numpy_to_vtk(data[key].astype(float))
                vtk_data.SetName(key)
                linesPolyData.GetCellData().AddArray(vtk_data)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetDataModeToBinary()
    writer.SetFileName(filename)
    writer.SetInputData(linesPolyData)
    writer.Write()
    return


def read_vmtk_centerline(*files: Iterable) -> tuple[np.ndarray, np.ndarray]:
    centerpoints = list()
    radius = list()

    for file in files:
        with open(file, "r") as f:
            data = json.loads(f.read())
            data = data["markups"][0]

        centerpoints.append(np.array([d["position"] for d in data["controlPoints"]]).T)

        try:
            rad = data["measurements"][-1]
            radius.append(np.array(rad["controlPointValues"]))
        except KeyError:
            radius = np.nan * np.ones_like(centerpoints[-1])

    return np.concatenate(centerpoints, axis=1), np.concatenate(radius)


class Curve(object):
    def __init__(
        self,
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        *args,
        **kwargs,
    ):
        """
        Initialize the ParametricCurve with the given data points and smoothing factor.

        Parameters:
        t (np.ndarray): Array of parameter values.
        x (np.ndarray): Array of x-coordinates corresponding to t.
        y (np.ndarray): Array of y-coordinates corresponding to t.
        z (np.ndarray): Array of z-coordinates corresponding to t.
        """
        self.x = UnivariateSpline(t, x, *args, **kwargs)
        self.y = UnivariateSpline(t, y, *args, **kwargs)
        self.z = UnivariateSpline(t, z, *args, **kwargs)

    def __call__(self, t: float | np.ndarray, nu=0) -> np.ndarray:
        """
        Evaluate the curve at given parameter value(s) t.

        Parameters:
        t (float | np.ndarray): Parameter value(s) at which to evaluate the curve.

        Returns:
        np.ndarray: Array of shape (3,) if t is a single value or (len(t), 3) if t is an array,
                    containing [x(t), y(t), z(t)].
        """
        t = np.asarray(t)
        data = [self.x(t, nu=nu), self.y(t, nu=nu), self.z(t, nu=nu)]
        if t.ndim == 0:  # Single value case
            return np.array(data)
        else:  # Array case
            return np.vstack(data).T


class CenterLine(object):
    """Class to handle the vessel centerline. It is called internally by the Stent class.

    Args:
        object (Stent_Config): Stent configuration dictionary.
    """

    def __init__(self, config: Stent_Config) -> None:
        ctrl_path = config.ctrl.path
        if ctrl_path.endswith(".json"):
            centerpoints, _ = read_vmtk_centerline(ctrl_path)
        elif ctrl_path.endswith(".vtp") or ctrl_path.endswith(".vtk"):
            centerpoints, _ = read_centerline_from_vtp(ctrl_path)
            centerpoints = centerpoints.T

        self.init_curve(*centerpoints, config=config)

    def init_curve(self, x, y, z, config: Stent_Config) -> None:
        s = config.ctrl.x_s
        sR = config.ctrl.dx_s

        t = np.linspace(0, 1, len(x))
        self.x = Curve(t, x, y, z, k=3, s=s)
        t = np.array([self.length(0, l) for l in t])
        self.x = Curve(t, x, y, z, k=3, s=s)

        self.s_end = t[-1]
        self.s_start = config.ctrl.s_start
        self.r = None

        # Compute smooth versions of the curves for the numerics
        t = np.linspace(0, self.s_end, 100)
        n, t1, t2 = self.gen_basis(t)
        self.n = Curve(t, n[:, 0], n[:, 1], n[:, 2], k=3, s=0)
        self.t1 = Curve(t, t1[:, 0], t1[:, 1], t1[:, 2], k=3, s=0)
        self.t2 = Curve(t, t2[:, 0], t2[:, 1], t2[:, 2], k=3, s=0)

        R = self.compute_R(t)
        self.R = Curve(t, R[:, 0], R[:, 1], R[:, 2], k=1, s=0)
        self.mag_R = UnivariateSpline(t, np.linalg.norm(R, axis=1), k=1, ext=3, s=sR)
        return

    def compute_R(self, t: np.ndarray) -> np.ndarray:
        """Compute major radius of the curve.

        Args:
            t (np.ndarray): Curve parameter.

        Returns:
            np.ndarray: _description_
        """

        R = self.x(t, nu=2)
        N = self.x(t, nu=1)

        m1 = 1 / np.sum(N**2, axis=1)
        m2 = m1**2 * np.einsum("ki,ki->k", N, R)
        K = m1[:, None] * R - m2[:, None] * N
        b = np.einsum("ki->k", K**2)[:, None]
        R = np.divide(K, b, where=b != 0, out=np.inf * np.ones_like(K))
        return R

    def length(self, a: float = 0.0, b: float = 1.0) -> float:
        def func(t) -> float:
            return np.linalg.norm(self.x(t, nu=1))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            integ = quad(func, a, b)[0]
        return integ

    def gen_basis(self, t: np.ndarray) -> Iterable[np.ndarray]:
        """Compute basis along the centerline.

        Args:
            t (np.ndarray): Curve paramter as a vector.

        Returns:
            Iterable[np.ndarray]: Arrays of basis vectors for each t [n,t1,t2].
        """

        def N(t) -> np.ndarray:
            n = self.x(t, nu=1)
            if n.ndim == 2:
                n /= np.linalg.norm(n, axis=1)[:, None]
            else:
                n /= np.linalg.norm(n)
            return n

        def R(t) -> np.ndarray:
            if isinstance(t, float) or isinstance(t, int):
                t = np.array([t])

            maj_R = self.x(t, nu=2)

            n = N(t)
            m1 = 1 / np.sum(n**2, axis=1)
            m2 = m1**2 * np.einsum("ki,ki->k", n, maj_R)
            K = m1[:, None] * maj_R - m2[:, None] * n
            maj_R = K / np.einsum("ki->k", K**2)[:, None]
            return np.squeeze(maj_R)

        # Compute basis using double reflection method
        t1 = np.zeros((len(t), 3))
        t1[0] = normalized(R(t[0]))

        n = N(t)
        V1 = np.diff(self(t), axis=0)
        C1 = 2 / np.sum(V1 * V1, axis=1)[:, None] * V1
        V2 = n[1:] - n[:-1] + C1 * np.sum(n[:-1] * V1, axis=1)[:, None]
        C2 = 2 / np.sum(V2 * V2, axis=1)[:, None] * V2

        for i in range(len(t) - 1):
            RL = t1[i] - C1[i] * np.sum(t1[i] * V1[i])
            t1[i + 1] = RL - C2[i] * np.sum(RL * V2[i])

        t2 = np.cross(n, t1)

        return n, t1, t2

    def __call__(self, t: float | np.ndarray) -> np.ndarray:
        """Evaluate centerline position.

        Args:
            t (float | np.ndarray): Curve parameter.

        Returns:
            np.ndarray: Centerline positions.
        """
        return self.x(t)

    def output_centerline_info(self, t: np.ndarray, file_path: str | Path) -> None:
        write_centerline_to_vtp(
            file_path,
            self(t),
            line_type="line",
            data={
                "r": self.r(t),
                "t1": self.t1(t),
                "t2": self.t2(t),
                "vec_R": self.R(t),
                "R": self.mag_R(t),
            },
        )
        return
