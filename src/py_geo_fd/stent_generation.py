from __future__ import annotations
from typing import Callable
from pathlib import Path
import sys
import time

import meshio, vtk
import numpy as np
from scipy.integrate import quad, cumulative_trapezoid
from scipy.interpolate import UnivariateSpline, CubicSpline, RectBivariateSpline

from py_geo_fd.centerlines import CenterLine, normalized
from py_geo_fd.stent_meshing import mesh_strut, orient_tetras
from py_geo_fd.wall_adaptation import Adapt_Radius
from py_geo_fd.stent_config import load_config


class ProgressBar(object):
    """Simple terminal progress bar for long-running loops."""

    def __init__(
        self,
        total: int,
        label: str,
        width: int = 28,
        min_interval: float = 0.25,
        enabled: bool | None = None,
    ) -> None:
        self.total = max(int(total), 1)
        self.label = label
        self.width = width
        self.min_interval = min_interval
        self.enabled = sys.stdout.isatty() if enabled is None else enabled
        self.start = time.perf_counter()
        self.last_update = 0.0

        if self.enabled:
            self._draw(0)

    def _draw(self, done: int) -> None:
        done = max(0, min(done, self.total))
        ratio = done / self.total
        filled = int(self.width * ratio)
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = time.perf_counter() - self.start
        msg = (
            f"\r{self.label}: [{bar}] {100.0 * ratio:6.2f}% "
            f"({done}/{self.total})  {elapsed:6.1f}s"
        )
        print(msg, end="", flush=True)

    def update(self, done: int, force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        if force or done >= self.total or (now - self.last_update) >= self.min_interval:
            self._draw(done)
            self.last_update = now

    def close(self) -> None:
        if not self.enabled:
            return
        self._draw(self.total)
        print(flush=True)


def numerical_diff(f: Callable, x: float, h: float = 1e-6) -> float:
    """
    Compute the numerical derivative of a function at a given point.

    Parameters:
        f (function): The function to differentiate.
        x (float): The point at which to compute the derivative.
        h (float, optional): The step size for the numerical approximation. Default is 1e-4.

    Returns:
        float: The numerical derivative of the function at the given point.
    """
    derivative = 8 * (f(x + h) - f(x - h))
    derivative += f(x - 2 * h) - f(x + 2 * h)
    return derivative / (12 * h)


def newton(
    f: Callable,
    df: Callable,
    x0: float,
    tol: float = 1e-5,
    maxiter: int = 1000,
    w: float = 1.0,
) -> float:
    """
    Implement Newton's method for finding the root of a function.

    Parameters:
        f (function): The function for which the root is to be found.
        df (function): The derivative of the function.
        x0 (float): The initial guess for the root.
        tol (float, optional): The tolerance for convergence. Defaults to 1e-6.
        maxiter (int, optional): The maximum number of iterations. Defaults to 50.

    Returns:
        float: The approximate root of the function.
    """

    err = np.inf
    for i in range(maxiter):
        err = f(x0) / df(x0)
        x = x0 - w * err
        if np.abs(err) < tol or np.abs(f(x0)) < tol or np.isnan(err):
            return x
        x0 = x
    print(
        f"Warning: Newtons method did not converge with:\n\t|f(x)| = {np.abs(f(x0))}\n\t|f(x)/df(x)| = {np.abs(err)}\n Returning np.nan."
    )
    return np.nan


def rodrigues_rot(P: np.ndarray, n0: np.ndarray, n1: np.ndarray) -> np.ndarray:
    """RODRIGUES ROTATION
    - Rotate given points based on a starting and ending vector
    - Axis k and angle of rotation theta given by vectors n0,n1
    P_rot = P*cos(theta) + (k x P)*sin(theta) + k*<k,P>*(1-cos(theta))
    """

    # If P is only 1d array (coords of single point), fix it to be matrix
    if P.ndim == 1:
        P = P[np.newaxis, :]

    # Get vector of rotation k and angle theta
    n0 = n0 / np.linalg.norm(n0)
    n1 = n1 / np.linalg.norm(n1)
    k = np.cross(n0, n1)
    if np.linalg.norm(k) < 1e-10:
        return P

    k = k / np.linalg.norm(k)
    theta = np.arccos(np.dot(n0, n1))

    # Compute rotated points
    P_rot = np.zeros((len(P), 3))
    for i in range(len(P)):
        P_rot[i] = (
            P[i] * np.cos(theta)
            + np.cross(k, P[i]) * np.sin(theta)
            + k * np.dot(k, P[i]) * (1 - np.cos(theta))
        )

    return P_rot


class Stent(object):
    """
    Class for generating a stent mesh based on a given configuration.

    Parameters:
        config (str | Path): The configuration object or the path to the json configuration file.

    Methods:
        save_stent: Save the stent mesh to files.
        save_stent_to_vtp: Write centerline data to a VTK PolyData file (.vtp).
        save_envelope: Save the envelope as a vtu file.
        save_centerline: Save the centerline as a vtp file.

    """

    def __init__(self, config_file: str | Path) -> None:

        config = load_config(config_file)

        # Stent centerline params
        self.C = CenterLine(config)

        # Number cpu cores to use to compute the centerline - vessel distances
        self.n_cpu = config.n_cpu  # Number of CPUs

        # Wall distance (that fits the radius of the vessel)
        self.adapt = Adapt_Radius(config, self.C)  # r(t, th)

        # Stent configuration
        self.config = config.st

        # t = np.linspace(2, 9, 20)
        # self.C.output_centerline_info(t=t, file_path="demo/centerline.vtp")

        self.precompute_winding()

        return

    def generate_pts(self) -> np.ndarray:
        """
        Generate the points of the stent centerline and the corresponding
        winding factors and wire angles.
        """
        # Shorthand notation of variables used later:
        # Number of points along the centerline
        N = self.config.num.n_segments

        # Postprocessing buffers
        ls = np.linspace(0, self.config.geom.lw, N)
        ts = self.t(ls)
        thetas = np.array([f(ls) for f in self.th]) % (2 * np.pi)
        positions = self.C(ts)[None, :] + np.einsum(
            "ik,ikj->ikj",
            self.adapt(ts[None, :], thetas),
            np.einsum("ki,jk->jki", self.C.t1(ts), np.cos(thetas))
            + np.einsum("ki,jk->jki", self.C.t2(ts), np.sin(thetas)),
        )
        return positions

    def compute_porosity(self, alphas: np.ndarray) -> np.ndarray:
        """
        Compute the porosity of a stent based on the given wire angles.

        Parameters:
        - alphas: numpy array of wire angles in radians

        Returns:
        - porosity: numpy array of porosity values
        """
        # Wire spacing
        w = self.config.geom.w

        # Wire diameter
        dw = self.config.geom.wire_radius

        # Theoretical porosities
        porosity = (1 - 2 * dw / w / np.sin(2 * alphas)) ** 2

        # Invalid porosities outside the angle range
        alpha_min = np.arcsin(dw / w) / 2
        alpha_max = (np.pi - np.arcsin(dw / w)) / 2
        invalid_porosities = (alphas < alpha_min) | (alphas > alpha_max)
        porosity[invalid_porosities] = np.nan

        return porosity

    def compute_K(self, r: float, R: float) -> float:
        """
        Compute the winding factor K for a cylindrical stent.

        Parameters:
        - r (float): The radial distance from the centerline of the stent.
        - R (float): The radius of the stent.

        Returns:
        - K (float): The computed winding factor.

        Raises:
        - None

        """
        # Wire distance
        w = self.config.geom.w

        # Total number of wires
        Nw = self.config.geom.Nw

        # Radial extension of the stent
        re = w * Nw / (2 * np.pi)

        # (capped) Radius of the stent
        r = np.min((self.config.geom.d_nom / 2, r))

        # Compute the winding factor for a cylindrical stent
        K_cyl_local = 1 / np.sqrt(1 - (r / re) ** 2)

        # If the major radius is much larger than the wire radius,
        # return the winding factor for a cylindrical stent
        if R > 20 * r:
            return K_cyl_local

        # Under-relaxation factor for newtons method based on observations
        under_relaxation = (r / self.config.geom.d_nom) ** 2

        # Define the function to integrate
        def integ(K, r, R):
            def K_comp(x):
                return np.sqrt(1 - ((1 + r / R * np.cos(x)) / K) ** 2)

            return quad(K_comp, 0, 2 * np.pi)[0] / (2 * np.pi)

        K = newton(
            lambda K: r / re - integ(K, r, R),
            lambda K: numerical_diff(lambda k: -integ(k, r=r, R=R), K),
            x0=2 + r / R,
            w=under_relaxation,
        )

        if np.isnan(K) or K < K_cyl_local:
            print(
                f"Warning: Computation of winding factor K did not converge. Using cylindric winding factor: {K_cyl_local}. Make sure to properly smooth the centerline, to ensure a smooth enough curve for numerical root finding."
            )
            return K_cyl_local

        return K

    def compute_theta(self, l: float, th_cly: np.ndarray) -> np.ndarray:
        """
        Compute the angle between the wire and the centerline.

        Parameters:
        - l (float): Length position.
        - th_cly (float): The angle around the centerline.

        Returns:
        - theta (float): The computed angle.
        """

        # Compute porosities for envelope
        t = self.t(l)
        R = self.C.mag_R(t)
        r = self.C.r(t)
        K = self.K(l)
        th_new = np.zeros_like(th_cly)

        t1 = self.C.t1(t)
        t2 = self.C.t2(t)
        vec_R = normalized(self.C.R(t))
        vec_R_dot_t1 = float(np.dot(t1, vec_R))
        vec_R_dot_t2 = float(np.dot(t2, vec_R))

        # Wire distance
        w = self.config.geom.w

        # Total number of wires
        Nw = self.config.geom.Nw

        # Radial extension of the stent
        re = w * Nw / (2 * np.pi)

        # (capped) Radius of the stent
        r = np.min((self.config.geom.d_nom / 2, r))
        curvature_ratio = r / R

        def k_comp(th):
            th = np.asarray(th)
            term = vec_R_dot_t1 * np.cos(th) + vec_R_dot_t2 * np.sin(th)
            local = (1 + curvature_ratio * term) / K

            # Clamp small numerical overshoots that can otherwise break sqrt.
            local = np.clip(local, -1.0, 1.0)
            return np.sqrt(np.clip(1 - local**2, 0.0, None))

        # Precompute cumulative integrals once per axial position to avoid
        # repeatedly calling adaptive quadrature in Newton's iterations.
        n_int = max(512, 8 * len(th_cly))
        th_grid = np.linspace(0.0, 2 * np.pi, n_int)
        k_grid = k_comp(th_grid)
        integ_grid = cumulative_trapezoid(k_grid, th_grid, initial=0.0)
        period = 2 * np.pi
        period_integral = float(integ_grid[-1])

        def integ(th):
            th = float(th)
            n_periods = np.floor(th / period)
            th_mod = th - n_periods * period
            return n_periods * period_integral + np.interp(th_mod, th_grid, integ_grid)

        for k, th_c in enumerate(th_cly):
            th_new[k] = newton(
                lambda th: th_c * r / re - integ(th),
                lambda th: -max(float(k_comp(th)), 1e-12),
                x0=th_c,
                w=0.75,
            )
            if np.isnan(th_new[k]):
                print("Applying cylindric winding on nearly straight segment.")
                return th_cly

        if th_cly[-1] == 2 * np.pi:
            vals = th_new - th_cly
            vals[-1] = 0
            th_cly
            th_new_fct = CubicSpline(
                th_cly,
                vals,
                bc_type="periodic",
            )
        else:
            th_new_fct = CubicSpline(
                np.append(th_cly, 2 * np.pi),
                np.append(th_new - th_cly, 0),
                bc_type="periodic",
            )

        mapped_angle = th_new_fct(th_cly) + th_cly

        return mapped_angle

    def precompute_winding(self, N: int = 200) -> None:
        """
        Generate winding factors and wire angles along the stent centerline.

        Parameters:
        - N (int): The number of points along the centerline for fitting.
        """
        # Shorthand notation of variables used later:
        # Number of points along the centerline
        lw = self.config.geom.lw

        # Step size in wire length
        dl = lw / (N - 1)

        # Total number of wires
        Nw = self.config.geom.Nw

        # Wire distance
        w = self.config.geom.w

        # Total number of wires
        Nw = self.config.geom.Nw

        # Radial extension of the stent
        re = w * Nw / (2 * np.pi)

        # Postprocessing buffers
        ls = np.linspace(0, lw, N + 1, endpoint=True)
        ts = np.zeros(N + 1)
        Ks = np.zeros(N)

        # Starting point
        t = self.C.s_start

        # Starting angles
        theta = np.linspace(0, 2 * np.pi, Nw // 2, endpoint=False)
        theta = np.array([theta, theta]).flatten()
        th_cyl = np.linspace(0, 2 * np.pi, Nw)

        # Winding directions
        dirs = np.ones(Nw // 2)
        dirs = np.array([dirs, -dirs]).flatten()

        prog_k = ProgressBar(N, "Precompute winding K")

        for k in range(N):
            # Get radius and major radius
            r = self.C.r(t)
            R = self.C.mag_R(t)

            # Compute winding factor
            ts[k] = t
            Ks[k] = self.compute_K(r, R)

            # Increase centerline position
            t += dl / Ks[k]
            prog_k.update(k + 1)

        prog_k.close()

        self.effective_length = t - self.C.s_start
        ts[-1] = t
        self.t = UnivariateSpline(ls, ts, s=0, k=2, ext=0)
        self.K = UnivariateSpline(ls[:-1], Ks, s=0, k=2, ext=3)

        # Compute the winding angles
        new_th = np.zeros((N, len(th_cyl)))
        prog_th = ProgressBar(N, "Precompute winding angles")
        for i, l in enumerate(ls[:-1]):
            new_th[i] = self.compute_theta(l, th_cyl)
            prog_th.update(i + 1)
        prog_th.close()
        self.new_th = RectBivariateSpline(ls[:-1], th_cyl, new_th, kx=2, ky=2, s=0)

        # Get structured angles
        thetas = dirs[:, None] * ls[None, :-1] / re + theta[:, None]
        thetas = np.mod(thetas, 2 * np.pi)
        thetas = self.new_th(ls[:-1, None], thetas.T, grid=False).T

        # Map angles to continuous range over wires
        self.th = [
            UnivariateSpline(ls[:-1], np.unwrap(th), s=0, k=2, ext=0) for th in thetas
        ]

        self._print_geometry_summary()

        return

    def _print_geometry_summary(
        self,
        n_axial: int = 120,
        n_radial: int = 120,
    ) -> None:
        """Print deployment summary with key geometric metrics."""

        ls = np.linspace(0, self.config.geom.lw, n_axial)
        t = self.t(ls)
        th = np.linspace(0, 2 * np.pi, n_radial, endpoint=False)

        R = self.C.mag_R(t)[:, None]
        vec_R = normalized(self.C.R(t), axis=-1)
        r = self.adapt(t, th)
        K = self.K(ls)[:, None]

        v = np.einsum("ik,j->ijk", self.C.t1(t), np.cos(th))
        v += np.einsum("ik,j->ijk", self.C.t2(t), np.sin(th))
        term = np.einsum("ijk,ik->ij", v, vec_R)

        alpha_term = np.sqrt(np.clip(K**2 - (1 - r / R * term) ** 2, 0.0, None)) / K
        alpha_term = np.clip(alpha_term, -1.0, 1.0)
        alphas = np.arcsin(alpha_term)
        porosity = self.compute_porosity(alphas)

        diam = 2.0 * r
        alpha_deg = np.rad2deg(alphas)
        metal_coverage = 1.0 - porosity

        finite_diam = np.isfinite(diam)
        finite_alpha = np.isfinite(alpha_deg)
        finite_porosity = np.isfinite(porosity)
        finite_coverage = np.isfinite(metal_coverage)

        print("\nStent deployment summary")
        print(f"  Effective stent length [mm]: {self.effective_length:.2f}")

        if np.any(finite_diam):
            print(
                "  Adapted diameter [mm]: "
                f"min {np.min(diam[finite_diam]):.3f}, "
                f"max {np.max(diam[finite_diam]):.3f}, "
                f"mean {np.mean(diam[finite_diam]):.3f}"
            )

        if np.any(finite_alpha):
            print(
                "  Wire angle alpha [deg]: "
                f"min {np.min(alpha_deg[finite_alpha]):.2f}, "
                f"max {np.max(alpha_deg[finite_alpha]):.2f}, "
                f"mean {np.mean(alpha_deg[finite_alpha]):.2f}"
            )

        if np.any(finite_porosity):
            print(
                "  Porosity [-]: "
                f"min {np.min(porosity[finite_porosity]):.4f}, "
                f"max {np.max(porosity[finite_porosity]):.4f}, "
                f"mean {np.mean(porosity[finite_porosity]):.4f}"
            )

        if np.any(finite_coverage):
            print(
                "  Metal coverage [-]: "
                f"min {np.min(metal_coverage[finite_coverage]):.4f}, "
                f"max {np.max(metal_coverage[finite_coverage]):.4f}, "
                f"mean {np.mean(metal_coverage[finite_coverage]):.4f}"
            )

        print(
            "  Winding factor K [-]: "
            f"min {np.min(K):.4f}, max {np.max(K):.4f}, mean {np.mean(K):.4f}"
        )

        d_nom = self.config.geom.d_nom
        if np.any(finite_diam) and d_nom > 0:
            utilization = diam[finite_diam] / d_nom
            print(
                "  Diameter utilization d/d_nom [-]: "
                f"min {np.min(utilization):.4f}, "
                f"max {np.max(utilization):.4f}, "
                f"mean {np.mean(utilization):.4f}"
            )

    def compute_alpha_discrete(
        self, pts: np.ndarray, ts: np.ndarray, thetas: np.ndarray, tol: float = 1e-3
    ) -> np.ndarray:
        """Compute the angles between wires for the deployed stent after radius adjustment.

        Parameters:
            pts (np.ndarray): Discrete wire points in the shape [Nw, N, 3]
            ctrl_pts (np.ndarray): Discrete centerline points in the shape [N, 3]

        Returns:
            np.ndarry: Angles alpha in the shape [Nw, N]
        """

        if pts.ndim == 2:
            pts = pts[None]

        v1 = np.zeros_like(pts)
        v1[:, 1:-1] = pts[:, 2:] - pts[:, :-2]
        v1[:, 0] = pts[:, 1] - pts[:, 0]
        v1[:, -1] = pts[:, -1] - pts[:, -2]
        v1 = normalized(v1, axis=-1)

        ts_1 = ts - tol
        ts_2 = ts + tol
        v2 = self.C(ts_2)[None, :] + np.einsum(
            "ik,ikj->ikj",
            self.adapt(ts_2[None, :], thetas),
            np.einsum("ki,jk->jki", self.C.t1(ts_2), np.cos(thetas))
            + np.einsum("ki,jk->jki", self.C.t2(ts_2), np.sin(thetas)),
        )
        v2 -= self.C(ts_1)[None, :] + np.einsum(
            "ik,ikj->ikj",
            self.adapt(ts_1[None, :], thetas),
            np.einsum("ki,jk->jki", self.C.t1(ts_1), np.cos(thetas))
            + np.einsum("ki,jk->jki", self.C.t2(ts_1), np.sin(thetas)),
        )
        v2 = normalized(v2, axis=-1)

        alphas = np.arccos(np.sum(v1 * v2, axis=-1))

        return np.squeeze(alphas)

    def save_stent(self, file_path: str | Path) -> None:
        """
        Save the stent mesh to files.

        Parameters:
            file_path (str): The path to save the stent files.

        Returns:
            None
        """

        # Load generation settings
        # Resolution along the wire length
        N = self.config.num.n_segments

        # Total number of wires
        Nw = self.config.geom.Nw

        # Wire resolution (number of points along the wire circumference)
        wire_res = self.config.num.wire_resolution

        # Wire radius
        wire_rad = self.config.geom.wire_radius

        # Generate generic strutt cross-section
        # Angles of the cross-section
        th = np.linspace(0, 2 * np.pi, wire_res, endpoint=False)

        # Points of the cross-section
        circle = wire_rad * np.array([np.cos(th), np.sin(th), 0 * th]).T

        # Generate standart section meshing (to be adapted with indices)
        tetras_segment = mesh_strut(
            range(wire_res + 1), range(wire_res + 1, 2 * (wire_res + 1))
        )

        # Allocate buffers for points coordinates and tetrahedra connectivities
        tetras = np.zeros((Nw, (N - 1), 3 * wire_res, 4), dtype=int)
        points = np.zeros((Nw, N, wire_res + 1, 3), dtype=np.float64)

        # Compute centerpoints of all struts
        wire_lines = self.generate_pts()

        # Count for total number of segments
        n_seg_total = 0

        # Loop over each wire
        prog_wire = ProgressBar(Nw, "Meshing wires")
        for n_wire, wire in enumerate(wire_lines):
            # Get "old" centerpoint and normals
            c = wire[0][None, :]

            # Get orientation of the wire segment
            n = normalized(wire[1] - wire[0])

            # Reorient base cross-section with normal
            base = rodrigues_rot(circle, [0, 0, 1], n)

            # Shift it to the center
            base += c

            # Add first set of points
            points[n_wire, 0] = np.concatenate((c, base))

            # For each segment in a strut
            for n_seg in range(N - 1):

                # Get orientation of the wire segment
                n_new = normalized(wire[n_seg + 1] - wire[n_seg])

                # Rotate and shift base
                base = rodrigues_rot(base - c, n, n_new)
                n = n_new
                c = wire[n_seg + 1][None, :]
                base += c

                # Add points to mesh
                points[n_wire, n_seg + 1] = np.concatenate((c, base))

                # Add connectivities
                tetras[n_wire, n_seg] = tetras_segment + n_seg_total * (wire_res + 1)

                # Increase the total number of segments
                n_seg_total += 1

            n_seg_total += 1
            prog_wire.update(n_wire + 1)

        prog_wire.close()

        # Reshape points and tetras
        points = points.reshape(-1, 3)
        tetras = tetras.reshape(-1, 4)

        # Orient tetrahedra
        tetras = orient_tetras(tetras=tetras, points=points)

        # Write stent to two .t files for immersion
        # Create two mesh objects with points and tetras for
        # each of the winding directions
        tetr = [
            tetras.reshape((Nw, (N - 1), 3 * wire_res, 4))[: Nw // 2].reshape(-1, 4),
            tetras.reshape((Nw, (N - 1), 3 * wire_res, 4))[Nw // 2 :].reshape(-1, 4),
        ]

        for tet, name in zip(tetr, ["_n", "_p"]):
            vtu_file = file_path + name + ".vtu"
            mesh = meshio.Mesh(points=points, cells=[("tetra", tet)])
            meshio.vtu.write(vtu_file, mesh, binary=True)

        # Save stent to vtu file for visualization
        # with post processing quantities
        mesh = meshio.Mesh(
            points,
            [("tetra", tetras)],
        )

        meshio.vtu.write(file_path + ".vtu", mesh, binary=True)
        return

    def save_stent_to_vtp(self, filename: str | Path) -> None:
        """
        Write centerline data to a VTK PolyData file (.vtp).

        Parameters:
            filename (str): The path and filename of the output VTP file.

        Returns:
            None
        """

        points = self.generate_pts()
        # data = {}

        # Create a vtkPoints object and store the points in it
        vtk_points = vtk.vtkPoints()
        # Create a cell array to store the lines in and add the lines to it
        lines = vtk.vtkCellArray()

        pts_per_wire = len(points[0])
        start_id = 0

        # Fill values
        for wire in points:
            for i, point in enumerate(wire):
                vtk_points.InsertNextPoint(point)
                if i < len(wire) - 1:
                    line = vtk.vtkLine()
                    line.GetPointIds().SetId(0, start_id + i)
                    line.GetPointIds().SetId(1, start_id + i + 1)
                    lines.InsertNextCell(line)
            start_id += pts_per_wire

        # Create a polydata to store everything in
        linesPolyData = vtk.vtkPolyData()

        # Add the points to the dataset
        linesPolyData.SetPoints(vtk_points)

        # Add the lines to the dataset
        linesPolyData.SetLines(lines)

        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetDataModeToBinary()
        writer.SetFileName(filename)
        writer.SetInputData(linesPolyData)
        writer.Write()
        return

    def save_envelope(
        self, file_path: str | Path, N: int = 100, Nw: int = 100, dim: int = 2
    ) -> None:
        """
        Save the envelope as a vtu file.

        Parameters:
            file_path (str): The path to save the stent files.
            N (int): The number of points along the stent length for postprocessing.
            Nw (int): The number of points along the stent circumference for postprocessing.
            dim (int, optional): Dimension of envelope. 2 corresponds to a surface in 3D and 3 corresponds to an envelope of the stents thickness. Defaults to 2.

        Returns:
            None
        """

        # Postprocessing buffers
        ls = np.linspace(0, self.config.geom.lw, N)

        # Compute porosities for envelope
        t = self.t(ls)
        th = np.linspace(0, 2 * np.pi, Nw, endpoint=False)
        new_th = np.zeros((N, Nw))
        R = self.C.mag_R(t)[:, None]
        vec_R = normalized(self.C.R(t), axis=-1)
        r = self.adapt(t, th)
        K = self.K(ls)[:, None]

        # Compute angular offset to have the correct local winding
        v = np.einsum("ik,j->ijk", self.C.t1(t), np.cos(th))
        v += np.einsum("ik,j->ijk", self.C.t2(t), np.sin(th))
        term = np.einsum("ijk,ik->ij", v, vec_R)
        term = np.sqrt(K**2 - (1 - r / R * term) ** 2) / K

        # Integrate angles
        alphas = np.arcsin(term)

        porosity = self.compute_porosity(alphas)

        new_th = self.new_th(ls, th, grid=True)

        # Save envelope to vtu file
        self.adapt.output_wall(
            t=t,
            th=new_th,
            file_path=file_path,
            data={
                "alpha": (180 * alphas / np.pi).reshape(-1),
                "porosity": porosity.reshape(-1),
            },
            dim=dim,
        )
        return

    def save_centerline(self, file_path: str | Path, N: int = 50) -> None:
        """
        Save the centerline as a vtp file.

        Parameters:
            file_path (str): The path to save the stent files.
            N (int): The number of points along the stent length for postprocessing.

        Returns:
            None
        """

        if not file_path.endswith(".vtp"):
            file_path += ".vtp"

        # Postprocessing buffers
        ls = np.linspace(0, self.config.geom.lw, N)
        self.C.output_centerline_info(t=self.t(ls), file_path=file_path)
        return
